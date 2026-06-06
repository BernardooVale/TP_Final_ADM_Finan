"""
simulador_portfolio.py
======================
Simulador de alocação ótima entre Renda Fixa (RF) e Renda Variável (RV).

Objetivo
--------
Dado um capital inicial, determina quanto alocar em RF e RV de forma que,
mesmo no pior cenário simulado da RV, o patrimônio final cubra o capital inicial.

Fluxo principal
---------------
1. Valida e baixa histórico dos tickers via yfinance.
2. Calibra distribuição t-Student por MLE para cada ativo (captura caudas gordas).
3. Simula N cenários × D dias via Monte Carlo com correlação entre ativos (Cholesky).
4. Estima a perda esperada da RV no percentil de confiança escolhido.
5. Resolve algebricamente a alocação RF/RV que cobre essa perda.
6. Opcionalmente: otimiza pesos da RV minimizando CVaR, rebalanceia periodicamente
   e incorpora aportes periódicos com rendimento por ciclos completos do título.

Dependências
------------
    pip install numpy pandas yfinance scipy
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.optimize import minimize_scalar
from numba import njit, prange

from mineracao import baixar_retornos
from defs import RiscoAlvo, FrequenciaRentabilidadeRendaFix, FrequenciaAporte, DIAS_UTEIS_APORTE, DIAS_UTEIS_RF, AlocacaoResultado

# ═══════════════════════════════════════════════════════════════
# Calibração estatística
# ═══════════════════════════════════════════════════════════════

def _calibrar_t_student(serie: pd.Series) -> tuple[float, float, float, float, float, float]:
    """
    Ajusta distribuição t-Student aos retornos de um ativo via MLE
    e calibra parâmetros GARCH(1,1) sobre os resíduos padronizados.

    GARCH(1,1)
    ----------
    sigma2_t = omega + alpha * epsilon2_{t-1} + beta * sigma2_{t-1}

    - omega : variância base (piso)
    - alpha : peso do choque recente (reatividade)
    - beta  : peso da variância anterior (persistência)
    - alpha + beta < 1 garante estacionariedade — variância reverte à média

    Retorna (nu, mu, sigma, omega, alpha, beta).
    sigma aqui é o desvio incondicional (longo prazo), usado como sigma2_0.
    """
    nu, mu, sigma = stats.t.fit(serie)

    # Resíduos padronizados: remove média e escala pela volatilidade incondicional
    residuos = (serie.values - mu) / sigma          # shape (T,)

    # Variância incondicional de longo prazo: sigma2_bar = omega / (1 - alpha - beta)
    # Usada para inicializar sigma2_0 na simulação
    sigma2_bar = sigma ** 2

    def log_verossimilhanca_negativa(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10                             # penaliza parâmetros inválidos

        T       = len(residuos)
        sigma2  = np.empty(T)
        sigma2[0] = sigma2_bar                      # inicializa com variância incondicional

        for t in range(1, T):
            sigma2[t] = omega + alpha * residuos[t-1]**2 + beta * sigma2[t-1]

        # Log-verossimilhança da normal padrão (resíduos já padronizados)
        ll = -0.5 * np.sum(np.log(sigma2) + residuos**2 / sigma2)
        return -ll                                  # negativa pois minimize()

    # Ponto inicial: omega pequeno, alpha+beta moderado e estacionário
    w0     = [sigma2_bar * 0.05, 0.1, 0.85]
    bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.9999)]

    res = optimize.minimize(
        log_verossimilhanca_negativa, w0,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    omega, alpha, beta = res.x if res.success else (sigma2_bar * 0.05, 0.1, 0.85)
    return nu, mu, sigma, float(omega), float(alpha), float(beta)

def _calibrar_nu_copula(retornos: pd.DataFrame) -> float:
    """
    Estima nu da cópula-t via MLE 1D sobre distâncias de Mahalanobis.

    Fluxo:
    1. Transforma cada série em uniformes via rank (empírico CDF)
    2. Transforma uniformes em normais padrão (probability integral transform)
    3. Computa distância de Mahalanobis: d_t = x_t^T Sigma^{-1} x_t
       que segue qui-quadrado escalonado sob t-multivariada
    4. MLE 1D sobre essas distâncias para estimar nu

    Mais preciso que proxy marginal — captura dependência conjunta de cauda
    sem custo de MLE multivariada exata (O(k³) por iteração).
    """
    n, k = retornos.shape

    # Passos 1-2: uniformes empíricas → normais padrão (igual à versão anterior)
    uniformes = retornos.rank() / (n + 1)
    normais   = stats.norm.ppf(uniformes.values)          # (n, k)

    # Passo 3: matriz de correlação empírica e sua inversa
    corr_emp = np.corrcoef(normais, rowvar=False)         # (k, k)
    
    # Regularização leve para garantir invertibilidade
    corr_emp += np.eye(k) * 1e-6
    corr_inv = np.linalg.inv(corr_emp)                   # (k, k)

    # Distância de Mahalanobis ao quadrado para cada observação
    # d_t = x_t^T Sigma^{-1} x_t — escalar por observação
    maha = np.einsum('ti,ij,tj->t', normais, corr_inv, normais)  # (n,)

    # Passo 4: MLE 1D — sob t-multivariada com nu graus de liberdade,
    # (nu/k) * d_t ~ F(k, nu) ou equivalente: d_t * nu/k ~ chi2(k) escalonado
    # Mais direto: log-verossimilhança da t-multivariada marginalizada em d_t
    def neg_ll(nu: float) -> float:
        # d_t * nu / k ~ chi2(nu) sob H0 — MLE sobre essa estatística
        scale = nu / k
        return -float(np.sum(stats.chi2.logpdf(maha * scale, df=nu)))

    res = minimize_scalar(neg_ll, bounds=(2.1, 50.0), method="bounded")

    nu_copula = float(res.x) if res.success else 10.0    # fallback conservador
    print(f"  nu cópula calibrado: {nu_copula:.2f} {'(caudas pesadas)' if nu_copula < 10 else '(próximo gaussiana)'}")
    return nu_copula

def _calibrar_todos(
    retornos: pd.DataFrame,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:

    fitted = [_calibrar_t_student(retornos[t]) for t in tickers]
    nus    = np.array([f[0] for f in fitted])
    mus    = np.array([f[1] for f in fitted])
    sigmas = np.array([f[2] for f in fitted])
    omegas = np.array([f[3] for f in fitted])
    alphas = np.array([f[4] for f in fitted])
    betas  = np.array([f[5] for f in fitted])

    corr      = retornos.corr().values
    nu_copula = _calibrar_nu_copula(retornos)             # passa DataFrame já filtrado

    for t, o, a, b in zip(tickers, omegas, alphas, betas):
        print(f"  GARCH {t}: omega={o:.2e}  alpha={a:.4f}  beta={b:.4f}  persistência={a+b:.4f}")

    return nus, mus, sigmas, corr, nu_copula, omegas, alphas, betas

# ═══════════════════════════════════════════════════════════════
# Conversão de taxa de RF
# ═══════════════════════════════════════════════════════════════

def _taxa_diaria_rf(rate: float, frequencia: FrequenciaRentabilidadeRendaFix) -> float:
    """
    Converte a taxa da RF para equivalente diária via juros compostos.

    Não altera o valor final — apenas universaliza o código para aceitar
    taxas em qualquer frequência. Ex: 14,5% a.a. → ~0,054% a.d.

    Fórmula: taxa_diaria = (1 + rate)^(1/n) - 1
    onde n = dias úteis da frequência informada.
    """
    n = DIAS_UTEIS_RF[frequencia]
    return (1 + rate) ** (1 / n) - 1


# ═══════════════════════════════════════════════════════════════
# Aportes periódicos
# ═══════════════════════════════════════════════════════════════

def _valor_futuro_aportes(
    valorAporte: float,
    frequenciaAporte: FrequenciaAporte,
    diasInvestimento: int,
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
) -> float:
    """
    Calcula o valor futuro total dos aportes periódicos investidos em RF.

    Regra de rendimento
    -------------------
    Um aporte só rende em ciclos COMPLETOS da frequência do título RF.
    Dias restantes que não fecham um ciclo ficam parados (valor nominal).

    Isso reflete a realidade de títulos como CDB/LCI/LCA que pagam
    apenas no vencimento ou em datas fixas de cupom.

    Exemplo
    -------
    RF trimestral (63 dias), aporte mensal, horizonte 252 dias:
      - Aporte no dia 21 → 231 dias restantes → floor(231/63) = 3 ciclos
        → valor futuro = aporte × (1 + taxa_trimestral)^3
      - Aporte no dia 42 → 210 dias restantes → floor(210/63) = 3 ciclos
      - Aporte no dia 63 → 189 dias restantes → floor(189/63) = 3 ciclos
      - Aporte no dia 84 → 168 dias restantes → floor(168/63) = 2 ciclos

    Caso especial: RF diária → ciclo = 1 → rende todo dia (comportamento contínuo).
    """
    intervalo = DIAS_UTEIS_APORTE[frequenciaAporte]
    ciclo_rf  = DIAS_UTEIS_RF[frequenciaRendaFixa]

    total = 0.0
    # Itera pelos dias de aporte (exclui dia 0, que é o capital inicial)
    for d in range(intervalo, diasInvestimento, intervalo):
        dias_restantes   = diasInvestimento - d
        ciclos_completos = dias_restantes // ciclo_rf  # floor implícito da divisão inteira
        diaria = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)
        taxa_ciclo = (1 + diaria) ** ciclo_rf - 1
        total += valorAporte * (1 + taxa_ciclo) ** ciclos_completos

    return total


# ═══════════════════════════════════════════════════════════════
# Simulação Monte Carlo
# ═══════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def _garch_scan(
    epsilon: np.ndarray,
    omegas:  np.ndarray,
    alphas:  np.ndarray,
    betas:   np.ndarray,
    mus:     np.ndarray,
    sigmas:  np.ndarray,
) -> np.ndarray:
    """
    Evolução GARCH(1,1) vetorizada via Numba.

    Loop externo (cenários) paralelizado com prange — cada cenário é
    independente, zero contenção. Loop interno (dias) sequencial por
    necessidade: sigma2[t] depende de sigma2[t-1].

    sigma2_0 = variância incondicional: omega / (1 - alpha - beta)

    Compilado em C na primeira chamada (cache=True evita recompilação
    entre sessões).
    """
    S, D, A   = epsilon.shape
    retornos  = np.empty((S, D, A))

    for s in prange(S):                                  # paralelo entre cenários
        # sigma2 inicial: variância incondicional por ativo
        sigma2 = np.empty(A)
        for a in range(A):
            denom    = 1.0 - alphas[a] - betas[a]
            sigma2[a] = omegas[a] / denom if denom > 1e-8 else sigmas[a] ** 2

        for d in range(D):                               # sequencial — dependência temporal
            for a in range(A):
                sigma_t            = sigma2[a] ** 0.5
                retornos[s, d, a]  = mus[a] + sigma_t * epsilon[s, d, a]
                choque             = (sigma_t * epsilon[s, d, a]) ** 2
                sigma2[a]          = omegas[a] + alphas[a] * choque + betas[a] * sigma2[a]

    return retornos

def _simular_retornos_diarios(
    mus:              np.ndarray,
    sigmas:           np.ndarray,
    nus:              np.ndarray,
    chol:             np.ndarray,
    n_sim:            int,
    diasInvestimento: int,
    n_assets:         int,
    z_fixo:           np.ndarray | None,
    nu_copula:        float = 30.0,
    omegas:           np.ndarray | None = None,
    alphas:           np.ndarray | None = None,
    betas:            np.ndarray | None = None,
) -> np.ndarray:
    """
    Gera retornos diários correlacionados com cópula-t e volatilidade GARCH(1,1).

    Fluxo
    -----
    1. Gera z normal padrão (S, D, A)
    2. Cópula-t: divide z por sqrt(chi2_compartilhado / nu_copula)
    3. Correlaciona via Cholesky: y = t_copula @ L^T
    4. Transforma y para uniformes via CDF_t(nu_copula)
    5. Aplica PPF marginal t-Student de cada ativo → inovações epsilon (S, D, A)
    6. GARCH: _garch_scan evolui sigma2_t em loop compilado (Numba)
       Sem GARCH: sigma fixo, caminho numpy puro (retrocompatível)
    """
    S, D, A = n_sim, diasInvestimento, n_assets

    z = z_fixo if z_fixo is not None else np.random.standard_normal((S, D, A))

    # ── Cópula-t ──
    chi2_shared = np.random.chisquare(nu_copula, size=(S, D, 1)) / nu_copula
    t_copula    = z / np.sqrt(chi2_shared)
    y           = t_copula @ chol.T

    y_flat = y.reshape(-1, A)
    u_flat = np.empty_like(y_flat)
    for i in range(A):
        u_flat[:, i] = stats.t.cdf(y_flat[:, i], df=nu_copula)

    x_flat = np.empty_like(u_flat)
    for i in range(A):
        x_flat[:, i] = stats.t.ppf(u_flat[:, i], df=nus[i], loc=0.0, scale=1.0)

    epsilon = x_flat.reshape(S, D, A)

    # ── Sem GARCH: caminho original ──
    if omegas is None or alphas is None or betas is None:
        return epsilon * sigmas[np.newaxis, np.newaxis, :] + mus[np.newaxis, np.newaxis, :]

    # ── Com GARCH: loop compilado ──
    return _garch_scan(
        epsilon,
        omegas.astype(np.float64),
        alphas.astype(np.float64),
        betas.astype(np.float64),
        mus.astype(np.float64),
        sigmas.astype(np.float64),
    )

def _cholesky_seguro(corr: np.ndarray) -> np.ndarray:
    """
    Tenta Cholesky direto. Se falhar (matriz não-positiva-definida por
    outliers ou multicolinearidade), aplica correção de Higham via
    eigendecomposição — clipa autovalores negativos e reconstrói.
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        print("  ⚠ Correlação não-positiva-definida — aplicando correção de Higham")
        vals, vecs = np.linalg.eigh(corr)
        vals       = np.maximum(vals, 1e-8)          # clipa autovalores negativos
        corr_fixed = vecs @ np.diag(vals) @ vecs.T
        # Renormaliza diagonal para 1 (preserva estrutura de correlação)
        d          = np.sqrt(np.diag(corr_fixed))
        corr_fixed = corr_fixed / np.outer(d, d)
        return np.linalg.cholesky(corr_fixed)

def monteCarlo(
    mus: np.ndarray,
    sigmas: np.ndarray,
    nus: np.ndarray,
    matrizCorrelacao: np.ndarray,
    proporcaoAcao: np.ndarray,
    diasInvestimento: int,
    numSimulacoes: int = 1_000_000,
    diasRebalanceamento: int | None = None,
    z_fixo: np.ndarray | None = None,
    chunk_size: int = 50_000,
    nu_copula: float = 30.0,
    omegas=None,
    alphas=None,
    betas=None
) -> np.ndarray:
    
    chol     = _cholesky_seguro(matrizCorrelacao)
    n_assets = len(mus)
    resultados = []

    for start in range(0, numSimulacoes, chunk_size):
        end    = min(start + chunk_size, numSimulacoes)
        n_chunk = end - start

        # z_fixo só faz sentido na otimização (mesmo ruído entre iterações)
        # em simulação normal, cada chunk gera seu próprio ruído
        z_chunk = (
            z_fixo[start:end] if z_fixo is not None
            else None
        )

        r_diario = _simular_retornos_diarios(
            mus, sigmas, 
            nus, chol, 
            n_chunk, diasInvestimento, 
            n_assets, z_chunk, 
            nu_copula, omegas, 
            alphas, betas
        )

        if diasRebalanceamento is None:
            portfolio_diario = r_diario @ proporcaoAcao
            resultados.append(np.prod(1 + portfolio_diario, axis=1) - 1)
        else:
            pesos = np.tile(proporcaoAcao, (n_chunk, 1)).astype(float)
            valor = np.ones(n_chunk)
            for d in range(diasInvestimento):
                r_d    = r_diario[:, d, :]
                valor *= 1 + (pesos * r_d).sum(axis=1)
                pesos  = pesos * (1 + r_d)
                pesos /= pesos.sum(axis=1, keepdims=True)
                if (d + 1) % diasRebalanceamento == 0:
                    pesos[:] = proporcaoAcao
            resultados.append(valor - 1)

    return np.concatenate(resultados)


# ═══════════════════════════════════════════════════════════════
# Métricas de risco/retorno
# ═══════════════════════════════════════════════════════════════

def _retorno_cenario_alvo(
    retornosCumulativos: np.ndarray,
    confianca: float,
    riscoAlvo: RiscoAlvo,
) -> float:
    """
    Estima a perda esperada da RV no tail de risco.

    Filtra os cenários abaixo do limiar (pior (1-confiança)% dos resultados)
    e retorna a média ou o mínimo desse grupo, conforme `riscoAlvo`.
    """
    limiar   = np.percentile(retornosCumulativos, (1 - confianca) * 100)
    cenarios = retornosCumulativos[retornosCumulativos <= limiar]

    if not len(cenarios):
        return 0.0

    return float(cenarios.mean()) if riscoAlvo == RiscoAlvo.MEDIA else float(cenarios.min())


def _calcular_sharpe_sortino(
    retornosCumulativos: np.ndarray,
    retorno_rf_periodo: float,
) -> tuple[float, float]:
    """
    Calcula Sharpe e Sortino da carteira simulada.

    Sharpe  = retorno excedente médio / desvio padrão total
    Sortino = retorno excedente médio / desvio padrão apenas dos retornos negativos

    O Sortino é mais justo: volatilidade positiva não deveria ser penalizada.
    Ambos usam o retorno da RF no período como benchmark.
    """
    excesso = retornosCumulativos - retorno_rf_periodo
    sharpe  = excesso.mean() / (excesso.std() + 1e-12)

    # Downside deviation: desvio apenas dos cenários abaixo da RF
    downside = excesso[excesso < 0]
    dd_std   = downside.std() if len(downside) > 1 else 1e-12
    sortino  = excesso.mean() / (dd_std + 1e-12)

    return float(sharpe), float(sortino)


# ═══════════════════════════════════════════════════════════════
# Cálculo de alocação RF/RV
# ═══════════════════════════════════════════════════════════════

def _resolver_alocacao(
    capitalTotal: float,
    crescimentoRF: float,
    retornoAcoes: float,
) -> tuple[float, float]:
    """
    Resolve algebricamente quanto alocar em RF e RV.

    Objetivo: RF_final + RV_final >= capitalTotal no cenário alvo.

    Sendo:
        RF_final = alocRF × crescimentoRF
        RV_final = alocRV × (1 + retornoAcoes)
        alocRV   = capitalTotal - alocRF

    Isolando alocRF:
        alocRF = capitalTotal × (-retornoAcoes) / (crescimentoRF - (1 + retornoAcoes))

    Resultado clipado em [0, capitalTotal] para evitar alocações negativas.
    """
    denom     = crescimentoRF - (1 + retornoAcoes)
    alocRF    = capitalTotal * (-retornoAcoes) / denom
    alocRF    = float(np.clip(alocRF, 0, capitalTotal))
    alocRV    = capitalTotal - alocRF
    return alocRF, alocRV


def _montar_resultado(
    capitalTotal: float,
    alocRF: float,
    alocRV: float,
    crescimentoRF: float,
    retornoAcoes: float,
    retornosCumulativos: np.ndarray,
    capitalAportes: float,
    confianca: float,
    riscoAlvo: RiscoAlvo,
    diasInvestimento: int,
    proporcaoAcao: np.ndarray,
    tickers: list[str],
    retorno_rf_periodo: float,
) -> AlocacaoResultado:
    """
    Monta o objeto AlocacaoResultado com todos os campos calculados.

    Inclui distribuição completa do patrimônio final (RF fixo + RV variável
    por cenário) e métricas Sharpe/Sortino.
    """
    rf_final   = alocRF * crescimentoRF + capitalAportes
    rv_perda   = alocRV * (-retornoAcoes)
    rv_final   = alocRV * (1 + retornoAcoes)

    # Patrimônio final em cada cenário simulado: RF (fixo) + RV (variável)
    distribuicao = alocRF * crescimentoRF + capitalAportes + alocRV * (1 + retornosCumulativos)

    sharpe, sortino = _calcular_sharpe_sortino(retornosCumulativos, retorno_rf_periodo)

    return AlocacaoResultado(
        capitalTotal=capitalTotal,
        alocadoRendaFixa=alocRF,
        alocadoRendaVariavel=alocRV,
        saldoFinalRendaFixa=rf_final,
        perdaEsperadaRendaVariavel=rv_perda,
        patrimonioFinal=rf_final + rv_final,
        confianca=confianca,
        riscoAlvo=riscoAlvo,
        diasInvestimento=diasInvestimento,
        proporcaoAcao=proporcaoAcao,
        tickers=tickers,
        distribuicaoPatrimonio=distribuicao,
        sharpe=sharpe,
        sortino=sortino,
    )

def _simular_para_pesos(
    capitalTotal: float,
    proporcaoAcao: np.ndarray,
    tickers: list[str],
    mus: np.ndarray,
    sigmas: np.ndarray,
    nus: np.ndarray,
    corr: np.ndarray,
    crescimentoRF: float,
    retorno_rf_periodo: float,
    riscoAlvo: RiscoAlvo,
    diasInvestimento: int,
    confianca: float,
    numSimulacoes: int,
    diasRebalanceamento: int | None,
    capitalAportes: float,
    retornosCumulativos: np.ndarray | None = None,
    nu_copulas: float = 30.0,
    omegas = None,
    alphas = None,
    betas = None,
) -> tuple["AlocacaoResultado", np.ndarray]:
    """
    Executa Monte Carlo e monta o resultado para um dado vetor de pesos.

    Aceita `retornosCumulativos` pré-calculado para evitar re-simulação
    quando os pesos não mudam (ex: entre a versão original e a exibição).
    """
    if retornosCumulativos is None:
        retornosCumulativos = monteCarlo(
            mus, sigmas, nus, corr, proporcaoAcao,
            diasInvestimento, numSimulacoes, diasRebalanceamento,
            nu_copula=nu_copulas, omegas=omegas, alphas=alphas, betas=betas   
        )

    retornoAcoes    = _retorno_cenario_alvo(retornosCumulativos, confianca, riscoAlvo)
    alocRF, alocRV  = _resolver_alocacao(capitalTotal, crescimentoRF, retornoAcoes)

    resultado = _montar_resultado(
        capitalTotal, alocRF, alocRV, crescimentoRF, retornoAcoes,
        retornosCumulativos, capitalAportes, confianca, riscoAlvo,
        diasInvestimento, proporcaoAcao, tickers, retorno_rf_periodo,
    )
    return resultado, retornosCumulativos


# ═══════════════════════════════════════════════════════════════
# Otimização de pesos
# ═══════════════════════════════════════════════════════════════

def _otimizar_pesos(
    mus: np.ndarray,
    sigmas: np.ndarray,
    nus: np.ndarray,
    corr: np.ndarray,
    diasInvestimento: int,
    confianca: float,
    numSimulacoes: int,
    diasRebalanceamento: int | None,
    poupaTempo: bool,
    nu_copulas: float = 30.0,
    omegas = None,
    alphas = None,
    betas = None,
) -> np.ndarray:
    """
    Encontra os pesos da carteira RV que minimizam o CVaR via Nelder-Mead.

    Estratégia
    ----------
    - Função objetivo: CVaR da distribuição de retornos simulados.
    - Pesos sempre positivos e normalizados para somar 1 (via abs + normalização).
    - z_fixo sempre pré-gerado — superfície objetivo determinística em todas
      as iterações, independente de poupaTempo. Evita que Nelder-Mead persiga
      ruído entre iterações em vez de gradiente real.
    - poupaTempo reduz n_sim e afrouxa tolerâncias, mas mantém z fixo.
    - Validação final re-simula com novo ruído para evitar overfitting ao z fixo.
    """
    n     = len(mus)
    n_sim = numSimulacoes // 20 if poupaTempo else numSimulacoes // 5
    opts  = (
        {"maxiter": 150, "xatol": 1e-2, "fatol": 1e-3} if poupaTempo
        else {"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4}
    )

    # Sempre fixo — determinismo na superfície objetivo
    z_fixo = np.random.standard_normal((n_sim, diasInvestimento, n))

    print("  Otimizando pesos (minimização de CVaR)...")
    contador = [0]

    def objetivo(w: np.ndarray) -> float:
        contador[0] += 1
        print(f"  Iteração {contador[0]}", end="\r")

        w_norm = np.abs(w) / np.abs(w).sum()
        ret    = monteCarlo(mus, sigmas, nus, corr, w_norm,
                            diasInvestimento, n_sim, diasRebalanceamento, z_fixo,
                            nu_copula=nu_copulas, omegas=omegas, alphas=alphas, betas=betas)
        limiar = np.percentile(ret, (1 - confianca) * 100)
        return float(ret[ret <= limiar].mean())

    w0  = np.ones(n) / n
    res = optimize.minimize(objetivo, w0, method="Nelder-Mead", options=opts)

    print()
    w_opt = np.abs(res.x) / np.abs(res.x).sum()
    pesos_fmt = {t: round(float(w), 4) for t, w in zip(range(n), w_opt)}
    print(f"  Pesos otimizados: {pesos_fmt}")

    return w_opt

# ═══════════════════════════════════════════════════════════════
# Função principal
# ═══════════════════════════════════════════════════════════════

def simulacaoPortifolio(
    capitalTotal: float,
    tickers: list[str],
    proporcaoAcao: list[float],
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
    riscoAlvo: RiscoAlvo,
    diasInvestimento: int,
    confianca: float = 0.95,
    numSimulacoes: int = 1_000_000,
    periodo: str = "3y",
    otimizacao: bool = False,
    diasRebalanceamento: int | None = None,
    valorAporte: float = 0.0,
    frequenciaAporte: FrequenciaAporte | None = None,
    poupaTempo: bool = False,
) -> AlocacaoResultado:
    """
    Simula a alocação ótima entre RF e RV dado um capital e perfil de risco.

    Parâmetros
    ----------
    capitalTotal            : valor total disponível (R$)
    tickers                 : símbolos dos ativos RV (ex: ["PETR4.SA", "VALE3.SA"])
    proporcaoAcao           : proporção de cada ativo na carteira RV (soma = 1)
    rentabilidadeRendaFixa  : taxa da RF (ex: 0.145 = 14,5%)
    frequenciaRendaFixa     : frequência da taxa informada
    riscoAlvo               : como medir a perda esperada (MEDIA ou PIOR)
    diasInvestimento        : horizonte em dias úteis (252 = 1 ano)
    confianca               : cobertura desejada (ex: 0.95 = 95% dos cenários)
    numSimulacoes           : quantidade de cenários Monte Carlo
    periodo                 : janela histórica para calibração ("2y", "3y" etc)
    otimizacao              : se True, calcula também com pesos que minimizam CVaR
    diasRebalanceamento     : rebalanceia RV a cada N dias úteis (None = sem)
    valorAporte             : valor do aporte periódico em RF (R$); 0 = sem aportes
    frequenciaAporte        : intervalo entre aportes (MENSAL, TRIMESTRAL, SEMESTRAL)
    poupaTempo              : se True, usa menos simulações e z fixo na otimização
    """

    # ── 1. Validação e download dos dados históricos ──
    retornos, tickers = baixar_retornos(tickers, periodo)

    # Renormaliza pesos caso algum ticker tenha sido removido na validação
    proporcaoAcao = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao /= proporcaoAcao.sum()

    # ── 2. Calibração das distribuições e correlação ──
    nus, mus, sigmas, corr, nu_copula, omegas, alphas, betas = _calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros da RF ──
    diario_rf          = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)
    crescimentoRF      = (1 + diario_rf) ** diasInvestimento   # fator de crescimento total
    retorno_rf_periodo = crescimentoRF - 1                     # retorno percentual do período

    # ── 4. Aportes periódicos ──
    capitalAportes = 0.0
    if valorAporte > 0 and frequenciaAporte is not None:
        capitalAportes = _valor_futuro_aportes(
            valorAporte, frequenciaAporte, diasInvestimento,
            rentabilidadeRendaFixa, frequenciaRendaFixa,
        )
        print(f"  Aportes: valor futuro total em RF = R$ {capitalAportes:,.2f}")

    # ── 5. Monte Carlo com pesos originais ──
    print(f"Simulando {numSimulacoes} cenários × {diasInvestimento} dias...")
    resultado, retornosCumulativos = _simular_para_pesos(
        capitalTotal, proporcaoAcao, tickers,
        mus, sigmas, nus, corr,
        crescimentoRF, retorno_rf_periodo,
        riscoAlvo, diasInvestimento, confianca,
        numSimulacoes, diasRebalanceamento, capitalAportes,
        nu_copulas=nu_copula, omegas=omegas, alphas=alphas, betas=betas
    )

    # ── 6. Otimização de pesos (opcional) ──
    if otimizacao:
        print("Otimizando pesos...")
        w_opt = _otimizar_pesos(
            mus, sigmas, nus, corr,
            diasInvestimento, confianca, numSimulacoes,
            diasRebalanceamento, poupaTempo,
            nu_copulas=nu_copula, omegas=omegas,
            alphas=alphas, betas=betas
        )
        resultado_opt, _ = _simular_para_pesos(
            capitalTotal, w_opt, tickers,
            mus, sigmas, nus, corr,
            crescimentoRF, retorno_rf_periodo,
            riscoAlvo, diasInvestimento, confianca,
            numSimulacoes, diasRebalanceamento, capitalAportes,
            nu_copulas=nu_copula, omegas=omegas, alphas=alphas, betas=betas
        )
        resultado.otimizado = resultado_opt

    print(resultado)
    return resultado


# ═══════════════════════════════════════════════════════════════
# Exemplo de uso
# ═══════════════════════════════════════════════════════════════

simulacaoPortifolio(
    capitalTotal=100_000,                                       # capital inicial
    tickers=["PETR4.SA", "VALE3.SA", "ITUB4.SA"],               # tickers na carteira
    proporcaoAcao=[0.4, 0.35, 0.25],                            # proporcao que cada acao tem na carteira
    rentabilidadeRendaFixa=0.145,                               # rentabilidade da renda fixa
    frequenciaRendaFixa=FrequenciaRentabilidadeRendaFix.ANUAL,  # frequencia da rentabilidade da renda fixa
    riscoAlvo=RiscoAlvo.PIOR,                                   # qual o cenario que o usuario quer estar preparado PIOR/MEDIO
    diasInvestimento=252,                                       # dias para se operar
    confianca=0.95,                                             # quantos % dos cenarios o usuario quer considerar/ estar preparado
    otimizacao=False,                                            # otimizacao da relevancia que cada acao
    diasRebalanceamento=63,                                     # rebalanceia trimestralmente
    valorAporte=1_000,                                          # valor dos aportes
    frequenciaAporte=FrequenciaAporte.MENSAL,                   # frequencia dos aportes
    poupaTempo=False,                                            # recomendado caso otimização também seja True, apesar de diminuir um pouco a acuracia
)