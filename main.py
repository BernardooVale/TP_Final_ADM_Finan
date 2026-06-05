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
import yfinance as yf
from scipy import stats, optimize
from scipy.special import ndtr
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Enumerações
# ═══════════════════════════════════════════════════════════════

class RiscoAlvo(str, Enum):
    """Define como a perda esperada da RV é calculada nos cenários ruins."""
    MEDIA = "media"  # média dos cenários abaixo do limiar de confiança
    PIOR  = "pior"   # mínimo absoluto dos cenários abaixo do limiar (CVaR extremo)


class FrequenciaRentabilidadeRendaFix(str, Enum):
    """Frequência em que a taxa da RF é expressa."""
    DIARIO     = "diario"
    MENSAL     = "mensal"
    TRIMESTRAL = "trimestral"
    ANUAL      = "anual"


class FrequenciaAporte(str, Enum):
    """Intervalo entre aportes periódicos."""
    MENSAL     = "mensal"      # a cada 21 dias úteis
    TRIMESTRAL = "trimestral"  # a cada 63 dias úteis
    SEMESTRAL  = "semestral"   # a cada 126 dias úteis


# ═══════════════════════════════════════════════════════════════
# Constantes de mapeamento
# ═══════════════════════════════════════════════════════════════

# Dias úteis equivalentes por frequência de RF
DIAS_UTEIS_RF: dict[FrequenciaRentabilidadeRendaFix, int] = {
    FrequenciaRentabilidadeRendaFix.DIARIO:      1,
    FrequenciaRentabilidadeRendaFix.MENSAL:     21,
    FrequenciaRentabilidadeRendaFix.TRIMESTRAL:  63,
    FrequenciaRentabilidadeRendaFix.ANUAL:      252,
}

# Dias úteis equivalentes por frequência de aporte
DIAS_UTEIS_APORTE: dict[FrequenciaAporte, int] = {
    FrequenciaAporte.MENSAL:     21,
    FrequenciaAporte.TRIMESTRAL:  63,
    FrequenciaAporte.SEMESTRAL:  126,
}


# ═══════════════════════════════════════════════════════════════
# Dataclass de resultado
# ═══════════════════════════════════════════════════════════════

@dataclass
class AlocacaoResultado:
    """
    Contém todos os dados de saída da simulação.

    Campos principais
    -----------------
    capitalTotal               : capital inicial investido
    alocadoRendaFixa           : parcela alocada em RF
    alocadoRendaVariavel       : parcela alocada em RV
    saldoFinalRendaFixa        : valor da RF ao fim do horizonte (inclui aportes)
    perdaEsperadaRendaVariavel : perda da RV no cenário alvo
    patrimonioFinal            : RF final + RV após perda esperada
    distribuicaoPatrimonio     : patrimônio final em todos os cenários simulados
    sharpe / sortino           : métricas de risco/retorno da carteira RV simulada
    otimizado                  : resultado com pesos otimizados (None se não solicitado)
    """
    capitalTotal:               float
    alocadoRendaFixa:           float
    alocadoRendaVariavel:       float
    saldoFinalRendaFixa:        float
    perdaEsperadaRendaVariavel: float
    patrimonioFinal:            float
    confianca:                  float
    riscoAlvo:                  RiscoAlvo
    diasInvestimento:           int
    proporcaoAcao:              np.ndarray
    tickers:                    list[str]
    distribuicaoPatrimonio:     np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe:                     float = 0.0
    sortino:                    float = 0.0
    otimizado:                  "AlocacaoResultado | None" = None

    # ── Formatação do output ──

    def _str_percentis(self) -> str:
        """Formata os percentis P5/P25/P50/P75/P95 da distribuição do patrimônio final."""
        if not len(self.distribuicaoPatrimonio):
            return ""
        percentis = np.percentile(self.distribuicaoPatrimonio, [5, 25, 50, 75, 95])
        linhas = ["\n  Distribuição do patrimônio final (R$):"]
        for p, val in zip([5, 25, 50, 75, 95], percentis):
            linhas.append(f"    P{p:>2}:  R$ {val:>12,.2f}")
        return "\n".join(linhas)

    def _str_bloco(self, titulo: str = "") -> str:
        """
        Monta o bloco de texto formatado para exibição no terminal.
        O cabeçalho (cab) inclui linha decorativa e título opcional.
        """
        cab = f"\n{'═'*55}\n"
        if titulo:
            cab += f"  {titulo}\n{'─'*55}\n"

        pesos_fmt = "  ".join(
            f"{t}: {p*100:.1f}%" for t, p in zip(self.tickers, self.proporcaoAcao)
        )
        return (
            f"{cab}"
            f"  Pesos RV:            {pesos_fmt}\n"
            f"  Capital total:       R$ {self.capitalTotal:>12,.2f}\n"
            f"  Alocado em RF:       R$ {self.alocadoRendaFixa:>12,.2f}  ({self.alocadoRendaFixa/self.capitalTotal*100:.1f}%)\n"
            f"  Alocado em RV:       R$ {self.alocadoRendaVariavel:>12,.2f}  ({self.alocadoRendaVariavel/self.capitalTotal*100:.1f}%)\n"
            f"{'─'*55}\n"
            f"  RF ao fim ({self.diasInvestimento}d):   R$ {self.saldoFinalRendaFixa:>12,.2f}\n"
            f"  Perda esperada RV:   R$ {self.perdaEsperadaRendaVariavel:>12,.2f}\n"
            f"{'─'*55}\n"
            f"  Patrimônio final:    R$ {self.patrimonioFinal:>12,.2f}\n"
            f"  Cobertura:           {'✓ >= capital inicial' if self.patrimonioFinal >= self.capitalTotal else '✗ insuficiente'}\n"
            f"  Cenário:             {self.riscoAlvo.value} | confiança {self.confianca*100:.0f}%\n"
            f"  Sharpe:              {self.sharpe:.4f}\n"
            f"  Sortino:             {self.sortino:.4f}\n"
            f"{self._str_percentis()}\n"
            f"{'═'*55}\n"
        )

    def __str__(self) -> str:
        s = self._str_bloco("Pesos originais")
        if self.otimizado is not None:
            s += self.otimizado._str_bloco("Pesos otimizados (mín. CVaR)")
        return s


# ═══════════════════════════════════════════════════════════════
# Aquisição e validação de dados
# ═══════════════════════════════════════════════════════════════

def _baixar_retornos(tickers: list[str], periodo: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Baixa histórico de preços via yfinance e calcula retornos diários.

    Remove tickers sem dados suficientes (< 30 observações) e avisa o usuário.
    Retorna o DataFrame de retornos e a lista de tickers válidos.
    """
    print("Baixando e validando tickers...")
    precos = yf.download(tickers, period=periodo, auto_adjust=True, progress=False)["Close"]

    # yfinance retorna Series quando há apenas um ticker — força DataFrame
    if isinstance(precos, pd.Series):
        precos = precos.to_frame(name=tickers[0])

    # Filtra tickers com dados insuficientes para calibração confiável
    validos   = [t for t in tickers if t in precos.columns and precos[t].notna().sum() > 30]
    invalidos = set(tickers) - set(validos)

    if invalidos:
        print(f"  ⚠ Tickers ignorados (sem dados suficientes): {invalidos}")
    if not validos:
        raise ValueError("Nenhum ticker válido encontrado. Verifique os símbolos informados.")

    retornos = precos[validos].pct_change().dropna()
    return retornos, validos


# ═══════════════════════════════════════════════════════════════
# Calibração estatística
# ═══════════════════════════════════════════════════════════════

def _calibrar_t_student(serie: pd.Series) -> tuple[float, float, float]:
    """
    Ajusta distribuição t-Student aos retornos de um ativo via MLE.

    A t-Student captura caudas gordas melhor que a normal,
    representando com mais fidelidade eventos extremos de mercado.

    Retorna (nu, mu, sigma): graus de liberdade, média e escala.
    """
    nu, mu, sigma = stats.t.fit(serie)
    return nu, mu, sigma


def _calibrar_todos(retornos: pd.DataFrame, tickers: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibra t-Student para cada ativo e computa a matriz de correlação.

    Retorna (nus, mus, sigmas, corr) como arrays numpy.
    """
    fitted = [_calibrar_t_student(retornos[t]) for t in tickers]
    nus    = np.array([f[0] for f in fitted])
    mus    = np.array([f[1] for f in fitted])
    sigmas = np.array([f[2] for f in fitted])

    # Matriz de correlação histórica — preservada na simulação via decomposição de Cholesky
    corr = retornos.corr().values

    return nus, mus, sigmas, corr


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
        total += valorAporte * (1 + rentabilidadeRendaFixa) ** ciclos_completos

    return total


# ═══════════════════════════════════════════════════════════════
# Simulação Monte Carlo
# ═══════════════════════════════════════════════════════════════

def _simular_retornos_diarios(
    mus: np.ndarray,
    sigmas: np.ndarray,
    nus: np.ndarray,
    chol: np.ndarray,
    n_sim: int,
    diasInvestimento: int,
    n_assets: int,
    z_fixo: np.ndarray | None,
) -> np.ndarray:
    """
    Gera retornos diários correlacionados para todos os ativos e cenários.

    Usa a decomposição de Cholesky da matriz de correlação para introduzir
    dependência entre os ativos. Os retornos individuais seguem t-Student.

    Retorna array de shape (n_sim, diasInvestimento, n_assets).
    """
    # Ruído base: normal padrão independente por ativo
    # Reutiliza z_fixo se fornecido (otimização de tempo — evita re-sampling)
    z = z_fixo if z_fixo is not None else np.random.standard_normal((n_sim, diasInvestimento, n_assets))

    # Reshape para aplicar PPF em array 2D em vez de 3D fatiado
    S, D, A = z.shape
    z_flat  = z.reshape(-1, A)                    # (S*D, A)
    u_flat  = np.empty_like(z_flat)

    for i in range(A):                            # loop só sobre ativos (3), não S*D
        u_flat[:, i] = stats.t.ppf(
            ndtr(z_flat[:, i]),           
            df=nus[i], loc=mus[i], scale=sigmas[i]
        )

    u = u_flat.reshape(S, D, A)
    return u @ chol.T


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
) -> np.ndarray:
    
    chol     = np.linalg.cholesky(matrizCorrelacao)
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
            mus, sigmas, nus, chol, n_chunk, diasInvestimento, n_assets, z_chunk
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
) -> np.ndarray:
    """
    Encontra os pesos da carteira RV que minimizam o CVaR via Nelder-Mead.

    Estratégia
    ----------
    - Função objetivo: CVaR da distribuição de retornos simulados.
    - Pesos sempre positivos e normalizados para somar 1 (via abs + normalização).
    - Se `poupaTempo=True`: usa menos simulações, z pré-fixado e tolerâncias maiores,
      reduzindo drasticamente o tempo ao custo de precisão levemente menor.

    z_fixo: ruído base pré-gerado e reutilizado em todas as iterações do otimizador.
    Isso elimina re-sampling e torna a busca determinística — o otimizador não
    persegue ruído entre iterações, convergindo mais rápido.
    """
    n     = len(mus)
    n_sim = numSimulacoes // 20 if poupaTempo else numSimulacoes // 5
    opts  = (
        {"maxiter": 150, "xatol": 1e-2, "fatol": 1e-3} if poupaTempo
        else {"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4}
    )

    # Pré-gera ruído base uma única vez para todas as iterações do otimizador
    z_fixo = np.random.standard_normal((n_sim, diasInvestimento, n)) if poupaTempo else None

    print("  Otimizando pesos (minimização de CVaR)...")
    contador = [0]  # lista para permitir mutação dentro do closure

    def objetivo(w: np.ndarray) -> float:
        """Função objetivo: CVaR dos retornos simulados para os pesos w."""
        contador[0] += 1
        print(f"  Iteração {contador[0]}", end="\r")

        w_norm = np.abs(w) / np.abs(w).sum()
        ret    = monteCarlo(mus, sigmas, nus, corr, w_norm,
                            diasInvestimento, n_sim, diasRebalanceamento, z_fixo)
        limiar = np.percentile(ret, (1 - confianca) * 100)
        return float(ret[ret <= limiar].mean())

    w0  = np.ones(n) / n  # ponto inicial: pesos iguais
    res = optimize.minimize(objetivo, w0, method="Nelder-Mead", options=opts)

    print()  # quebra linha após os \r do contador
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
    retornos, tickers = _baixar_retornos(tickers, periodo)

    # Renormaliza pesos caso algum ticker tenha sido removido na validação
    proporcaoAcao = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao /= proporcaoAcao.sum()

    # ── 2. Calibração das distribuições e correlação ──
    nus, mus, sigmas, corr = _calibrar_todos(retornos, tickers)

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
    )

    # ── 6. Otimização de pesos (opcional) ──
    if otimizacao:
        print("Otimizando pesos...")
        w_opt = _otimizar_pesos(
            mus, sigmas, nus, corr,
            diasInvestimento, confianca, numSimulacoes,
            diasRebalanceamento, poupaTempo,
        )
        resultado_opt, _ = _simular_para_pesos(
            capitalTotal, w_opt, tickers,
            mus, sigmas, nus, corr,
            crescimentoRF, retorno_rf_periodo,
            riscoAlvo, diasInvestimento, confianca,
            numSimulacoes, diasRebalanceamento, capitalAportes,
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