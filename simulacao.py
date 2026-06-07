import numpy as np
from scipy import optimize

from defs import (
    RiscoAlvo, ParametrosRF,
    AlocacaoResultado, ParametrosCalibrados,
    ResultadoMeta, ResultadoDuploObjetivo,
    PontoParetoPatrimonio, RestricaoPiso,
    RestricaoMeta, ResultadoTempoMeta, 
    FrequenciaAporte, DIAS_UTEIS_APORTE,
    ResultadoDesacumulacao, ResultadoComparador,
    TipoEstrategiaBase, EstrategiaUsuario,
    MetricasEstrategia
)
from monte_carlo import monteCarlo, _gerar_inovacoes, _cholesky_seguro, _garch_scan_tempo_meta, _garch_scan_desacumulacao


def _cvar(retornos: np.ndarray, confianca: float) -> float:
    """CVaR: média dos retornos abaixo do percentil (1 - confiança)."""
    limiar   = np.percentile(retornos, (1 - confianca) * 100)
    tail     = retornos[retornos <= limiar]
    return float(tail.mean()) if len(tail) else 0.0


def _retorno_cenario_alvo(
    retornosCumulativos: np.ndarray,
    confianca:           float,
    riscoAlvo:           RiscoAlvo,
) -> float:
    """
    Perda esperada da RV no tail de risco.

    MEDIA → CVaR clássico (média do tail).
    PIOR  → mínimo absoluto do tail (CVaR extremo).
    """
    limiar   = np.percentile(retornosCumulativos, (1 - confianca) * 100)
    cenarios = retornosCumulativos[retornosCumulativos <= limiar]

    if not len(cenarios):
        return 0.0

    return float(cenarios.mean()) if riscoAlvo == RiscoAlvo.MEDIA else float(cenarios.min())


def _calcular_sharpe_sortino(
    retornosCumulativos: np.ndarray,
    retorno_rf_periodo:  float,
) -> tuple[float, float]:
    """
    Sharpe e Sortino da carteira simulada.

    Sharpe  = excesso médio / desvio padrão total
    Sortino = excesso médio / desvio padrão dos retornos abaixo da RF

    Sortino penaliza apenas volatilidade negativa — mais justo para distribuições assimétricas.
    """
    excesso  = retornosCumulativos - retorno_rf_periodo
    sharpe   = excesso.mean() / (excesso.std() + 1e-12)

    downside = excesso[excesso < 0]
    dd_std   = downside.std() if len(downside) > 1 else 1e-12
    sortino  = excesso.mean() / (dd_std + 1e-12)

    return float(sharpe), float(sortino)

def _resolver_alocacao(
    capitalTotal:  float,
    crescimentoRF: float,
    retornoAcoes:  float,
) -> tuple[float, float]:
    """
    Resolve algebricamente a alocação RF/RV.

    Objetivo: RF_final + RV_final >= capitalTotal no cenário alvo.

    alocRF = capitalTotal × (-retornoAcoes) / (crescimentoRF - (1 + retornoAcoes))

    Clipado em [0, capitalTotal] para evitar alocações negativas.
    """
    denom  = crescimentoRF - (1 + retornoAcoes)
    alocRF = float(np.clip(capitalTotal * (-retornoAcoes) / denom, 0, capitalTotal))
    return alocRF, capitalTotal - alocRF

def _calcular_metricas_resultado(
    alocRF:              float,
    alocRV:              float,
    crescimentoRF:       float,
    retornoAcoes:        float,
    retornosCumulativos: np.ndarray,
    capitalAportes:      float,
    retorno_rf_periodo:  float,
) -> dict:
    """
    Computa métricas derivadas: saldos, distribuição e índices de risco/retorno.

    Separado de _construir_resultado para facilitar testes unitários.
    """
    rf_final     = alocRF * crescimentoRF + capitalAportes
    rv_perda     = alocRV * (-retornoAcoes)
    rv_final     = alocRV * (1 + retornoAcoes)
    distribuicao = rf_final + alocRV * (1 + retornosCumulativos)
    sharpe, sortino = _calcular_sharpe_sortino(retornosCumulativos, retorno_rf_periodo)

    return dict(
        rf_final=rf_final,
        rv_perda=rv_perda,
        rv_final=rv_final,
        distribuicao=distribuicao,
        sharpe=sharpe,
        sortino=sortino,
    )

def _construir_resultado(
    capitalTotal:        float,
    alocRF:              float,
    alocRV:              float,
    retornoAcoes:        float,
    retornosCumulativos: np.ndarray,
    capitalAportes:      float,
    confianca:           float,
    riscoAlvo:           RiscoAlvo,
    diasInvestimento:    int,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    rf:                  ParametrosRF,
) -> AlocacaoResultado:
    """Constrói AlocacaoResultado a partir de métricas já calculadas."""
    m = _calcular_metricas_resultado(
        alocRF, alocRV, rf.crescimento, retornoAcoes,
        retornosCumulativos, capitalAportes, rf.retorno_periodo,
    )
    return AlocacaoResultado(
        capitalTotal               = capitalTotal,
        alocadoRendaFixa           = alocRF,
        alocadoRendaVariavel       = alocRV,
        saldoFinalRendaFixa        = m["rf_final"],
        perdaEsperadaRendaVariavel = m["rv_perda"],
        patrimonioFinal            = m["rf_final"] + m["rv_final"],
        confianca                  = confianca,
        riscoAlvo                  = riscoAlvo,
        diasInvestimento           = diasInvestimento,
        proporcaoAcao              = proporcaoAcao,
        tickers                    = tickers,
        distribuicaoPatrimonio     = m["distribuicao"],
        sharpe                     = m["sharpe"],
        sortino                    = m["sortino"],
    )

def simular_para_pesos(
    capitalTotal:        float,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    riscoAlvo:           RiscoAlvo,
    diasInvestimento:    int,
    confianca:           float,
    numSimulacoes:       int,
    diasRebalanceamento: int | None,
    capitalAportes:      float,
    retornosCumulativos: np.ndarray | None = None,
) -> tuple[AlocacaoResultado, np.ndarray]:
    """
    Executa Monte Carlo e monta resultado para um dado vetor de pesos.

    Aceita retornosCumulativos pré-calculado para evitar re-simulação
    quando os pesos não mudam (ex: exibição pós-otimização).
    """
    if retornosCumulativos is None:
        retornosCumulativos = monteCarlo(
            params, proporcaoAcao, diasInvestimento, numSimulacoes, diasRebalanceamento,
        )

    retornoAcoes   = _retorno_cenario_alvo(retornosCumulativos, confianca, riscoAlvo)
    alocRF, alocRV = _resolver_alocacao(capitalTotal, rf.crescimento, retornoAcoes)

    resultado = _construir_resultado(
        capitalTotal, alocRF, alocRV, retornoAcoes,
        retornosCumulativos, capitalAportes, confianca, riscoAlvo,
        diasInvestimento, proporcaoAcao, tickers, rf,
    )
    return resultado, retornosCumulativos


def _normalizar_pesos(w: np.ndarray) -> np.ndarray:
    """Garante pesos positivos normalizados para somar 1."""
    return np.abs(w) / np.abs(w).sum()


def otimizar_pesos(
    params:              ParametrosCalibrados,
    diasInvestimento:    int,
    confianca:           float,
    numSimulacoes:       int,
    diasRebalanceamento: int | None,
    poupaTempo:          bool,
) -> np.ndarray:
    """
    Pesos da carteira RV que minimizam CVaR via Nelder-Mead.

    z_fixo pré-gerado garante superfície objetivo determinística —
    evita que o otimizador persiga ruído entre iterações.
    poupaTempo reduz n_sim e afrouxa tolerâncias mantendo z fixo.
    """
    n     = len(params.mus)
    n_sim = numSimulacoes // 20 if poupaTempo else numSimulacoes // 5
    opts  = (
        {"maxiter": 150, "xatol": 1e-2, "fatol": 1e-3} if poupaTempo
        else {"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4}
    )
    z_fixo   = np.random.standard_normal((n_sim, diasInvestimento, n))
    contador = [0]

    print("  Otimizando pesos (minimização de CVaR)...")

    def objetivo(w: np.ndarray) -> float:
        contador[0] += 1
        print(f"  Iteração {contador[0]}", end="\r")
        ret = monteCarlo(params, _normalizar_pesos(w), diasInvestimento, n_sim, diasRebalanceamento, z_fixo)
        return _cvar(ret, confianca)

    res   = optimize.minimize(objetivo, np.ones(n) / n, method="Nelder-Mead", options=opts)
    w_opt = _normalizar_pesos(res.x)

    print()
    print(f"  Pesos otimizados: { {i: round(float(w), 4) for i, w in enumerate(w_opt)} }")
    return w_opt

# =========================
# Meta de patrimonio
# =========================

def _prob_atingir_meta(
    distribuicao_patrimonio: np.ndarray,
    meta:                    float,
) -> float:
    """Fração dos cenários onde o patrimônio final >= meta."""
    return float((distribuicao_patrimonio >= meta).mean())


def _patrimonio_final(
    alocRV:              float,
    capitalTotal:        float,
    retornosCumulativos: np.ndarray,
    crescimentoRF:       float,
    capitalAportes:      float,
) -> np.ndarray:
    """
    Distribuição do patrimônio final para uma dada alocRV.

    patrimônio_s = RF_final + alocRV * (1 + retRV_s)
    RF_final     = (capitalTotal - alocRV) * crescimentoRF + capitalAportes
    """
    alocRF  = capitalTotal - alocRV
    rf_final = alocRF * crescimentoRF + capitalAportes
    return rf_final + alocRV * (1 + retornosCumulativos)


def simular_meta_patrimonio(
    capitalTotal:        float,
    meta:                float,
    probabilidade:       float,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    diasInvestimento:    int,
    numSimulacoes:       int       = 1_000_000,
    diasRebalanceamento: int | None = None,
    capitalAportes:      float     = 0.0,
    tol:                 float     = 0.005,
) -> ResultadoMeta:
    """
    Busca binária sobre alocRV para atingir P(patrimônio >= meta) >= probabilidade.

    Estratégia
    ----------
    Simula retornos RV uma única vez (z fixo implícito via monteCarlo sem z_fixo —
    superfície estocástica, mas n_sim grande o suficiente para estabilidade).
    A busca binária varia apenas a alocação, não re-simula: usa a mesma
    distribuição de retornos cumulativos para todos os candidatos.

    Retorna a menor alocRV que satisfaz a condição (mínimo risco para a meta).
    Se nem alocRV = capitalTotal basta, marca atingivel=False e retorna 100% RV.

    Parâmetros
    ----------
    tol : tolerância da busca binária em R$ (para quando |hi - lo| < tol)
    """

    # Simula RV uma vez — reutilizado em toda a busca
    print(f"Simulando {numSimulacoes} cenários para busca de meta...")
    retornosCumulativos = monteCarlo(
        params, proporcaoAcao, diasInvestimento, numSimulacoes, diasRebalanceamento,
    )

    def prob_para_aloc(alocRV: float) -> tuple[float, np.ndarray]:
        dist = _patrimonio_final(alocRV, capitalTotal, retornosCumulativos, rf.crescimento, capitalAportes)
        return _prob_atingir_meta(dist, meta), dist

    # Verifica viabilidade com 100% RV
    prob_max, dist_max = prob_para_aloc(capitalTotal)
    if prob_max < probabilidade:
        percentil = float(np.percentile(dist_max, (1 - probabilidade) * 100))
        print(f"  ⚠ Meta não atingível: prob máxima com 100% RV = {prob_max*100:.1f}%")
        return ResultadoMeta(
            capitalTotal           = capitalTotal,
            meta                   = meta,
            probabilidade_alvo     = probabilidade,
            probabilidade_real     = prob_max,
            alocadoRendaFixa       = 0.0,
            alocadoRendaVariavel   = capitalTotal,
            patrimonio_p_alvo      = percentil,
            atingivel              = False,
            distribuicaoPatrimonio = dist_max,
        )

    # Busca binária: menor alocRV que satisfaz prob >= probabilidade
    lo, hi = 0.0, capitalTotal
    dist_final = dist_max  # fallback

    for _ in range(50):                      # 50 iterações → precisão < R$ 0,001 para 100k
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        prob_mid, dist_mid = prob_para_aloc(mid)
        if prob_mid >= probabilidade:
            hi         = mid
            dist_final = dist_mid
        else:
            lo = mid

    alocRV  = hi
    alocRF  = capitalTotal - alocRV
    prob_real, dist_final = prob_para_aloc(alocRV)
    percentil = float(np.percentile(dist_final, (1 - probabilidade) * 100))

    print(f"  Solução: RF=R${alocRF:,.0f} ({alocRF/capitalTotal*100:.1f}%)  "
          f"RV=R${alocRV:,.0f} ({alocRV/capitalTotal*100:.1f}%)  "
          f"prob={prob_real*100:.1f}%")

    return ResultadoMeta(
        capitalTotal           = capitalTotal,
        meta                   = meta,
        probabilidade_alvo     = probabilidade,
        probabilidade_real     = prob_real,
        alocadoRendaFixa       = alocRF,
        alocadoRendaVariavel   = alocRV,
        patrimonio_p_alvo      = percentil,
        atingivel              = True,
        distribuicaoPatrimonio = dist_final,
    )
    
# =======================
# Objetivo duplo
# =======================

def _avaliar_alocacao_dupla(
    alocRV:              float,
    capitalTotal:        float,
    retornosCumulativos: np.ndarray,
    crescimentoRF:       float,
    capitalAportes:      float,
    piso:                RestricaoPiso,
    meta:                RestricaoMeta,
) -> "PontoParetoPatrimonio":
    """Computa prob_piso e prob_meta para uma alocRV candidata."""
    from defs import PontoParetoPatrimonio
    dist       = _patrimonio_final(alocRV, capitalTotal, retornosCumulativos, crescimentoRF, capitalAportes)
    prob_piso  = float((dist >= piso.valor).mean())
    prob_meta  = float((dist >= meta.valor).mean())
    return PontoParetoPatrimonio(
        alocRV    = alocRV,
        alocRF    = capitalTotal - alocRV,
        prob_piso = prob_piso,
        prob_meta = prob_meta,
    )


def _busca_binaria_alocRV(
    capitalTotal:        float,
    retornosCumulativos: np.ndarray,
    crescimentoRF:       float,
    capitalAportes:      float,
    piso:                RestricaoPiso,
    meta:                RestricaoMeta,
    buscar_minimo:       bool,
    tol:                 float = 0.5,
) -> float | None:
    """
    Busca binária sobre alocRV para encontrar extremos do intervalo viável.

    buscar_minimo=True  → menor alocRV onde ambas as restrições são satisfeitas
    buscar_minimo=False → maior alocRV onde ambas as restrições são satisfeitas

    Retorna None se nenhum extremo existir (inviável).

    Monotonicidade:
        prob_piso decresce com alocRV (mais RV → mais risco de ficar abaixo do piso)
        prob_meta cresce  com alocRV (mais RV → mais chance de atingir a meta)
    O intervalo viável é contíguo (se existir): ambas satisfeitas entre
    alocRV_lo_critico (prob_piso começa a violar) e alocRV_hi_critico (prob_meta satisfeita).
    """
    def satisfaz(alocRV: float) -> bool:
        p = _avaliar_alocacao_dupla(
            alocRV, capitalTotal, retornosCumulativos,
            crescimentoRF, capitalAportes, piso, meta,
        )
        return p.prob_piso >= piso.confianca and p.prob_meta >= meta.confianca

    # Verifica extremos
    sat_zero  = satisfaz(0.0)
    sat_total = satisfaz(capitalTotal)

    if buscar_minimo:
        # Menor alocRV viável: começa em 0 e sobe até encontrar região viável
        if sat_zero:
            return 0.0
        if not sat_total:
            return None   # nenhuma alocação satisfaz ambas
        lo, hi = 0.0, capitalTotal
        for _ in range(60):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2
            if satisfaz(mid):
                hi = mid
            else:
                lo = mid
        return hi
    else:
        # Maior alocRV viável: começa em capitalTotal e desce
        if sat_total:
            return capitalTotal
        if not sat_zero:
            return None
        lo, hi = 0.0, capitalTotal
        for _ in range(60):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2
            if satisfaz(mid):
                lo = mid
            else:
                hi = mid
        return lo


def simular_duplo_objetivo(
    capitalTotal:        float,
    piso:                RestricaoPiso,
    meta:                RestricaoMeta,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    diasInvestimento:    int,
    numSimulacoes:       int        = 1_000_000,
    diasRebalanceamento: int | None = None,
    capitalAportes:      float      = 0.0,
    n_pontos_pareto:     int        = 20,
    tol:                 float      = 0.5,
) -> ResultadoDuploObjetivo:
    """
    Encontra a fronteira de Pareto entre proteção de piso e atingimento de meta.

    Simula RV uma vez; busca binária e varredura da fronteira reutilizam
    a mesma distribuição de retornos cumulativos.

    Parâmetros
    ----------
    piso             : patrimônio mínimo + confiança desejada
    meta             : patrimônio-alvo + probabilidade mínima de atingi-lo
    n_pontos_pareto  : pontos na curva Pareto entre os extremos viáveis
    tol              : tolerância da busca binária em R$

    Retorno
    -------
    ResultadoDuploObjetivo com:
    - ponto conservador (min RV satisfazendo ambas)
    - ponto agressivo   (max RV satisfazendo ambas)
    - fronteira de Pareto entre eles
    """

    assert piso.valor < meta.valor,    "Piso deve ser menor que meta"
    assert 0 < piso.confianca < 1,     "Confiança do piso deve estar em (0, 1)"
    assert 0 < meta.confianca < 1,     "Confiança da meta deve estar em (0, 1)"
    assert piso.valor < capitalTotal * (1 + rf.retorno_periodo), \
        "Piso acima do retorno garantido pela RF — inviável por construção"

    print(f"Simulando {numSimulacoes} cenários para duplo objetivo...")
    retornosCumulativos = monteCarlo(
        params, proporcaoAcao, diasInvestimento, numSimulacoes, diasRebalanceamento,
    )

    def _avalia(alocRV: float) -> PontoParetoPatrimonio:
        return _avaliar_alocacao_dupla(
            alocRV, capitalTotal, retornosCumulativos,
            rf.crescimento, capitalAportes, piso, meta,
        )

    # ── Diagnóstico de viabilidade por restrição ──
    p_zero  = _avalia(0.0)
    p_total = _avalia(capitalTotal)

    piso_viavel  = p_zero.prob_piso  >= piso.confianca   # 0% RV já satisfaz piso?
    meta_viavel  = p_total.prob_meta >= meta.confianca   # 100% RV satisfaz meta?

    if not meta_viavel:
        msg = (f"Meta inatingível: prob máxima com 100% RV = {p_total.prob_meta*100:.1f}% "
               f"< {meta.confianca*100:.0f}% exigido")
        return ResultadoDuploObjetivo(
            capitalTotal=capitalTotal, piso=piso, meta=meta,
            viavel=False, ponto_minimo_rv=None, ponto_maximo_rv=None,
            fronteira=[], mensagem=msg,
        )

    if not piso_viavel and p_total.prob_piso < piso.confianca:
        msg = (f"Piso inatingível mesmo com 0% RV: prob_piso={p_zero.prob_piso*100:.1f}% "
               f"< {piso.confianca*100:.0f}% exigido. "
               "Considere reduzir exigência de piso ou aumentar a taxa RF.")
        return ResultadoDuploObjetivo(
            capitalTotal=capitalTotal, piso=piso, meta=meta,
            viavel=False, ponto_minimo_rv=None, ponto_maximo_rv=None,
            fronteira=[], mensagem=msg,
        )

    # ── Busca dos extremos do intervalo viável ──
    aloc_min = _busca_binaria_alocRV(
        capitalTotal, retornosCumulativos, rf.crescimento, capitalAportes,
        piso, meta, buscar_minimo=True, tol=tol,
    )
    aloc_max = _busca_binaria_alocRV(
        capitalTotal, retornosCumulativos, rf.crescimento, capitalAportes,
        piso, meta, buscar_minimo=False, tol=tol,
    )

    if aloc_min is None or aloc_max is None:
        msg = ("Não existe alocação que satisfaça piso e meta simultaneamente. "
               "As restrições são mutuamente exclusivas para esses ativos.")
        return ResultadoDuploObjetivo(
            capitalTotal=capitalTotal, piso=piso, meta=meta,
            viavel=False, ponto_minimo_rv=None, ponto_maximo_rv=None,
            fronteira=[], mensagem=msg,
        )

    ponto_min = _avalia(aloc_min)
    ponto_max = _avalia(aloc_max)

    # ── Fronteira de Pareto entre os extremos ──
    alocs_pareto = np.linspace(aloc_min, aloc_max, n_pontos_pareto)
    fronteira    = [_avalia(a) for a in alocs_pareto]

    msg = (f"Intervalo viável: RV entre R${aloc_min:,.0f} "
           f"({aloc_min/capitalTotal*100:.1f}%) e "
           f"R${aloc_max:,.0f} ({aloc_max/capitalTotal*100:.1f}%)")

    print(f"  {msg}")

    return ResultadoDuploObjetivo(
        capitalTotal    = capitalTotal,
        piso            = piso,
        meta            = meta,
        viavel          = True,
        ponto_minimo_rv = ponto_min,
        ponto_maximo_rv = ponto_max,
        fronteira       = fronteira,
        mensagem        = msg,
    )
    
def simular_tempo_para_meta(
    capitalTotal:        float,
    meta:                float,
    fracao_rv:           float,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    rentabilidadeRFDiaria: float,          # taxa diária já convertida (de preparar_parametros_rf)
    diasInvestimento:    int,              # horizonte máximo em dias úteis
    numSimulacoes:       int       = 1_000_000,
    diasRebalanceamento: int | None = None,
    valorAporte:         float     = 0.0,
    frequenciaAporte:    FrequenciaAporte | None = None,
    percentis:           list[int] = [5, 10, 25, 50, 75, 90],
) -> ResultadoTempoMeta:
    """
    Simula a distribuição do tempo necessário para atingir `meta` a partir de `capitalTotal`.

    Diferença de abordagem vs demais funções
    -----------------------------------------
    Patrimônio evolui dia a dia dentro do kernel Numba (_garch_scan_tempo_meta).
    Cada cenário para assim que cruza a meta — sem acúmulo pós-cruzamento.
    Percentis calculados apenas sobre cenários que atingem a meta.

    Parâmetros
    ----------
    fracao_rv            : fração do capital total alocada em RV ∈ [0, 1]
    rentabilidadeRFDiaria: taxa diária RF — obter via _taxa_diaria_rf() de renda_fixa.py
    diasInvestimento     : horizonte máximo (cenários que não cruzam = "não atingido")
    valorAporte          : aporte periódico em R$ (0 = sem aportes)
    frequenciaAporte     : frequência do aporte (usa DIAS_UTEIS_APORTE)
    percentis            : percentis a reportar na distribuição de tempos
    """

    assert 0.0 <= fracao_rv <= 1.0, "fracao_rv deve estar em [0, 1]"
    assert meta > capitalTotal,     "meta deve ser maior que o capital inicial"

    fracao_rf = 1.0 - fracao_rv
    pesos_rv  = proporcaoAcao / proporcaoAcao.sum()
    chol      = _cholesky_seguro(params.corr)
    n         = len(params.mus)

    # Aporte: converte para valor por dia útil e intervalo em dias úteis
    intervalo_aporte = (
        DIAS_UTEIS_APORTE[frequenciaAporte]
        if frequenciaAporte is not None and valorAporte > 0 else 0
    )
    aporte_dia = valorAporte if intervalo_aporte > 0 else 0.0

    crescimento_rf_dia = 1.0 + rentabilidadeRFDiaria

    print(f"Simulando tempo para meta: {numSimulacoes} cenários × até {diasInvestimento} dias...")

    # Gera inovações completas — Numba para no cruzamento, mas precisa do array cheio
    z       = np.random.standard_normal((numSimulacoes, diasInvestimento, n))
    epsilon = _gerar_inovacoes(z, chol, params.nu_copula, params.nus)

    dias_cruzamento = _garch_scan_tempo_meta(
        epsilon.astype(np.float64),
        params.omegas.astype(np.float64),
        params.alphas.astype(np.float64),
        params.betas.astype(np.float64),
        params.mus.astype(np.float64),
        params.sigmas.astype(np.float64),
        pesos_rv.astype(np.float64),
        float(fracao_rf),
        float(crescimento_rf_dia),
        float(aporte_dia),
        int(intervalo_aporte),
        float(capitalTotal),
        float(meta),
    )

    # -1 = não atingiu; converte para NaN para separar na análise
    dias_float = dias_cruzamento.astype(float)
    dias_float[dias_float < 0] = np.nan

    atingiram       = dias_float[~np.isnan(dias_float)]
    n_nao_atingiu   = int(np.isnan(dias_float).sum())
    prob_atingir    = len(atingiram) / numSimulacoes

    if len(atingiram) == 0:
        print("  ⚠ Nenhum cenário atingiu a meta no horizonte simulado.")
        anos_uteis = 252
        return ResultadoTempoMeta(
            capitalTotal      = capitalTotal,
            meta              = meta,
            max_dias          = diasInvestimento,
            prob_atingir      = 0.0,
            percentis_dias    = {p: np.nan for p in percentis},
            percentis_anos    = {p: np.nan for p in percentis},
            dias_nao_atingido = n_nao_atingiu,
            dias_por_cenario  = dias_float,
        )

    anos_uteis     = 252
    vals_dias      = {p: float(np.percentile(atingiram, p)) for p in percentis}
    vals_anos      = {p: v / anos_uteis for p, v in vals_dias.items()}

    print(f"  P(atingir): {prob_atingir*100:.1f}%  |  "
          f"P50={vals_anos[50]:.1f}a  P10={vals_anos.get(10, float('nan')):.1f}a  "
          f"P90={vals_anos.get(90, float('nan')):.1f}a")

    return ResultadoTempoMeta(
        capitalTotal      = capitalTotal,
        meta              = meta,
        max_dias          = diasInvestimento,
        prob_atingir      = prob_atingir,
        percentis_dias    = vals_dias,
        percentis_anos    = vals_anos,
        dias_nao_atingido = n_nao_atingiu,
        dias_por_cenario  = dias_float,
    )
    
def _rodar_desacumulacao(
    epsilon:            np.ndarray,
    params:             ParametrosCalibrados,
    chol:               np.ndarray,
    pesos_rv:           np.ndarray,
    fracao_rf:          float,
    crescimento_rf_dia: float,
    saque:              float,
    intervalo_saque:    int,
    capitalTotal:       float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera inovações e roda kernel de desacumulação para um valor de saque.

    Reutiliza o mesmo `epsilon` pré-gerado — superfície determinística
    para busca binária sem re-simular.
    """

    # epsilon já gerado externamente; inovações re-derivadas do mesmo z
    # (epsilon aqui É a inovação final — passa direto ao kernel)
    return _garch_scan_desacumulacao(
        epsilon.astype(np.float64),
        params.omegas.astype(np.float64),
        params.alphas.astype(np.float64),
        params.betas.astype(np.float64),
        params.mus.astype(np.float64),
        params.sigmas.astype(np.float64),
        pesos_rv.astype(np.float64),
        float(fracao_rf),
        float(crescimento_rf_dia),
        float(saque),
        int(intervalo_saque),
        float(capitalTotal),
    )

def simular_desacumulacao(
    capitalTotal:        float,
    saque:               float,
    frequenciaSaque:     "FrequenciaAporte",
    fracao_rv:           float,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    rentabilidadeRFDiaria: float,
    diasInvestimento:    int,
    numSimulacoes:       int       = 1_000_000,
    diasRebalanceamento: int | None = None,   # reservado — não implementado no kernel
    limite_ruina:        float     = 0.05,
    percentis_duracao:   list[int] = [10, 25, 50, 75, 90],
    tol_saque:           float     = 1.0,
) -> ResultadoDesacumulacao:
    """
    Simula fase de desacumulação com saques periódicos e calcula métricas de ruína.

    Sub-problemas resolvidos com uma única simulação
    -------------------------------------------------
    1. Prob. de ruína para `saque` informado
    2. Taxa de saque sustentável (busca binária, mesmo epsilon)
    3. Distribuição do tempo até ruína (cenários que arruínam)

    Estratégia de saque
    -------------------
    Saque é debitado da parcela RF. Se RF insuficiente, liquida RV para cobrir
    o déficit (venda forçada). Ruína ocorre quando RF + RV <= 0.

    Parâmetros
    ----------
    saque                : valor retirado por período (R$)
    frequenciaSaque      : periodicidade do saque (usa DIAS_UTEIS_APORTE)
    fracao_rv            : fração do capital em RV ∈ [0, 1]
    rentabilidadeRFDiaria: taxa diária RF (de _taxa_diaria_rf)
    diasInvestimento     : horizonte máximo em dias úteis
    limite_ruina         : prob. máxima de ruína tolerada na busca sustentável
    tol_saque            : tolerância da busca binária em R$
    diasRebalanceamento  : reservado para versão futura do kernel
    """

    assert 0.0 <= fracao_rv <= 1.0, "fracao_rv deve estar em [0, 1]"
    assert saque >= 0,              "saque deve ser não-negativo"
    assert 0 < limite_ruina < 1,    "limite_ruina deve estar em (0, 1)"

    fracao_rf        = 1.0 - fracao_rv
    pesos_rv         = proporcaoAcao / proporcaoAcao.sum()
    chol             = _cholesky_seguro(params.corr)
    intervalo_saque  = DIAS_UTEIS_APORTE[frequenciaSaque]
    n                = len(params.mus)

    print(f"Simulando desacumulação: {numSimulacoes:,} cenários × {diasInvestimento} dias...")

    # Gera inovações uma vez — reutilizadas em toda a busca binária
    z       = np.random.standard_normal((numSimulacoes, diasInvestimento, n))
    epsilon = _gerar_inovacoes(z, chol, params.nu_copula, params.nus)
    del z   # libera memória

    def _rodar(s: float) -> tuple[np.ndarray, np.ndarray]:
        return _rodar_desacumulacao(
            epsilon, params, chol, pesos_rv, fracao_rf,
            1.0 + rentabilidadeRFDiaria, s, intervalo_saque, capitalTotal,
        )

    # ── 1. Métricas para o saque informado ──
    dia_ruina, pat_final = _rodar(saque)

    arruinou     = dia_ruina > 0
    prob_ruina   = float(arruinou.mean())
    prob_sobrev  = 1.0 - prob_ruina

    sobreviventes = pat_final[~arruinou]
    pat_mediano   = float(np.median(sobreviventes)) if len(sobreviventes) else None

    # Distribuição de duração (só cenários que arruínam)
    dias_ruina_arr = dia_ruina[arruinou].astype(float)
    if len(dias_ruina_arr):
        anos_uteis = 252
        p_dias = {p: float(np.percentile(dias_ruina_arr, p)) for p in percentis_duracao}
        p_anos = {p: v / anos_uteis for p, v in p_dias.items()}
    else:
        p_dias, p_anos = {}, {}

    # ── 2. Taxa de saque sustentável (busca binária) ──
    # Verifica se saque=0 já apresenta ruína (por perdas de mercado puras)
    _, pat_zero = _rodar(0.0)
    prob_ruina_zero = float((pat_zero <= 0).mean())

    if prob_ruina_zero > limite_ruina:
        saque_sust = None
        print(f"  ⚠ Prob. ruína com saque=0: {prob_ruina_zero*100:.1f}% > limite {limite_ruina*100:.0f}%")
    else:
        # Teto da busca: saque que quase certamente arruína (heurística: 2× retorno RF)
        saque_max = capitalTotal * rf.retorno_periodo * 2 / (diasInvestimento / intervalo_saque)
        # Garante que saque_max realmente viola o limite
        while True:
            d_max, _ = _rodar(saque_max)
            if (d_max > 0).mean() > limite_ruina:
                break
            saque_max *= 2.0

        lo, hi = 0.0, saque_max
        print(f"  Buscando taxa sustentável em [0, R${saque_max:,.0f}]...")
        for _ in range(50):
            if hi - lo < tol_saque:
                break
            mid = (lo + hi) / 2
            d_mid, _ = _rodar(mid)
            if float((d_mid > 0).mean()) <= limite_ruina:
                lo = mid
            else:
                hi = mid
        saque_sust = lo
        print(f"  Saque sustentável: R$ {saque_sust:,.2f} / {frequenciaSaque.value}")

    print(f"  Prob. ruína (saque informado): {prob_ruina*100:.1f}%")

    return ResultadoDesacumulacao(
        capitalTotal       = capitalTotal,
        saque_simulado     = saque,
        frequencia_saque   = frequenciaSaque,
        saque_sustentavel  = saque_sust,
        prob_ruina         = prob_ruina,
        limite_ruina_alvo  = limite_ruina,
        percentis_duracao  = {"dias": p_dias, "anos": p_anos},
        prob_sobreviver    = prob_sobrev,
        patrimonio_mediano = pat_mediano,
        max_dias           = diasInvestimento,
    )
    
def _metricas_distribuicao(
    nome:         str,
    dist:         np.ndarray,
    capitalTotal: float,
    meta:         float | None,
    fracao_rv:    float | None,
) -> MetricasEstrategia:
    """Calcula MetricasEstrategia a partir de uma distribuição de patrimônio."""
    q1, med, q3 = np.percentile(dist, [25, 50, 75])
    media        = float(dist.mean())
    prob_perda   = float((dist < capitalTotal).mean())
    prob_meta    = float((dist >= meta).mean()) if meta is not None else None
    return MetricasEstrategia(
        nome          = nome,
        fracao_rv     = fracao_rv,
        q1            = float(q1),
        mediana       = float(med),
        q3            = float(q3),
        media         = media,
        prob_meta     = prob_meta,
        prob_perda    = prob_perda,
        retorno_medio = media / capitalTotal - 1.0,
    )


def comparar_estrategias(
    capitalTotal:      float,
    proporcaoAcao:     np.ndarray,
    tickers:           list[str],
    params:            ParametrosCalibrados,
    rf:                ParametrosRF,
    diasInvestimento:  int,
    estrategia_usuario: EstrategiaUsuario | None         = None,
    estrategias_base:   list[TipoEstrategiaBase] | None  = None,
    meta:               float | None                       = None,
    numSimulacoes:      int                                = 500_000,
    diasRebalanceamento: int | None                        = None,
) -> ResultadoComparador:
    """
    Compara estratégias lado a lado sobre o mesmo capital e horizonte.

    Comparação justa
    ----------------
    Todas as estratégias com RV != 0 compartilham o mesmo z_fixo —
    mesma realização de mercado. Diferenças nos resultados refletem
    apenas a alocação, não sorte amostral.

    Estratégias disponíveis
    -----------------------
    - Fixas via TipoEstrategiaBase: 100% RF, 100% RV, 75/25, 25/75
    - Usuário via EstrategiaUsuario: qualquer AlocacaoResultado ou
      ResultadoMeta já calculado — passa resultado.distribuicaoPatrimonio

    Parâmetros
    ----------
    estrategia_usuario  : estratégia customizada do usuário (opcional)
    estrategias_base    : subconjunto de TipoEstrategiaBase a incluir;
                          None = todas as quatro
    meta                : patrimônio-alvo para P(meta); None = omite coluna
    """

    if estrategias_base is None:
        from defs import TipoEstrategiaBase as _T
        estrategias_base = list(_T)

    pesos_rv = proporcaoAcao / proporcaoAcao.sum()
    n        = len(params.mus)

    # Mapa fracao_rv por estratégia fixa
    _FRACAO_RV: dict = {
        TipoEstrategiaBase.RF_100:    0.00,
        TipoEstrategiaBase.RV_100:    1.00,
        TipoEstrategiaBase.RV75_RF25: 0.75,
        TipoEstrategiaBase.RV25_RF75: 0.25,
    }

    # Identifica estratégias que precisam de simulação RV
    fracoes_rv_necessarias = {_FRACAO_RV[e] for e in estrategias_base if _FRACAO_RV[e] > 0}

    print(f"Comparando estratégias: {numSimulacoes:,} cenários × {diasInvestimento} dias...")

    # Simula RV uma vez por vetor de pesos — z_fixo compartilhado
    z_fixo   = np.random.standard_normal((numSimulacoes, diasInvestimento, n))
    cache_rv: dict[float, np.ndarray] = {}

    for frv in sorted(fracoes_rv_necessarias):
        cache_rv[frv] = monteCarlo(
            params, pesos_rv, diasInvestimento,
            numSimulacoes, diasRebalanceamento, z_fixo,
        )

    def _dist_para_fracao(frv: float) -> np.ndarray:
        """Patrimônio final R$ para uma dada fracao_rv."""
        if frv == 0.0:
            return np.full(numSimulacoes, capitalTotal * rf.crescimento)
        ret_rv = cache_rv[frv]
        rf_cap = capitalTotal * (1.0 - frv)
        rv_cap = capitalTotal * frv
        return rf_cap * rf.crescimento + rv_cap * (1.0 + ret_rv)

    metricas: list[MetricasEstrategia] = []

    # ── Estratégias fixas ──
    for eb in estrategias_base:
        frv  = _FRACAO_RV[eb]
        dist = _dist_para_fracao(frv)
        metricas.append(_metricas_distribuicao(eb.value, dist, capitalTotal, meta, frv))
        print(f"  ✓ {eb.value}")

    # ── Estratégia do usuário ──
    if estrategia_usuario is not None:
        metricas.append(_metricas_distribuicao(
            estrategia_usuario.nome,
            estrategia_usuario.distribuicaoPatrimonio,
            capitalTotal, meta,
            estrategia_usuario.fracao_rv,
        ))
        print(f"  ✓ {estrategia_usuario.nome} (usuário)")

    # Ordena por mediana decrescente
    metricas.sort(key=lambda m: m.mediana, reverse=True)

    return ResultadoComparador(
        estrategias  = metricas,
        capitalTotal = capitalTotal,
        meta         = meta,
    )