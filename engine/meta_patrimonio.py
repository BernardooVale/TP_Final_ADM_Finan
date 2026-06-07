import numpy as np

from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.results import ResultadoMeta, RestricaoMeta, RestricaoPiso, ResultadoDuploObjetivo
from modelos.pareto import PontoParetoPatrimonio
from engine.monte_carlo import monteCarlo

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
    
def _avaliar_alocacao_dupla(
    alocRV:              float,
    capitalTotal:        float,
    retornosCumulativos: np.ndarray,
    crescimentoRF:       float,
    capitalAportes:      float,
    piso:                RestricaoPiso,
    meta:                RestricaoMeta,
) -> PontoParetoPatrimonio:
    """Computa prob_piso e prob_meta para uma alocRV candidata."""
    from modelos.defs import PontoParetoPatrimonio
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