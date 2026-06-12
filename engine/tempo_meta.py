import numpy as np

from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.defs import FrequenciaAporte, DIAS_UTEIS_APORTE
from modelos.results import ResultadoTempoMeta
from engine.monte_carlo import _cholesky_seguro, _iterar_chunks, _t_grid
from engine.kernels import _kernel_tempo_meta

def simular_tempo_para_meta(
    capitalTotal:         float,
    meta:                 float,
    fracao_rv:            float,
    proporcaoAcao:        np.ndarray,
    tickers:              list[str],
    params:               ParametrosCalibrados,
    rf:                   ParametrosRF,
    rentabilidadeRFDiaria: float,
    diasInvestimento:     int,
    numSimulacoes:        int       = 1_000_000,
    diasRebalanceamento:  int | None = None,
    valorAporte:          float     = 0.0,
    frequenciaAporte:     FrequenciaAporte | None = None,
    percentis:            list[int] = [5, 10, 25, 50, 75, 90],
    chunk_size:           int       = 50_000,
) -> ResultadoTempoMeta:
    """
    Simula a distribuição do tempo necessário para atingir `meta` a partir de `capitalTotal`.

    Diferença de abordagem vs demais funções
    -----------------------------------------
    Patrimônio evolui dia a dia dentro do kernel Numba (_kernel_tempo_meta).
    Cada cenário para assim que cruza a meta — sem acúmulo pós-cruzamento.
    Inovações geradas dentro do kernel dia a dia — sem pré-alocação de epsilon.
    Percentis calculados apenas sobre cenários que atingem a meta.

    Parâmetros
    ----------
    fracao_rv            : fração do capital total alocada em RV ∈ [0, 1]
    rentabilidadeRFDiaria: taxa diária RF — obter via _taxa_diaria_rf() de renda_fixa.py
    diasInvestimento     : horizonte máximo (cenários que não cruzam = "não atingido")
    valorAporte          : aporte periódico em R$ (0 = sem aportes)
    frequenciaAporte     : frequência do aporte (usa DIAS_UTEIS_APORTE)
    percentis            : percentis a reportar na distribuição de tempos
    chunk_size           : cenários por chunk — controla granularidade do progresso
    """

    assert 0.0 <= fracao_rv <= 1.0, "fracao_rv deve estar em [0, 1]"
    assert meta > capitalTotal,     "meta deve ser maior que o capital inicial"

    fracao_rf = 1.0 - fracao_rv
    pesos_rv  = proporcaoAcao / proporcaoAcao.sum()
    chol      = _cholesky_seguro(params.corr)

    intervalo_aporte = (
        DIAS_UTEIS_APORTE[frequenciaAporte]
        if frequenciaAporte is not None and valorAporte > 0 else 0
    )
    aporte_dia         = valorAporte if intervalo_aporte > 0 else 0.0
    crescimento_rf_dia = 1.0 + rentabilidadeRFDiaria

    print(f"Simulando tempo para meta: {numSimulacoes} cenários × até {diasInvestimento} dias "
          f"(chunks de {chunk_size})...")

    dias_cruzamento = np.empty(numSimulacoes, dtype=np.float64)

    # Monta grades t-Student uma vez — passadas ao kernel para CDF/PPF inline
    x_grid_cop, u_grid_cop = _t_grid(params.nu_copula)
    A = len(params.mus)
    x_grids_mar = np.empty((A, len(x_grid_cop)))
    u_grids_mar = np.empty((A, len(x_grid_cop)))
    for a in range(A):
        x_grids_mar[a], u_grids_mar[a] = _t_grid(float(params.nus[a]))

    for start, end in _iterar_chunks(numSimulacoes, chunk_size):
        n_chunk = end - start
        dias_cruzamento[start:end] = _kernel_tempo_meta(
            params.mus.astype(np.float64),
            params.sigmas.astype(np.float64),
            pesos_rv.astype(np.float64),
            chol.astype(np.float64),
            float(params.nu_copula),
            x_grid_cop.astype(np.float64),
            u_grid_cop.astype(np.float64),
            x_grids_mar.astype(np.float64),
            u_grids_mar.astype(np.float64),
            float(fracao_rf),
            float(crescimento_rf_dia),
            float(aporte_dia),
            int(intervalo_aporte),
            float(capitalTotal),
            float(meta),
            int(n_chunk),
            int(diasInvestimento),
        )
        print(f"  {end:>{len(str(numSimulacoes))}}/{numSimulacoes} cenários...", end="\r")

    print()

    dias_float = dias_cruzamento
    dias_float[dias_float < 0] = np.nan

    atingiram     = dias_float[~np.isnan(dias_float)]
    n_nao_atingiu = int(np.isnan(dias_float).sum())
    prob_atingir  = len(atingiram) / numSimulacoes

    if len(atingiram) == 0:
        print("  ⚠ Nenhum cenário atingiu a meta no horizonte simulado.")
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

    anos_uteis = 252
    vals_dias  = {p: float(np.percentile(atingiram, p)) for p in percentis}
    vals_anos  = {p: v / anos_uteis for p, v in vals_dias.items()}

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