import numpy as np

from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.defs import FrequenciaAporte, DIAS_UTEIS_APORTE
from modelos.results import ResultadoTempoMeta
from monte_carlo import _cholesky_seguro, _iterar_chunks, _gerar_inovacoes, _sigma2_iniciais
from kernels import _garch_scan_tempo_meta

def simular_tempo_para_meta(
    capitalTotal:        float,
    meta:                float,
    fracao_rv:           float,
    proporcaoAcao:       np.ndarray,
    tickers:             list[str],
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    rentabilidadeRFDiaria: float,
    diasInvestimento:    int,
    numSimulacoes:       int       = 1_000_000,
    diasRebalanceamento: int | None = None,
    valorAporte:         float     = 0.0,
    frequenciaAporte:    FrequenciaAporte | None = None,
    percentis:           list[int] = [5, 10, 25, 50, 75, 90],
    chunk_size:          int       = 50_000,
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
    chunk_size           : cenários por chunk — controla pico de memória RAM
                           (50_000 × 2520 dias × 3 ativos × 8 bytes ≈ 3 GB por chunk)
    """

    assert 0.0 <= fracao_rv <= 1.0, "fracao_rv deve estar em [0, 1]"
    assert meta > capitalTotal,     "meta deve ser maior que o capital inicial"

    fracao_rf = 1.0 - fracao_rv
    pesos_rv  = proporcaoAcao / proporcaoAcao.sum()
    chol      = _cholesky_seguro(params.corr)
    n         = len(params.mus)

    intervalo_aporte = (
        DIAS_UTEIS_APORTE[frequenciaAporte]
        if frequenciaAporte is not None and valorAporte > 0 else 0
    )
    aporte_dia         = valorAporte if intervalo_aporte > 0 else 0.0
    crescimento_rf_dia = 1.0 + rentabilidadeRFDiaria

    print(f"Simulando tempo para meta: {numSimulacoes} cenários × até {diasInvestimento} dias "
          f"(chunks de {chunk_size})...")

    dias_cruzamento = np.empty(numSimulacoes, dtype=np.float64)

    for start, end in _iterar_chunks(numSimulacoes, chunk_size):
        n_chunk = end - start
        z       = np.random.standard_normal((n_chunk, diasInvestimento, n))
        epsilon = _gerar_inovacoes(z, chol, params.nu_copula, params.nus)
        del z

        sigma2_0 = _sigma2_iniciais(params.omegas, params.alphas, params.betas, params.sigmas)

        dias_cruzamento[start:end] = _garch_scan_tempo_meta(
            epsilon.astype(np.float64),
            params.omegas.astype(np.float64),
            params.alphas.astype(np.float64),
            params.betas.astype(np.float64),
            params.mus.astype(np.float64),
            params.sigmas.astype(np.float64),
            sigma2_0.astype(np.float64),
            pesos_rv.astype(np.float64),
            float(fracao_rf),
            float(crescimento_rf_dia),
            float(aporte_dia),
            int(intervalo_aporte),
            float(capitalTotal),
            float(meta),
        )
        del epsilon

        concluidos = end
        print(f"  {concluidos:>{len(str(numSimulacoes))}}/{numSimulacoes} cenários...", end="\r")

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