import numpy as np

from modelos.defs import FrequenciaAporte, DIAS_UTEIS_APORTE
from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.results import ResultadoDesacumulacao
from engine.monte_carlo import _cholesky_seguro, _iterar_chunks, _gerar_inovacoes
from engine.kernels import _garch_scan_desacumulacao

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
    
    sigma2_0 = params.sigmas ** 2

    return _garch_scan_desacumulacao(
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
        float(saque),
        int(intervalo_saque),
        float(capitalTotal),
    )

def simular_desacumulacao(
    capitalTotal:          float,
    saque:                 float,
    frequenciaSaque:       FrequenciaAporte,
    fracao_rv:             float,
    proporcaoAcao:         np.ndarray,
    tickers:               list[str],
    params:                ParametrosCalibrados,
    rf:                    ParametrosRF,
    rentabilidadeRFDiaria: float,
    diasInvestimento:      int,
    numSimulacoes:         int        = 1_000_000,
    diasRebalanceamento:   int | None = None,
    limite_ruina:          float      = 0.05,
    percentis_duracao:     list[int]  = [10, 25, 50, 75, 90],
    tol_saque:             float      = 1.0,
    chunk_size:            int        = 50_000,
) -> ResultadoDesacumulacao:
    """
    Simula fase de desacumulação com saques periódicos e calcula métricas de ruína.

    Sub-problemas resolvidos com uma única geração de inovações
    -----------------------------------------------------------
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
    chunk_size           : cenários por chunk — controla pico de memória RAM
    diasRebalanceamento  : reservado para versão futura do kernel
    """

    assert 0.0 <= fracao_rv <= 1.0, "fracao_rv deve estar em [0, 1]"
    assert saque >= 0,              "saque deve ser não-negativo"
    assert 0 < limite_ruina < 1,    "limite_ruina deve estar em (0, 1)"

    fracao_rf       = 1.0 - fracao_rv
    pesos_rv        = proporcaoAcao / proporcaoAcao.sum()
    chol            = _cholesky_seguro(params.corr)
    intervalo_saque = DIAS_UTEIS_APORTE[frequenciaSaque]
    n               = len(params.mus)

    print(f"Simulando desacumulação: {numSimulacoes:,} cenários × {diasInvestimento} dias "
          f"(chunks de {chunk_size})...")

    # ── Pré-gera e persiste epsilon por chunk ──────────────────────────────────
    # Chunks são re-executados para cada valor de saque na busca binária,
    # mas épsilon é o mesmo — garante superfície determinística sem alocar
    # o array completo (n_sim, dias, n) de uma vez.
    #
    # Estratégia: salva epsilon de cada chunk em lista de arrays menores.
    # Custo: memória = n_sim × dias × n × 8 bytes, mas distribuída em chunks
    # e descartável após a busca binária.
    epsilons: list[np.ndarray] = []
    for start, end in _iterar_chunks(numSimulacoes, chunk_size):
        n_chunk = end - start
        z       = np.random.standard_normal((n_chunk, diasInvestimento, n))
        eps     = _gerar_inovacoes(z, chol, params.nu_copula, params.nus)
        epsilons.append(eps.astype(np.float64))
        del z
        print(f"  Gerando inovações: {end:>{len(str(numSimulacoes))}}/{numSimulacoes}...", end="\r")
    print()

    def _rodar_chunks(s: float) -> tuple[np.ndarray, np.ndarray]:
        """Roda kernel de desacumulação sobre todos os chunks para um dado saque."""
        dias_r_parts  = []
        pat_f_parts   = []
        for eps in epsilons:
            dr, pf = _rodar_desacumulacao(
                eps, params, chol, pesos_rv, fracao_rf,
                1.0 + rentabilidadeRFDiaria, s, intervalo_saque, capitalTotal,
            )
            dias_r_parts.append(dr)
            pat_f_parts.append(pf)
        return np.concatenate(dias_r_parts), np.concatenate(pat_f_parts)

    # ── 1. Métricas para o saque informado ────────────────────────────────────
    dia_ruina, pat_final = _rodar_chunks(saque)

    arruinou    = dia_ruina > 0
    prob_ruina  = float(arruinou.mean())
    prob_sobrev = 1.0 - prob_ruina

    sobreviventes = pat_final[~arruinou]
    pat_mediano   = float(np.median(sobreviventes)) if len(sobreviventes) else None

    dias_ruina_arr = dia_ruina[arruinou].astype(float)
    if len(dias_ruina_arr):
        anos_uteis = 252
        p_dias = {p: float(np.percentile(dias_ruina_arr, p)) for p in percentis_duracao}
        p_anos = {p: v / anos_uteis for p, v in p_dias.items()}
    else:
        p_dias, p_anos = {}, {}

    # ── 2. Taxa de saque sustentável (busca binária) ───────────────────────────
    _, pat_zero     = _rodar_chunks(0.0)
    prob_ruina_zero = float((pat_zero <= 0).mean())

    if prob_ruina_zero > limite_ruina:
        saque_sust = None
        print(f"  ⚠ Prob. ruína com saque=0: {prob_ruina_zero*100:.1f}% > limite {limite_ruina*100:.0f}%")
    else:
        saque_max = capitalTotal * rf.retorno_periodo * 2 / (diasInvestimento / intervalo_saque)
        max_dobras = 30  # guarda contra loop infinito
        for _ in range(max_dobras):
            d_max, _ = _rodar_chunks(saque_max)
            if (d_max > 0).mean() > limite_ruina:
                break
            saque_max *= 2.0
        else:
            # Nenhum saque dobrado violou o limite — incomum mas tratado
            saque_sust = saque_max
            print(f"  ⚠ Limite de dobras atingido; saque sustentável estimado em R$ {saque_sust:,.2f}")
            saque_max = None  # sinaliza para pular busca binária

        if saque_max is not None:
            lo, hi = 0.0, saque_max
            print(f"  Buscando taxa sustentável em [0, R${saque_max:,.0f}]...")
            for _ in range(50):
                if hi - lo < tol_saque:
                    break
                mid = (lo + hi) / 2
                d_mid, _ = _rodar_chunks(mid)
                if float((d_mid > 0).mean()) <= limite_ruina:
                    lo = mid
                else:
                    hi = mid
            saque_sust = lo
            print(f"  Saque sustentável: R$ {saque_sust:,.2f} / {frequenciaSaque.value}")

    print(f"  Prob. ruína (saque informado): {prob_ruina*100:.1f}%")

    # Libera memória das inovações após busca completa
    del epsilons

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