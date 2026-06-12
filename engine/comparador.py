import numpy as np

from modelos.results import ResultadoComparador
from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.estrategias import EstrategiaUsuario, TipoEstrategiaBase, MetricasEstrategia
from engine.monte_carlo import monteCarlo

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
    capitalTotal:        float,
    proporcaoAcao:       np.ndarray,
    params:              ParametrosCalibrados,
    rf:                  ParametrosRF,
    diasInvestimento:    int,
    estrategia_usuario:  EstrategiaUsuario | None = None,
    estrategias_base:    list[TipoEstrategiaBase] = list(TipoEstrategiaBase),
    meta:                float | None             = None,
    numSimulacoes:       int                      = 1_000_000,
    diasRebalanceamento: int | None               = None,
    retornosCumulativos: np.ndarray | None        = None
) -> ResultadoComparador:
    
    pesos_rv = proporcaoAcao / proporcaoAcao.sum()
    n        = len(params.mus)

    # Mapa fracao_rv por estratégia fixa
    _FRACAO_RV: dict = {
        TipoEstrategiaBase.RF_100:    0.00,
        TipoEstrategiaBase.RV_100:    1.00,
        TipoEstrategiaBase.RV75_RF25: 0.75,
        TipoEstrategiaBase.RV25_RF75: 0.25,
    }

    # Verifica se alguma estratégia selecionada precisa da curva de RV
    precisa_rv = any(_FRACAO_RV[e] > 0 for e in estrategias_base)

    # ── Reutilização ou Geração Única do Monte Carlo ──
    if precisa_rv:
        if retornosCumulativos is not None:
            print("  Reutilizando retornos cumulativos fornecidos pela API...")
        else:
            print(f"  Comparando estratégias: {numSimulacoes:,} cenários × {diasInvestimento} dias...")
            z_fixo = np.random.standard_normal((numSimulacoes, diasInvestimento, n))
            retornosCumulativos = monteCarlo(
                params, pesos_rv, diasInvestimento,
                numSimulacoes, diasRebalanceamento, z_fixo,
            )

    def _dist_para_fracao(frv: float) -> np.ndarray:
        """Mistura capital em RF determinística com RV simulada."""
        if frv == 0.0:
            return np.full(numSimulacoes, capitalTotal * rf.crescimento)
        
        # Aplica a fração de RV sobre o array único de retornos
        rf_cap = capitalTotal * (1.0 - frv)
        rv_cap = capitalTotal * frv
        return rf_cap * rf.crescimento + rv_cap * (1.0 + retornosCumulativos)

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
    ), retornosCumulativos