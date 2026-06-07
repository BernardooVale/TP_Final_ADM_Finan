import numpy as np

from modelos.results import ResultadoComparador
from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.estrategias import EstrategiaUsuario, TipoEstrategiaBase, MetricasEstrategia
from monte_carlo import monteCarlo

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
        from modelos.defs import TipoEstrategiaBase as _T
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