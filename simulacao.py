import numpy as np
from scipy import optimize

from defs import RiscoAlvo, ParametrosRF, AlocacaoResultado, ParametrosCalibrados
from monte_carlo import monteCarlo

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