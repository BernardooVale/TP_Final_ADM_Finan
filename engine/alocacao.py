import numpy as np

from modelos.params import ParametrosRF, ParametrosCalibrados
from modelos.defs import RiscoAlvo
from modelos.results import AlocacaoResultado
from engine.monte_carlo import monteCarlo
from engine.metricas import _retorno_cenario_alvo, _calcular_sharpe_sortino

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
    diasInvestimento:    int
) -> dict:
    """
    Computa métricas derivadas: saldos, distribuição e índices de risco/retorno.

    Separado de _construir_resultado para facilitar testes unitários.
    """
    rf_final     = alocRF * crescimentoRF + capitalAportes
    rv_perda     = alocRV * (-retornoAcoes)
    rv_final     = alocRV * (1 + retornoAcoes)
    distribuicao = rf_final + alocRV * (1 + retornosCumulativos)
    sharpe, sortino = _calcular_sharpe_sortino(
        retornosCumulativos,
        retorno_rf_periodo,
        diasInvestimento,     
    )

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
        diasInvestimento
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