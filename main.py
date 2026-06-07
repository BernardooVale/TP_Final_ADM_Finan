"""
main.py
=======
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
    pip install numpy pandas yfinance scipy numba
"""

import numpy as np

from data.mineracao import baixar_retornos
from data.calibracoes import calibrar_todos
from renda_fixa import preparar_aportes, preparar_parametros_rf
from modelos.defs import (
    FrequenciaRentabilidadeRendaFix,
    FrequenciaAporte,
    RiscoAlvo
)
from modelos.results import AlocacaoResultado
from engine.otimizacao import otimizar_pesos
from engine.alocacao import simular_para_pesos

def simulacaoPortifolio(
    capitalTotal:           float,
    tickers:                list[str],
    proporcaoAcao:          list[float],
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa:    FrequenciaRentabilidadeRendaFix,
    riscoAlvo:              RiscoAlvo,
    diasInvestimento:       int,
    confianca:              float = 0.95,
    numSimulacoes:          int = 1_000_000,
    periodo:                str = "3y",
    otimizacao:             bool = False,
    diasRebalanceamento:    int | None = None,
    valorAporte:            float = 0.0,
    frequenciaAporte:       FrequenciaAporte | None = None,
    poupaTempo:             bool = False,
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
    poupaTempo              : reduz simulações na otimização (menor acurácia)
    """

    # ── 1. Download e validação ──
    retornos, tickers = baixar_retornos(tickers, periodo)
    proporcaoAcao     = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao    /= proporcaoAcao.sum()

    # ── 2. Calibração ──
    params = calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros RF ──
    rf = preparar_parametros_rf(rentabilidadeRendaFixa, frequenciaRendaFixa, diasInvestimento)

    # ── 4. Aportes ──
    capitalAportes = preparar_aportes(
        valorAporte, frequenciaAporte, diasInvestimento,
        rentabilidadeRendaFixa, frequenciaRendaFixa,
    )

    # ── 5. Monte Carlo — pesos originais ──
    print(f"Simulando {numSimulacoes} cenários × {diasInvestimento} dias...")
    resultado, retornosCumulativos = simular_para_pesos(
        capitalTotal, proporcaoAcao, tickers, params, rf,
        riscoAlvo, diasInvestimento, confianca,
        numSimulacoes, diasRebalanceamento, capitalAportes,
    )

    # ── 6. Otimização de pesos (opcional) ──
    if otimizacao:
        print("Otimizando pesos...")
        w_opt = otimizar_pesos(
            params, diasInvestimento, confianca,
            numSimulacoes, diasRebalanceamento, poupaTempo,
        )
        resultado.otimizado, _ = simular_para_pesos(
            capitalTotal, w_opt, tickers, params, rf,
            riscoAlvo, diasInvestimento, confianca,
            numSimulacoes, diasRebalanceamento, capitalAportes,
        )

    print(resultado)
    return resultado

# ═══════════════════════════════════════════════════════════════
# Exemplo de uso
# ═══════════════════════════════════════════════════════════════

simulacaoPortifolio(
    capitalTotal           = 100_000,
    tickers                = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
    proporcaoAcao          = [0.4, 0.35, 0.25],
    rentabilidadeRendaFixa = 0.145,
    frequenciaRendaFixa    = FrequenciaRentabilidadeRendaFix.ANUAL,
    riscoAlvo              = RiscoAlvo.PIOR,
    diasInvestimento       = 252,
    confianca              = 0.95,
    otimizacao             = False,
    diasRebalanceamento    = 63,
    valorAporte            = 1_000,
    frequenciaAporte       = FrequenciaAporte.MENSAL,
    poupaTempo             = False,
)