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
        riscoAlvo              = RiscoAlvo.MEDIA,
        diasInvestimento       = 252,
        confianca              = 0.85,
        otimizacao             = False,
        diasRebalanceamento    = 63,
        valorAporte            = 1_000,
        frequenciaAporte       = FrequenciaAporte.MENSAL,
        poupaTempo             = False,
    )
    
from engine.comparador import comparar_estrategias

def compararEstrategias(
    tickers, proporcaoAcao, meta, capitalTotal, rentabilidadeRendaFixa, frequenciaRendaFixa, periodo = "3y", diasInvestimento = 252, numSimulacoes = 1_000_000
    ):
    
    # ── 1. Download e validação ──
    retornos, tickers = baixar_retornos(tickers, periodo)
    proporcaoAcao     = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao    /= proporcaoAcao.sum()

    # ── 2. Calibração ──
    params = calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros RF ──
    rf = preparar_parametros_rf(rentabilidadeRendaFixa, frequenciaRendaFixa, diasInvestimento)

    resultado = comparar_estrategias(
        capitalTotal     = capitalTotal,
        proporcaoAcao    = proporcaoAcao,
        tickers          = tickers,
        params           = params,
        rf               = rf,
        diasInvestimento = diasInvestimento,
        meta             = meta,
        numSimulacoes    = numSimulacoes,
    )

    print(resultado)
    return resultado

    compararEstrategias(["PETR4.SA", "VALE3.SA", "ITUB4.SA"], [0.4, 0.35, 0.25], 120_000, 100_000, 0.145, FrequenciaRentabilidadeRendaFix.ANUAL)
    
from engine.desacumulacao import simular_desacumulacao
from renda_fixa import _taxa_diaria_rf

def simularDesacumulacao(
    tickers:                list[str],
    proporcaoAcao:          list[float],
    capitalTotal:           float,
    saque:                  float,
    frequenciaSaque:        FrequenciaAporte,
    fracao_rv:              float,
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa:    FrequenciaRentabilidadeRendaFix,
    periodo:                str   = "3y",
    diasInvestimento:       int   = 252 * 20,
    numSimulacoes:          int   = 100_000,
    limite_ruina:           float = 0.05,
    diasRebalanceamento:    int | None = None,
):
    # ── 1. Download e validação ──
    retornos, tickers = baixar_retornos(tickers, periodo)
    proporcaoAcao     = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao    /= proporcaoAcao.sum()

    # ── 2. Calibração ──
    params = calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros RF ──
    rf                   = preparar_parametros_rf(rentabilidadeRendaFixa, frequenciaRendaFixa, diasInvestimento)
    rentabilidadeRFDiaria = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)

    resultado = simular_desacumulacao(
        capitalTotal          = capitalTotal,
        saque                 = saque,
        frequenciaSaque       = frequenciaSaque,
        fracao_rv             = fracao_rv,
        proporcaoAcao         = proporcaoAcao,
        tickers               = tickers,
        params                = params,
        rf                    = rf,
        rentabilidadeRFDiaria = rentabilidadeRFDiaria,
        diasInvestimento      = diasInvestimento,
        numSimulacoes         = numSimulacoes,
        limite_ruina          = limite_ruina,
        diasRebalanceamento   = diasRebalanceamento,
    )

    print(resultado)
    return resultado

    simularDesacumulacao(
        tickers                = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        proporcaoAcao          = [0.4, 0.35, 0.25],
        capitalTotal           = 100_000,
        saque                  = 1_000,
        frequenciaSaque        = FrequenciaAporte.MENSAL,
        fracao_rv              = 0.3,
        rentabilidadeRendaFixa = 0.145,
        frequenciaRendaFixa    = FrequenciaRentabilidadeRendaFix.ANUAL,
    )
    
from engine.fronteira import calcular_fronteira

def calcularFronteira(
    tickers:                list[str],
    proporcaoAcao:          list[float],
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa:    FrequenciaRentabilidadeRendaFix,
    periodo:                str   = "3y",
    diasInvestimento:       int   = 252,
    confianca:              float = 0.95,
    n_pontos:               int   = 10,
    n_sim:                  int   = 5_000,
    diasRebalanceamento:    int | None = None,
):
    # ── 1. Download e validação ──
    retornos, tickers = baixar_retornos(tickers, periodo)
    proporcaoAcao     = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao    /= proporcaoAcao.sum()

    # ── 2. Calibração ──
    params = calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros RF ──
    rf = preparar_parametros_rf(rentabilidadeRendaFixa, frequenciaRendaFixa, diasInvestimento)

    fronteira = calcular_fronteira(
        params     = params,
        tickers    = tickers,
        rf         = rf,
        diasInvest = diasInvestimento,
        confianca  = confianca,
        n_pontos   = n_pontos,
        n_sim      = n_sim,
        diasRebal  = diasRebalanceamento,
    )

    for ponto in fronteira.pontos:
        pesos_fmt = "  ".join(f"{t}: {w*100:.1f}%" for t, w in zip(ponto.tickers, ponto.pesos_rv))
        print(f"Retorno alvo: {ponto.retorno_alvo*100:.1f}%  "
              f"CVaR: {ponto.cvar*100:.1f}%  "
              f"RF: {ponto.fracao_rf*100:.1f}%  "
              f"RV: {ponto.fracao_rv*100:.1f}%  "
              f"Pesos: {pesos_fmt}")

    return fronteira

calcularFronteira(
    tickers                = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
    proporcaoAcao          = [0.4, 0.35, 0.25],
    rentabilidadeRendaFixa = 0.145,
    frequenciaRendaFixa    = FrequenciaRentabilidadeRendaFix.ANUAL,
)
    
from engine.meta_patrimonio import simular_meta_patrimonio

def simularMetaPatrimonio(
    tickers:                list[str],
    proporcaoAcao:          list[float],
    capitalTotal:           float,
    meta:                   float,
    probabilidade:          float,
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa:    FrequenciaRentabilidadeRendaFix,
    periodo:                str   = "3y",
    diasInvestimento:       int   = 252,
    numSimulacoes:          int   = 1_000_000,
    diasRebalanceamento:    int | None = None,
):
    # ── 1. Download e validação ──
    retornos, tickers = baixar_retornos(tickers, periodo)
    proporcaoAcao     = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao    /= proporcaoAcao.sum()

    # ── 2. Calibração ──
    params = calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros RF ──
    rf = preparar_parametros_rf(rentabilidadeRendaFixa, frequenciaRendaFixa, diasInvestimento)

    resultado = simular_meta_patrimonio(
        capitalTotal        = capitalTotal,
        meta                = meta,
        probabilidade       = probabilidade,
        proporcaoAcao       = proporcaoAcao,
        tickers             = tickers,
        params              = params,
        rf                  = rf,
        diasInvestimento    = diasInvestimento,
        numSimulacoes       = numSimulacoes,
        diasRebalanceamento = diasRebalanceamento,
    )

    print(resultado)
    return resultado

    simularMetaPatrimonio(
        tickers                = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        proporcaoAcao          = [0.4, 0.35, 0.25],
        capitalTotal           = 100_000,
        meta                   = 130_000,
        probabilidade          = 0.70,
        rentabilidadeRendaFixa = 0.145,
        frequenciaRendaFixa    = FrequenciaRentabilidadeRendaFix.ANUAL,
    )
    
from engine.tempo_meta import simular_tempo_para_meta
from renda_fixa import _taxa_diaria_rf

def simularTempoParaMeta(
    tickers:                list[str],
    proporcaoAcao:          list[float],
    capitalTotal:           float,
    meta:                   float,
    fracao_rv:              float,
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa:    FrequenciaRentabilidadeRendaFix,
    periodo:                str   = "3y",
    diasInvestimento:       int   = 252 * 10,
    numSimulacoes:          int   = 1_000_000,
    diasRebalanceamento:    int | None = None,
):
    # ── 1. Download e validação ──
    retornos, tickers = baixar_retornos(tickers, periodo)
    proporcaoAcao     = np.array(proporcaoAcao[:len(tickers)], dtype=float)
    proporcaoAcao    /= proporcaoAcao.sum()

    # ── 2. Calibração ──
    params = calibrar_todos(retornos, tickers)

    # ── 3. Parâmetros RF ──
    rf                    = preparar_parametros_rf(rentabilidadeRendaFixa, frequenciaRendaFixa, diasInvestimento)
    rentabilidadeRFDiaria = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)

    resultado = simular_tempo_para_meta(
        capitalTotal          = capitalTotal,
        meta                  = meta,
        fracao_rv             = fracao_rv,
        proporcaoAcao         = proporcaoAcao,
        tickers               = tickers,
        params                = params,
        rf                    = rf,
        rentabilidadeRFDiaria = rentabilidadeRFDiaria,
        diasInvestimento      = diasInvestimento,
        numSimulacoes         = numSimulacoes,
        diasRebalanceamento   = diasRebalanceamento,
    )

    print(resultado)
    return resultado

    simularTempoParaMeta(
        tickers                = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        proporcaoAcao          = [0.4, 0.35, 0.25],
        capitalTotal           = 100_000,
        meta                   = 200_000,
        fracao_rv              = 0.4,
        rentabilidadeRendaFixa = 0.145,
        frequenciaRendaFixa    = FrequenciaRentabilidadeRendaFix.ANUAL,
    )