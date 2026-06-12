from engine import (
    simular_para_pesos, comparar_estrategias,
    simular_desacumulacao, calcular_fronteira,
    simular_meta_patrimonio, simular_duplo_objetivo,
    otimizar_pesos, simular_tempo_para_meta
    )
from modelos.defs import FrequenciaRentabilidadeRendaFix, RiscoAlvo, FrequenciaAporte
from data.mineracao import baixar_retornos
from modelos.params import Simulacao, ParametrosRF, ParametrosCalibrados, BoundsAtivo
from data.calibracoes import calibrar_todos
from renda_fixa import preparar_aportes, _taxa_diaria_rf
from modelos.estrategias import EstrategiaUsuario, TipoEstrategiaBase
from modelos.pareto import RestricaoPiso, RestricaoMeta

import numpy as np

def alocacao(
    capitalTotal:           float,
    tickers:                list[str],
    proporcaoAcao:          list[float],
    paramsRF:               ParametrosRF,
    params:                 ParametrosCalibrados,
    riscoAlvo:              RiscoAlvo,
    diasInvestimento:       int,
    confianca:              float = 0.95,
    numSimulacoes:          int = 1_000_000,
    diasRebalanceamento:    int | None = None,
    valorAporte:            float = 0.0,
    frequenciaAporte:       FrequenciaAporte | None = None,
    retornosCumulativos:    np.ndarray | None = None,
):
    capitalAportes = preparar_aportes(
        valorAporte, frequenciaAporte, diasInvestimento,
        paramsRF.crescimento, paramsRF.retorno_periodo
    )
    
    resultado, retornosCumulativos = simular_para_pesos(capitalTotal, proporcaoAcao, tickers, params, paramsRF, riscoAlvo, diasInvestimento, confianca, numSimulacoes, diasRebalanceamento, capitalAportes, retornosCumulativos)
    
    simulacaoPrevia = Simulacao(0, retornosCumulativos)
    
    return resultado, simulacaoPrevia

def comparacao(
    capitalTotal: float,
    propocaoAcao: np.ndarray,
    params: ParametrosCalibrados,
    rf: ParametrosRF,
    diasInvestimento: int,
    estrategia_usuario: EstrategiaUsuario | None = None,
    estrategias_base: list[TipoEstrategiaBase] = list[TipoEstrategiaBase],
    meta: float | None = None,
    numSimulacoes: int = 1_000_000,
    diasRebalanceamento: int | None = None,
    retornosCumulativos: np.ndarray | None = None
):
    
    resultado, retornosCumulativos = comparar_estrategias(
        capitalTotal,
        propocaoAcao,
        params,
        rf,
        diasInvestimento,
        estrategia_usuario,
        estrategias_base,
        meta,
        numSimulacoes,
        diasRebalanceamento,
        retornosCumulativos
    )
    
    simulacaoPrevia = Simulacao(0, retornosCumulativos)
    
    return resultado, simulacaoPrevia

def desacumulacao(
    capitalTotal: float,
    saque: float,
    frequenciaSaque: FrequenciaAporte,
    fracao_rv: float,
    propocaoAcao: np.ndarray,
    tickers: list[str],
    params: ParametrosCalibrados,
    rf: ParametrosRF,
    diasInvestimento: int,
    numSimulacoes: int = 1_000_000,
    diasRebalanceamento: int | None = None,
    limite_ruina: float = 0.0,
    percentis_duracao: list[int] = [10,25,50,75,90],
    tol_saque: float = 1.0,
):
    
    resultado = simular_desacumulacao(
        capitalTotal,
        saque,
        frequenciaSaque,
        fracao_rv,
        propocaoAcao,
        tickers,
        params,
        rf,
        _taxa_diaria_rf(rf.crescimento, rf.retorno_periodo),
        diasInvestimento,
        numSimulacoes,
        diasRebalanceamento,
        limite_ruina,
        percentis_duracao,
        tol_saque
    )
    
    return resultado

def fronteira(
    params: ParametrosCalibrados,
    tickers: list[str],
    rf: ParametrosRF,
    diasInvest: int,
    confianca: float,
    numPontos: int = 5,
    numSimulacoes: int = 100_000,
    diasRebalanceamento: int | None = None,
    bounds: BoundsAtivo | None = None,
    retornos_alvo: list[float] | None = None
):
    
    resultado = calcular_fronteira(
        params,
        tickers,
        rf,
        diasInvest,
        confianca,
        numPontos,
        numSimulacoes,
        diasRebalanceamento,
        bounds,
        retornos_alvo
    )
    
    return resultado

def meta(
    capitalTotal: float,
    meta: float,
    probabilidade: float,
    proporcaoAcao: np.ndarray,
    params: ParametrosCalibrados,
    rf: ParametrosRF,
    diasInvestimento: int,
    numSimulacoes: int = 1_000_000,
    diasRebalanceamento: int | None = None,
    capitalAportes: float = 0.0,
    tol: float = 1,
    retornosCumulativos:    np.ndarray | None = None,
):
    
    resultado, retornosCumulativos = simular_meta_patrimonio(
        capitalTotal,
        meta,
        probabilidade,
        proporcaoAcao,
        params,
        rf,
        diasInvestimento,
        numSimulacoes,
        diasRebalanceamento,
        capitalAportes,
        tol,
        retornosCumulativos
    )
    
    simulacaoPrevia = Simulacao(0, retornosCumulativos)
    
    return resultado, simulacaoPrevia

def duploObjetivo(
    capitalTotal: float,
    piso: RestricaoPiso,
    meta: RestricaoMeta,
    proporcaoAcao: np.ndarray,
    tickers: list[str],
    params: ParametrosCalibrados,
    rf: ParametrosRF,
    diasInvestimento: int,
    numSimulacoes: int = 1_000_000,
    diasRebalanceamento: int | None = None,
    capitalAportes: float = 0.0,
    n_pontos_pareto: int = 5,
    tol: float = 1,
    retornosCumulativos: np.ndarray | None = None
):
    
    resultado, retornosCumulativos = simular_duplo_objetivo(
        capitalTotal,
        piso,
        meta,
        proporcaoAcao,
        tickers,
        params,
        rf,
        diasInvestimento,
        numSimulacoes,
        diasRebalanceamento,
        capitalAportes,
        n_pontos_pareto,
        tol
    )
    
    simulacaoPrevia = Simulacao(0, retornosCumulativos)
    
    return resultado, simulacaoPrevia

def alocacaoOtimizada(
    params: ParametrosCalibrados,
    diasInvestimento: int,
    confianca: float,
    numSimulacoes: int,
    diasRebalanceamento: int | None,
    poupaTempo: bool,
    resultadosCumulativos: np.ndarray | None = None
):

    resultado, resultadosCumulativos = otimizar_pesos(
        params,
        diasInvestimento,
        confianca,
        numSimulacoes,
        diasRebalanceamento,
        poupaTempo,
        resultadosCumulativos
    )

def tempoMeta(
    capitalTotal: float,
    meta: float,
    fracao_rv: float,
    proporcaoAcao: np.ndarray,
    tickers: list[str],
    params: ParametrosCalibrados,
    rf: ParametrosRF,
    diasInvestimento: int,
    numSimulacoes: int = 1_000_000,
    diasRebalanceamento: int | None = None,
    valorAporte: float = 0,
    frequenciaAporte: FrequenciaAporte | None = None,
    percentis: list[int] = [5,10,25,50,75,90],
    chunk_size: int = 50_000
):
    return simular_tempo_para_meta(
        capitalTotal,
        meta,
        fracao_rv,
        proporcaoAcao,
        tickers,
        params,
        rf,
        _taxa_diaria_rf(rf.crescimento, rf.retorno_periodo),
        diasInvestimento,
        numSimulacoes,
        diasRebalanceamento,
        valorAporte,
        frequenciaAporte,
        percentis,
        chunk_size
    )

def controle():
    while True:
        
        resposta = input(
            
            """
            Escolha a funcionalidade
            (1) - Quanto alocar em RF e RV (simular_para_pesos)
            (2) - Comparar estrategias
            (3) - simular desacumulacao
            (4) - Fronteira de pareto
            (5) - meta patrimonio
            (6) - simular duplo objetivo
            (7) - Quanto alocar otimizando carteira
            (8) - simular tempo para meta
            \n
            """       
        )
        
        if resposta == 1:
            alocacao()
        elif resposta == 2:
            comparacao()
        elif resposta == 3:
            desacumulacao()
        elif resposta == 4:
            fronteira()
        elif resposta == 5:
            meta()
        elif resposta == 6:
            duploObjetivo()
        elif resposta == 7:
            alocacaoOtimizada()
        elif resposta == 8:
            tempoMeta()
        else: 
            break
        
acoes = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
propocao = [0.4, 0.35, 0.25]
rentabilidadeRendaFixa = 0.145
diasInvestimento = 252
periodo = "3y"

retornos, tickers = baixar_retornos(acoes, periodo)
params = calibrar_todos(retornos, tickers)
paramsRF = ParametrosRF(rentabilidadeRendaFixa, FrequenciaRentabilidadeRendaFix.ANUAL)