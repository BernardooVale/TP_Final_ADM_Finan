from engine import (
    simular_para_pesos, comparar_estrategias,
    simular_desacumulacao, calcular_fronteira,
    simular_meta_patrimonio, simular_duplo_objetivo,
    otimizar_pesos, simular_tempo_para_meta
    )
from modelos.defs import FrequenciaRentabilidadeRendaFix, RiscoAlvo, FrequenciaAporte
from data.mineracao import baixar_retornos
from modelos.params import Simulacao, ParametrosRF, ParametrosCalibrados
from data.calibracoes import calibrar_todos
from renda_fixa import preparar_aportes
from modelos.estrategias import EstrategiaUsuario, TipoEstrategiaBase, MetricasEstrategia

import pandas as pd
import numpy as np

def alocacao(
    capitalTotal:           float,
    tickers:                list[str],
    proporcaoAcao:          list[float],
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa:    FrequenciaRentabilidadeRendaFix,
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
    
    paramsRF = ParametrosRF(rentabilidadeRendaFixa, frequenciaRendaFixa)
    capitalAportes = preparar_aportes(
        valorAporte, frequenciaAporte, diasInvestimento,
        rentabilidadeRendaFixa, frequenciaRendaFixa,
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
    pass

def desacumulacao():
    pass

def fronteira():
    pass

def meta():
    pass

def duploObjetivo():
    pass

def alocacaoOtimizada():
    pass

def tempoMeta():
    pass

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