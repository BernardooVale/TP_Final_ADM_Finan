from engine import (
    simular_para_pesos, comparar_estrategias,
    simular_desacumulacao, calcular_fronteira,
    simular_meta_patrimonio, simular_duplo_objetivo,
    otimizar_pesos, simular_tempo_para_meta
    )
from modelos.defs import FrequenciaRentabilidadeRendaFix, RiscoAlvo, FrequenciaAporte

def alocacao(
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
):
    pass

def comparacao():
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