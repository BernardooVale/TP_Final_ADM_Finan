from modelos.defs import (
    FrequenciaRentabilidadeRendaFix,
    FrequenciaAporte,
    DIAS_UTEIS_APORTE,
    DIAS_UTEIS_RF,
)
from modelos.params import ParametrosRF

def _taxa_diaria_rf(rate: float, frequencia: FrequenciaRentabilidadeRendaFix) -> float:
    """Converte taxa RF para equivalente diária via juros compostos."""
    return (1 + rate) ** (1 / DIAS_UTEIS_RF[frequencia]) - 1


def _valor_futuro_aportes(
    valorAporte: float,
    frequenciaAporte: FrequenciaAporte,
    diasInvestimento: int,
    taxa_diaria: float,
) -> float:
    """
    Valor futuro total dos aportes periódicos investidos em RF com capitalização diária.
    Cada aporte rende pelo número exato de dias úteis que restam até o fim do prazo.
    """
    intervalo = DIAS_UTEIS_APORTE[frequenciaAporte]
    total = 0.0
    
    for d in range(intervalo, diasInvestimento, intervalo):
        dias_restantes = diasInvestimento - d
        total += valorAporte * (1 + taxa_diaria) ** dias_restantes

    return total


def preparar_parametros_rf(
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
    diasInvestimento: int,
) -> ParametrosRF:
    """Computa fator de crescimento, retorno do período e preserva a taxa diária."""
    diario = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)
    crescimento = (1 + diario) ** diasInvestimento
    
    return ParametrosRF(
        crescimento=crescimento, 
        retorno_periodo=crescimento - 1, 
        taxa_diaria=diario
    )


def preparar_aportes(
    valorAporte: float,
    frequenciaAporte: FrequenciaAporte | None,
    diasInvestimento: int,
    taxa_diaria: float,
) -> float:
    """Retorna capital futuro dos aportes sob a ótica de capitalização diária."""
    if valorAporte <= 0 or frequenciaAporte is None:
        return 0.0

    capital = _valor_futuro_aportes(
        valorAporte, frequenciaAporte, diasInvestimento, taxa_diaria
    )
    print(f"  Aportes: valor futuro total em RF = R$ {capital:,.2f}")
    return capital