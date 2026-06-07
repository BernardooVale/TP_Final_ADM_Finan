from modelos.defs import (
    FrequenciaRentabilidadeRendaFix,
    FrequenciaAporte,
    DIAS_UTEIS_APORTE,
    DIAS_UTEIS_RF,
)
from modelos.dataclass import ParametrosRF

def _taxa_diaria_rf(rate: float, frequencia: FrequenciaRentabilidadeRendaFix) -> float:
    """
    Converte taxa RF para equivalente diária via juros compostos.

    taxa_diaria = (1 + rate)^(1/n) - 1
    onde n = dias úteis da frequência informada.
    """
    return (1 + rate) ** (1 / DIAS_UTEIS_RF[frequencia]) - 1


def _taxa_ciclo_rf(
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
) -> float:
    """Taxa equivalente a um ciclo completo do título RF."""
    diaria    = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)
    ciclo_rf  = DIAS_UTEIS_RF[frequenciaRendaFixa]
    return (1 + diaria) ** ciclo_rf - 1


def _ciclos_completos(dias_restantes: int, ciclo_rf: int) -> int:
    """Quantos ciclos completos do título cabem nos dias restantes."""
    return dias_restantes // ciclo_rf


def _valor_futuro_aportes(
    valorAporte: float,
    frequenciaAporte: FrequenciaAporte,
    diasInvestimento: int,
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
) -> float:
    """
    Valor futuro total dos aportes periódicos investidos em RF.

    Aporte só rende em ciclos COMPLETOS da frequência do título —
    dias restantes que não fecham um ciclo ficam no valor nominal.
    Reflete CDB/LCI/LCA que pagam apenas no vencimento ou cupom fixo.
    """
    intervalo  = DIAS_UTEIS_APORTE[frequenciaAporte]
    ciclo_rf   = DIAS_UTEIS_RF[frequenciaRendaFixa]
    taxa_ciclo = _taxa_ciclo_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)

    total = 0.0
    for d in range(intervalo, diasInvestimento, intervalo):
        ciclos = _ciclos_completos(diasInvestimento - d, ciclo_rf)
        total += valorAporte * (1 + taxa_ciclo) ** ciclos

    return total


def preparar_parametros_rf(
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
    diasInvestimento: int,
) -> ParametrosRF:
    """Computa fator de crescimento e retorno percentual da RF no período."""
    diario        = _taxa_diaria_rf(rentabilidadeRendaFixa, frequenciaRendaFixa)
    crescimento   = (1 + diario) ** diasInvestimento
    return ParametrosRF(crescimento=crescimento, retorno_periodo=crescimento - 1)


def preparar_aportes(
    valorAporte: float,
    frequenciaAporte: FrequenciaAporte | None,
    diasInvestimento: int,
    rentabilidadeRendaFixa: float,
    frequenciaRendaFixa: FrequenciaRentabilidadeRendaFix,
) -> float:
    """Retorna capital futuro dos aportes, ou 0 se não configurados."""
    if valorAporte <= 0 or frequenciaAporte is None:
        return 0.0

    capital = _valor_futuro_aportes(
        valorAporte, frequenciaAporte, diasInvestimento,
        rentabilidadeRendaFixa, frequenciaRendaFixa,
    )
    print(f"  Aportes: valor futuro total em RF = R$ {capital:,.2f}")
    return capital