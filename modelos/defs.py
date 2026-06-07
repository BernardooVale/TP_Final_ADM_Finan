from enum import Enum

# ═══════════════════════════════════════════════════════════════
# Enumerações
# ═══════════════════════════════════════════════════════════════

class RiscoAlvo(str, Enum):
    """Define como a perda esperada da RV é calculada nos cenários ruins."""
    MEDIA = "media"  # média dos cenários abaixo do limiar de confiança
    PIOR  = "pior"   # mínimo absoluto dos cenários abaixo do limiar (CVaR extremo)


class FrequenciaRentabilidadeRendaFix(str, Enum):
    """Frequência em que a taxa da RF é expressa."""
    DIARIO     = "diario"
    MENSAL     = "mensal"
    TRIMESTRAL = "trimestral"
    ANUAL      = "anual"


class FrequenciaAporte(str, Enum):
    """Intervalo entre aportes periódicos."""
    MENSAL     = "mensal"      # a cada 21 dias úteis
    TRIMESTRAL = "trimestral"  # a cada 63 dias úteis
    SEMESTRAL  = "semestral"   # a cada 126 dias úteis

# ═══════════════════════════════════════════════════════════════
# Constantes de mapeamento
# ═══════════════════════════════════════════════════════════════

# Dias úteis equivalentes por frequência de RF
DIAS_UTEIS_RF: dict[FrequenciaRentabilidadeRendaFix, int] = {
    FrequenciaRentabilidadeRendaFix.DIARIO:      1,
    FrequenciaRentabilidadeRendaFix.MENSAL:     21,
    FrequenciaRentabilidadeRendaFix.TRIMESTRAL:  63,
    FrequenciaRentabilidadeRendaFix.ANUAL:      252,
}

# Dias úteis equivalentes por frequência de aporte
DIAS_UTEIS_APORTE: dict[FrequenciaAporte, int] = {
    FrequenciaAporte.MENSAL:     21,
    FrequenciaAporte.TRIMESTRAL:  63,
    FrequenciaAporte.SEMESTRAL:  126,
}