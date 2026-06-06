from enum import Enum
from dataclasses import dataclass, field
import numpy as np

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


# ═══════════════════════════════════════════════════════════════
# Dataclass de resultado
# ═══════════════════════════════════════════════════════════════

@dataclass
class AlocacaoResultado:
    """
    Contém todos os dados de saída da simulação.

    Campos principais
    -----------------
    capitalTotal               : capital inicial investido
    alocadoRendaFixa           : parcela alocada em RF
    alocadoRendaVariavel       : parcela alocada em RV
    saldoFinalRendaFixa        : valor da RF ao fim do horizonte (inclui aportes)
    perdaEsperadaRendaVariavel : perda da RV no cenário alvo
    patrimonioFinal            : RF final + RV após perda esperada
    distribuicaoPatrimonio     : patrimônio final em todos os cenários simulados
    sharpe / sortino           : métricas de risco/retorno da carteira RV simulada
    otimizado                  : resultado com pesos otimizados (None se não solicitado)
    """
    capitalTotal:               float
    alocadoRendaFixa:           float
    alocadoRendaVariavel:       float
    saldoFinalRendaFixa:        float
    perdaEsperadaRendaVariavel: float
    patrimonioFinal:            float
    confianca:                  float
    riscoAlvo:                  RiscoAlvo
    diasInvestimento:           int
    proporcaoAcao:              np.ndarray
    tickers:                    list[str]
    distribuicaoPatrimonio:     np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe:                     float = 0.0
    sortino:                    float = 0.0
    otimizado:                  "AlocacaoResultado | None" = None

    # ── Formatação do output ──

    def _str_percentis(self) -> str:
        """Formata os percentis P5/P25/P50/P75/P95 da distribuição do patrimônio final."""
        if not len(self.distribuicaoPatrimonio):
            return ""
        percentis = np.percentile(self.distribuicaoPatrimonio, [5, 25, 50, 75, 95])
        linhas = ["\n  Distribuição do patrimônio final (R$):"]
        for p, val in zip([5, 25, 50, 75, 95], percentis):
            linhas.append(f"    P{p:>2}:  R$ {val:>12,.2f}")
        return "\n".join(linhas)

    def _str_bloco(self, titulo: str = "") -> str:
        """
        Monta o bloco de texto formatado para exibição no terminal.
        O cabeçalho (cab) inclui linha decorativa e título opcional.
        """
        cab = f"\n{'═'*55}\n"
        if titulo:
            cab += f"  {titulo}\n{'─'*55}\n"

        pesos_fmt = "  ".join(
            f"{t}: {p*100:.1f}%" for t, p in zip(self.tickers, self.proporcaoAcao)
        )
        return (
            f"{cab}"
            f"  Pesos RV:            {pesos_fmt}\n"
            f"  Capital total:       R$ {self.capitalTotal:>12,.2f}\n"
            f"  Alocado em RF:       R$ {self.alocadoRendaFixa:>12,.2f}  ({self.alocadoRendaFixa/self.capitalTotal*100:.1f}%)\n"
            f"  Alocado em RV:       R$ {self.alocadoRendaVariavel:>12,.2f}  ({self.alocadoRendaVariavel/self.capitalTotal*100:.1f}%)\n"
            f"{'─'*55}\n"
            f"  RF ao fim ({self.diasInvestimento}d):   R$ {self.saldoFinalRendaFixa:>12,.2f}\n"
            f"  Perda esperada RV:   R$ {self.perdaEsperadaRendaVariavel:>12,.2f}\n"
            f"{'─'*55}\n"
            f"  Patrimônio final:    R$ {self.patrimonioFinal:>12,.2f}\n"
            f"  Cobertura:           {'✓ >= capital inicial' if self.patrimonioFinal >= self.capitalTotal else '✗ insuficiente'}\n"
            f"  Cenário:             {self.riscoAlvo.value} | confiança {self.confianca*100:.0f}%\n"
            f"  Sharpe:              {self.sharpe:.4f}\n"
            f"  Sortino:             {self.sortino:.4f}\n"
            f"{self._str_percentis()}\n"
            f"{'═'*55}\n"
        )

    def __str__(self) -> str:
        s = self._str_bloco("Pesos originais")
        if self.otimizado is not None:
            s += self.otimizado._str_bloco("Pesos otimizados (mín. CVaR)")
        return s