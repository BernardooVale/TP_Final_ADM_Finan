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
# Dataclass
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
    
@dataclass
class ParametrosCalibrados:
    nus:       np.ndarray
    mus:       np.ndarray
    sigmas:    np.ndarray
    omegas:    np.ndarray
    alphas:    np.ndarray
    betas:     np.ndarray
    corr:      np.ndarray
    nu_copula: float

@dataclass
class ParametrosRF:
    crescimento:     float
    retorno_periodo: float
    

@dataclass
class BoundsAtivo:
    """
    Limites de alocação por ativo.

    tickers_rv  : bounds para cada ação (min, max), na mesma ordem de `tickers`
    rf          : bounds para a fração total em RF (min, max); None = sem restrição
    
    Exemplo:
        BoundsAtivo(
            tickers_rv = [(0.1, 0.5), (0.1, 0.4), (0.0, 0.3)],
            rf         = (0.2, 0.8),
        )
    """
    tickers_rv: list[tuple[float, float]]         # [(min, max), ...] para cada ativo RV
    rf:         tuple[float, float] | None = None  # (min, max) fração RF no portfólio total


@dataclass
class PontoFronteira:
    """Um ponto na fronteira eficiente CVaR."""
    retorno_alvo:   float          # retorno-alvo do portfólio total no período
    cvar:           float          # CVaR realizado com os pesos ótimos
    retorno_medio:  float          # retorno médio simulado do portfólio total
    fracao_rf:      float          # fração do capital em RF
    fracao_rv:      float          # fração do capital em RV
    pesos_rv:       np.ndarray     # pesos dentro da parcela RV (soma = 1)
    tickers:        list[str]


@dataclass
class FronteiraEficiente:
    """
    Resultado completo da fronteira eficiente CVaR.

    pontos  : lista ordenada por retorno_alvo crescente
    tickers : ativos RV usados
    """
    pontos:  list[PontoFronteira]
    tickers: list[str]

    def to_dict(self) -> list[dict]:
        """Serializa para lista de dicts (fácil de converter em DataFrame)."""
        return [
            {
                "retorno_alvo":  p.retorno_alvo,
                "cvar":          p.cvar,
                "retorno_medio": p.retorno_medio,
                "fracao_rf":     p.fracao_rf,
                "fracao_rv":     p.fracao_rv,
                **{f"peso_{t}": float(w) for t, w in zip(p.tickers, p.pesos_rv)},
            }
            for p in self.pontos
        ]
        
@dataclass
class ResultadoMeta:
    """
    Alocação ótima para atingir uma meta de patrimônio com probabilidade p.

    Campos
    ------
    meta                : patrimônio-alvo (R$)
    probabilidade_alvo  : P(patrimônio >= meta) desejado (ex: 0.80)
    probabilidade_real  : P(patrimônio >= meta) obtido com a alocação encontrada
    alocadoRendaFixa    : parcela em RF (R$)
    alocadoRendaVariavel: parcela em RV (R$)
    patrimonio_p_alvo   : percentil correspondente à probabilidade na distribuição
                          ex: prob=0.80 → P20 da distribuição (cauda inferior 20%)
    atingivel           : False se nem 100% RV atinge a meta com a probabilidade dada
    distribuicaoPatrimonio: patrimônio final em todos os cenários simulados
    """
    capitalTotal:            float
    meta:                    float
    probabilidade_alvo:      float
    probabilidade_real:      float
    alocadoRendaFixa:        float
    alocadoRendaVariavel:    float
    patrimonio_p_alvo:       float
    atingivel:               bool
    distribuicaoPatrimonio:  np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        status = "✓ meta atingível" if self.atingivel else "✗ meta não atingível com os ativos informados"
        pct_rv = self.alocadoRendaVariavel / self.capitalTotal * 100
        pct_rf = self.alocadoRendaFixa    / self.capitalTotal * 100
        cab    = f"\n{'═'*55}\n  Simulação por Meta de Patrimônio\n{'─'*55}\n"
        return (
            f"{cab}"
            f"  Status:              {status}\n"
            f"  Capital total:       R$ {self.capitalTotal:>12,.2f}\n"
            f"  Meta (P*):           R$ {self.meta:>12,.2f}\n"
            f"{'─'*55}\n"
            f"  Alocado em RF:       R$ {self.alocadoRendaFixa:>12,.2f}  ({pct_rf:.1f}%)\n"
            f"  Alocado em RV:       R$ {self.alocadoRendaVariavel:>12,.2f}  ({pct_rv:.1f}%)\n"
            f"{'─'*55}\n"
            f"  Probabilidade alvo:  {self.probabilidade_alvo*100:.0f}%\n"
            f"  Probabilidade real:  {self.probabilidade_real*100:.1f}%\n"
            f"  Patrimônio P{(1-self.probabilidade_alvo)*100:.0f}:     R$ {self.patrimonio_p_alvo:>12,.2f}\n"
            f"{'═'*55}\n"
        )
        
@dataclass
class RestricaoPiso:
    """Patrimônio mínimo aceitável e cobertura desejada."""
    valor:      float   # R$ — ex: 95_000 (não perder mais de 5%)
    confianca:  float   # ex: 0.95  → P(patrimônio >= valor) >= 0.95


@dataclass
class RestricaoMeta:
    """Patrimônio-alvo e probabilidade mínima de atingi-lo."""
    valor:      float   # R$ — ex: 125_000
    confianca:  float   # ex: 0.60  → P(patrimônio >= valor) >= 0.60


@dataclass
class PontoParetoPatrimonio:
    """Um ponto na fronteira de Pareto piso × meta."""
    alocRV:         float   # R$ alocado em RV
    alocRF:         float   # R$ alocado em RF
    prob_piso:      float   # P(patrimônio >= piso) realizado
    prob_meta:      float   # P(patrimônio >= meta) realizado


@dataclass
class ResultadoDuploObjetivo:
    """
    Resultado da otimização com piso e meta simultâneos.

    viavel          : True se existe alguma alocação que satisfaz ambas as restrições
    ponto_minimo_rv : menor alocRV que satisfaz ambas (conservador — maximiza piso)
    ponto_maximo_rv : maior alocRV que satisfaz ambas (agressivo — maximiza meta)
    fronteira       : curva Pareto completa entre os dois extremos
    mensagem        : descrição do resultado ou da restrição inviável
    """
    capitalTotal:    float
    piso:            RestricaoPiso
    meta:            RestricaoMeta
    viavel:          bool
    ponto_minimo_rv: PontoParetoPatrimonio | None
    ponto_maximo_rv: PontoParetoPatrimonio | None
    fronteira:       list[PontoParetoPatrimonio]
    mensagem:        str

    def __str__(self) -> str:
        cab = f"\n{'═'*55}\n  Duplo Objetivo: Piso + Meta\n{'─'*55}\n"
        s = (
            f"{cab}"
            f"  Piso:   R$ {self.piso.valor:>12,.2f}  (confiança {self.piso.confianca*100:.0f}%)\n"
            f"  Meta:   R$ {self.meta.valor:>12,.2f}  (confiança {self.meta.confianca*100:.0f}%)\n"
            f"  Status: {self.mensagem}\n"
        )
        if not self.viavel or self.ponto_minimo_rv is None:
            return s + f"{'═'*55}\n"

        def _fmt_ponto(p: PontoParetoPatrimonio, label: str) -> str:
            pct_rv = p.alocRV / self.capitalTotal * 100
            pct_rf = p.alocRF / self.capitalTotal * 100
            return (
                f"{'─'*55}\n"
                f"  {label}\n"
                f"  RF: R$ {p.alocRF:>10,.2f} ({pct_rf:.1f}%)   "
                f"RV: R$ {p.alocRV:>10,.2f} ({pct_rv:.1f}%)\n"
                f"  P(>= piso): {p.prob_piso*100:.1f}%   "
                f"P(>= meta): {p.prob_meta*100:.1f}%\n"
            )

        s += _fmt_ponto(self.ponto_minimo_rv, "Conservador (mínimo RV)")
        if self.ponto_maximo_rv and abs(self.ponto_maximo_rv.alocRV - self.ponto_minimo_rv.alocRV) > 1:
            s += _fmt_ponto(self.ponto_maximo_rv, "Agressivo   (máximo RV)")

        if len(self.fronteira) > 2:
            s += f"{'─'*55}\n  Fronteira de Pareto ({len(self.fronteira)} pontos):\n"
            s += f"  {'RV (R$)':>12}  {'RF (%)':>6}  {'P(piso)':>8}  {'P(meta)':>8}\n"
            for p in self.fronteira:
                s += (f"  {p.alocRV:>12,.0f}  "
                      f"{p.alocRF/self.capitalTotal*100:>5.1f}%  "
                      f"{p.prob_piso*100:>7.1f}%  "
                      f"{p.prob_meta*100:>7.1f}%\n")

        return s + f"{'═'*55}\n"
    
@dataclass
class ResultadoTempoMeta:
    """
    Distribuição do tempo necessário para atingir uma meta de patrimônio.

    Campos
    ------
    meta                  : patrimônio-alvo (R$)
    prob_atingir          : fração dos cenários que atingem a meta dentro de max_dias
    percentis_dias        : dict {percentil: dias} — só sobre cenários que atingem
    percentis_anos        : idem em anos (dias / 252)
    dias_nao_atingido     : cenários (absoluto) que não atingiram a meta
    max_dias              : horizonte máximo simulado
    dias_por_cenario      : array completo de dias até cruzamento (NaN = não atingiu)
    """
    capitalTotal:       float
    meta:               float
    max_dias:           int
    prob_atingir:       float
    percentis_dias:     dict[int, float]   # {5: x, 10: y, 25: z, 50: w, 75: v, 90: u}
    percentis_anos:     dict[int, float]
    dias_nao_atingido:  int
    dias_por_cenario:   np.ndarray         # shape (n_sim,), NaN onde não atingiu

    def __str__(self) -> str:
        anos_uteis = 252
        cab = f"\n{'═'*55}\n  Tempo para Atingir Meta\n{'─'*55}\n"
        s = (
            f"{cab}"
            f"  Capital inicial:  R$ {self.capitalTotal:>12,.2f}\n"
            f"  Meta:             R$ {self.meta:>12,.2f}  "
            f"(+{(self.meta/self.capitalTotal - 1)*100:.1f}%)\n"
            f"  Horizonte máximo: {self.max_dias} dias úteis "
            f"({self.max_dias/anos_uteis:.1f} anos)\n"
            f"{'─'*55}\n"
            f"  P(atingir meta):  {self.prob_atingir*100:.1f}%  "
            f"({self.dias_nao_atingido:,} cenários não atingiram)\n"
            f"{'─'*55}\n"
            f"  Distribuição do tempo (cenários que atingem):\n"
        )
        for p, anos in self.percentis_anos.items():
            dias = self.percentis_dias[p]
            s += f"    P{p:<2}: {anos:>5.1f} anos  ({dias:>6.0f} dias úteis)\n"
        return s + f"{'═'*55}\n"
    
@dataclass
class ResultadoDesacumulacao:
    """
    Resultado da simulação de saques periódicos (fase de desacumulação).

    Campos
    ------
    saque_simulado      : valor do saque periódico informado (R$)
    saque_sustentavel   : maior saque com prob_ruina <= limite_ruina_alvo (R$)
                          None se nem saque=0 garante sustentabilidade
    prob_ruina          : P(patrimônio zera antes de max_dias) para saque_simulado
    limite_ruina_alvo   : limite de ruína usado na busca da taxa sustentável
    percentis_duracao   : distribuição do tempo até ruína, *apenas* cenários que
                          arruínam (dias úteis e anos)
    prob_sobreviver     : P(patrimônio > 0 ao fim de max_dias) para saque_simulado
    patrimonio_mediano  : patrimônio ao fim do horizonte nos cenários que sobrevivem (P50)
    max_dias            : horizonte máximo simulado
    """
    capitalTotal:         float
    saque_simulado:       float
    frequencia_saque:     "FrequenciaAporte"
    saque_sustentavel:    float | None
    prob_ruina:           float
    limite_ruina_alvo:    float
    percentis_duracao:    dict[str, dict[int, float]]  # {"dias": {p: v}, "anos": {p: v}}
    prob_sobreviver:      float
    patrimonio_mediano:   float | None   # None se todos arruínam
    max_dias:             int

    def __str__(self) -> str:
        anos_uteis = 252
        cab = f"\n{'═'*55}\n  Simulação de Desacumulação (Saques Periódicos)\n{'─'*55}\n"
        s = (
            f"{cab}"
            f"  Capital inicial:     R$ {self.capitalTotal:>12,.2f}\n"
            f"  Saque simulado:      R$ {self.saque_simulado:>12,.2f} "
            f"/ {self.frequencia_saque.value}\n"
            f"  Horizonte máximo:    {self.max_dias} dias úteis "
            f"({self.max_dias/anos_uteis:.1f} anos)\n"
            f"{'─'*55}\n"
            f"  Probabilidade de ruína:   {self.prob_ruina*100:.1f}%\n"
            f"  Probabilidade sobreviver: {self.prob_sobreviver*100:.1f}%\n"
        )
        if self.patrimonio_mediano is not None:
            s += f"  Patrimônio mediano (sobreviventes): R$ {self.patrimonio_mediano:>10,.2f}\n"

        s += f"{'─'*55}\n"
        if self.saque_sustentavel is not None:
            s += (f"  Saque sustentável (ruína ≤ {self.limite_ruina_alvo*100:.0f}%):\n"
                  f"    R$ {self.saque_sustentavel:>10,.2f} / {self.frequencia_saque.value}\n")
        else:
            s += f"  ✗ Nenhum saque sustentável com ruína ≤ {self.limite_ruina_alvo*100:.0f}%\n"

        if self.percentis_duracao["dias"]:
            s += f"{'─'*55}\n  Duração do patrimônio (cenários que arruínam):\n"
            for p, dias in self.percentis_duracao["dias"].items():
                anos = self.percentis_duracao["anos"][p]
                s += f"    P{p:<2}: {anos:>5.1f} anos  ({dias:>6.0f} dias úteis)\n"
        else:
            s += "  (Nenhum cenário arruinou — sem distribuição de duração)\n"

        return s + f"{'═'*55}\n"
    
class TipoEstrategiaBase(str, Enum):
    """Estratégias fixas disponíveis para comparação."""
    RF_100       = "100% RF"
    RV_100       = "100% RV"
    RV75_RF25    = "75% RV / 25% RF"
    RV25_RF75    = "25% RV / 75% RF"


@dataclass
class EstrategiaUsuario:
    """
    Estratégia customizada do usuário para o comparador.

    Aceita qualquer resultado que contenha uma distribuição de patrimônio
    já simulada (AlocacaoResultado ou ResultadoMeta).

    Parâmetros
    ----------
    nome                  : rótulo exibido na tabela comparativa
    distribuicaoPatrimonio: array (n_sim,) de patrimônio final em R$
                            — extraído de resultado.distribuicaoPatrimonio
    fracao_rv             : fração em RV usada (para exibição); None = desconhecido
    """
    nome:                   str
    distribuicaoPatrimonio: np.ndarray
    fracao_rv:              float | None = None


@dataclass
class MetricasEstrategia:
    """Métricas comparativas de uma estratégia."""
    nome:          str
    fracao_rv:     float | None
    q1:            float   # P25 do patrimônio final (R$)
    mediana:       float   # P50
    q3:            float   # P75
    media:         float
    prob_meta:     float | None   # P(patrimônio >= meta); None se meta não definida
    prob_perda:    float          # P(patrimônio < capitalTotal)
    retorno_medio: float          # (media / capitalTotal) - 1


@dataclass
class ResultadoComparador:
    """
    Comparação lado a lado de múltiplas estratégias sobre o mesmo capital.

    Campos
    ------
    estrategias  : métricas de cada estratégia, ordenadas por mediana decrescente
    capitalTotal : capital inicial comum
    meta         : patrimônio-alvo usado em prob_meta (None = não definido)
    """
    estrategias:  list[MetricasEstrategia]
    capitalTotal: float
    meta:         float | None

    def __str__(self) -> str:
        anos_uteis = 252
        cab = f"\n{'═'*71}\n  Comparador de Estratégias\n{'─'*71}\n"
        s   = (f"{cab}"
               f"  Capital inicial: R$ {self.capitalTotal:>12,.2f}\n")
        if self.meta:
            s += f"  Meta:            R$ {self.meta:>12,.2f}\n"
        s += f"{'─'*71}\n"

        # Cabeçalho da tabela
        col_meta   = "P(Meta)" if self.meta else "       "
        s += (f"  {'Estratégia':<22} {'Q1':>10} {'Mediana':>10} {'Q3':>10} "
              f"{'Retorno':>8} {col_meta:>8} {'P(Perda)':>9}\n")
        s += f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*9}\n"

        for e in self.estrategias:
            retorno_str  = f"{e.retorno_medio*100:>+.1f}%"
            prob_meta_str = (
                f"{e.prob_meta*100:>7.1f}%" if e.prob_meta is not None else "      —"
            )
            prob_perda_str = f"{e.prob_perda*100:>8.1f}%"
            s += (f"  {e.nome:<22} "
                  f"R${e.q1:>9,.0f} "
                  f"R${e.mediana:>9,.0f} "
                  f"R${e.q3:>9,.0f} "
                  f"{retorno_str:>8} "
                  f"{prob_meta_str:>8} "
                  f"{prob_perda_str:>9}\n")

        return s + f"{'═'*71}\n"