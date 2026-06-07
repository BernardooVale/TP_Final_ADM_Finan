from dataclasses import dataclass
import numpy as np

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