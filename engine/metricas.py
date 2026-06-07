import numpy as np
from modelos.defs import RiscoAlvo

def _cvar(retornos: np.ndarray, confianca: float) -> float:
    """CVaR: média dos retornos abaixo do percentil (1 - confiança)."""
    limiar   = np.percentile(retornos, (1 - confianca) * 100)
    tail     = retornos[retornos <= limiar]
    return float(tail.mean()) if len(tail) else 0.0

def _calcular_sharpe_sortino(
    retornosCumulativos: np.ndarray,
    retorno_rf_periodo:  float,
    diasInvestimento:    int,
) -> tuple[float, float]:
    """
    Sharpe e Sortino anualizados da carteira simulada.

    Anualizacao
    -----------
    Retornos cumulativos do período são convertidos para equivalente anual
    antes do cálculo, tornando os índices comparáveis entre horizontes distintos.

        fator = 252 / diasInvestimento
        retorno_anual = (1 + retorno_periodo) ** fator - 1

    Sharpe  = (E[excesso_anual]) / std(excesso_anual)
    Sortino = (E[excesso_anual]) / std(excesso_anual | excesso < 0)

    Sortino penaliza apenas volatilidade negativa — mais justo para
    distribuições assimétricas típicas de ativos de renda variável.
    """
    fator = 252 / diasInvestimento

    rv_anual = (1 + retornosCumulativos) ** fator - 1
    rf_anual = (1 + retorno_rf_periodo)  ** fator - 1

    excesso = rv_anual - rf_anual
    sharpe  = excesso.mean() / (excesso.std() + 1e-12)

    downside = excesso[excesso < 0]
    dd_std   = downside.std() if len(downside) > 1 else 1e-12
    sortino  = excesso.mean() / (dd_std + 1e-12)

    return float(sharpe), float(sortino)

def _retorno_cenario_alvo(
    retornosCumulativos: np.ndarray,
    confianca:           float,
    riscoAlvo:           RiscoAlvo,
) -> float:
    """
    Perda esperada da RV no tail de risco.

    MEDIA → CVaR clássico (média do tail).
    PIOR  → mínimo absoluto do tail (CVaR extremo).
    """
    limiar   = np.percentile(retornosCumulativos, (1 - confianca) * 100)
    cenarios = retornosCumulativos[retornosCumulativos <= limiar]

    if not len(cenarios):
        return 0.0

    return float(cenarios.mean()) if riscoAlvo == RiscoAlvo.MEDIA else float(cenarios.min())