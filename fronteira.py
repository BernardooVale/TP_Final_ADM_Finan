"""
fronteira.py
============
Fronteira eficiente CVaR para portfólios RF + RV.

Para cada retorno-alvo do portfólio total, encontra a alocação (fração RF/RV
e pesos internos da RV) que minimiza o CVaR, respeitando bounds por ativo.

Diferença de Markowitz
----------------------
- Métrica de risco: CVaR (coerente) em vez de variância
- Distribuição: t-Student + cópula-t + GARCH — sem premissa de normalidade
- RF explícita como ativo com retorno determinístico no período

Variáveis de otimização
-----------------------
x = [y_rf, w_1, w_2, ..., w_n]
    y_rf  : fração do capital total em RF  ∈ [0, 1]
    w_i   : peso do ativo i DENTRO da parcela RV (não relativo ao total)
    Restrição: sum(w_i) = 1, w_i >= 0

Retorno do portfólio total em um cenário s:
    R_s = y_rf * retRF + (1 - y_rf) * sum(w_i * R_i_s)

CVaR minimizado sobre a distribuição de R_s.
"""

import numpy as np
from scipy import optimize
from typing import Callable

from defs import (
    ParametrosCalibrados,
    ParametrosRF,
    BoundsAtivo,
    PontoFronteira,
    FronteiraEficiente,
    RiscoAlvo,
)
from monte_carlo import monteCarlo


# ═══════════════════════════════════════════════════════════════
# Retorno do portfólio total
# ═══════════════════════════════════════════════════════════════

def _retorno_portfolio_total(
    retornos_rv:  np.ndarray,   # (S,) retornos cumulativos RV
    fracao_rf:    float,
    retorno_rf:   float,        # retorno da RF no período (escalar)
) -> np.ndarray:
    """
    Retorno total do portfólio para cada cenário.

    R_total_s = fracao_rf * retRF + (1 - fracao_rf) * retRV_s
    """
    return fracao_rf * retorno_rf + (1 - fracao_rf) * retornos_rv


def _cvar_portfolio(
    retornos_rv:  np.ndarray,
    fracao_rf:    float,
    retorno_rf:   float,
    confianca:    float,
) -> float:
    """CVaR do portfólio total dado fracao_rf."""
    r_total = _retorno_portfolio_total(retornos_rv, fracao_rf, retorno_rf)
    limiar  = np.percentile(r_total, (1 - confianca) * 100)
    tail    = r_total[r_total <= limiar]
    return float(tail.mean()) if len(tail) else 0.0


def _retorno_medio_portfolio(
    retornos_rv: np.ndarray,
    fracao_rf:   float,
    retorno_rf:  float,
) -> float:
    return float(_retorno_portfolio_total(retornos_rv, fracao_rf, retorno_rf).mean())


# ═══════════════════════════════════════════════════════════════
# Bounds
# ═══════════════════════════════════════════════════════════════

def _montar_bounds_slsqp(
    bounds: BoundsAtivo | None,
    n:      int,
) -> list[tuple[float, float]]:
    """
    Monta lista de bounds para SLSQP no espaço [y_rf, w_1, ..., w_n].

    y_rf : bounds.rf ou (0, 1)
    w_i  : bounds.tickers_rv[i] ou (0, 1)
    
    Nota: w_i são pesos INTERNOS à RV (soma = 1 dentro da RV).
    Bounds do usuário em termos de fração do portfólio total precisam
    ser reescalados — mas só é possível se y_rf for fixo. Por simplicidade
    e consistência, tratamos w_i como pesos relativos à RV diretamente.
    Documentar essa premissa ao usuário.
    """
    rf_bounds = bounds.rf if bounds and bounds.rf else (0.0, 1.0)
    rv_bounds = (
        bounds.tickers_rv
        if bounds and bounds.tickers_rv
        else [(0.0, 1.0)] * n
    )
    return [rf_bounds] + rv_bounds


def _montar_restricoes_slsqp() -> list[dict]:
    """sum(w_i) = 1 para os pesos internos da RV (x[1:])."""
    return [{"type": "eq", "fun": lambda x: x[1:].sum() - 1.0}]


# ═══════════════════════════════════════════════════════════════
# Otimização para um retorno-alvo
# ═══════════════════════════════════════════════════════════════

def _otimizar_ponto(
    retorno_alvo: float,
    params:       ParametrosCalibrados,
    rf:           ParametrosRF,
    diasInvest:   int,
    confianca:    float,
    n_sim:        int,
    diasRebal:    int | None,
    bounds:       BoundsAtivo | None,
    z_fixo:       np.ndarray,
    tickers:      list[str],
) -> PontoFronteira | None:
    """
    Minimiza CVaR do portfólio total sujeito a retorno médio >= retorno_alvo.

    x = [y_rf, w_1, ..., w_n]
    Restrições:
        sum(w_i) = 1
        retorno_medio(x) >= retorno_alvo
    """
    n          = len(params.mus)
    bounds_lst = _montar_bounds_slsqp(bounds, n)
    restricoes = _montar_restricoes_slsqp()

    # Cache de simulações RV por vetor de pesos — evita re-simular quando
    # o otimizador testa o mesmo w repetidamente com y_rf diferente
    cache: dict[bytes, np.ndarray] = {}

    def _retornos_rv(w_rv: np.ndarray) -> np.ndarray:
        key = w_rv.tobytes()
        if key not in cache:
            cache[key] = monteCarlo(
                params, w_rv, diasInvest, n_sim, diasRebal, z_fixo,
            )
        return cache[key]

    def objetivo(x: np.ndarray) -> float:
        y_rf = float(np.clip(x[0], 0, 1))
        w_rv = np.abs(x[1:])
        s    = w_rv.sum()
        if s < 1e-12:
            return 0.0
        w_rv /= s
        return _cvar_portfolio(_retornos_rv(w_rv), y_rf, rf.retorno_periodo, confianca)

    restricoes = restricoes + [{
        "type": "ineq",
        "fun": lambda x: (
            _retorno_medio_portfolio(
                _retornos_rv(np.abs(x[1:]) / max(np.abs(x[1:]).sum(), 1e-12)),
                float(np.clip(x[0], 0, 1)),
                rf.retorno_periodo,
            ) - retorno_alvo
        ),
    }]

    # Ponto inicial: equidistribuído
    x0 = np.array([0.5] + [1.0 / n] * n)

    res = optimize.minimize(
        objetivo, x0,
        method  = "SLSQP",
        bounds  = bounds_lst,
        constraints = restricoes,
        options = {"maxiter": 300, "ftol": 1e-6},
    )

    if not res.success:
        return None

    y_rf = float(np.clip(res.x[0], 0, 1))
    w_rv = np.abs(res.x[1:])
    w_rv /= w_rv.sum()

    retornos_rv  = _retornos_rv(w_rv)
    cvar_final   = _cvar_portfolio(retornos_rv, y_rf, rf.retorno_periodo, confianca)
    ret_medio    = _retorno_medio_portfolio(retornos_rv, y_rf, rf.retorno_periodo)

    return PontoFronteira(
        retorno_alvo  = retorno_alvo,
        cvar          = cvar_final,
        retorno_medio = ret_medio,
        fracao_rf     = y_rf,
        fracao_rv     = 1.0 - y_rf,
        pesos_rv      = w_rv,
        tickers       = tickers,
    )


# ═══════════════════════════════════════════════════════════════
# Varredura da fronteira
# ═══════════════════════════════════════════════════════════════

def _estimar_range_retorno(
    params:    ParametrosCalibrados,
    rf:        ParametrosRF,
    diasInvest: int,
    n_sim:     int,
    z_fixo:    np.ndarray,
    diasRebal: int | None,
    n:         int,
) -> tuple[float, float]:
    """
    Estima retorno mínimo (100% RF) e máximo (100% RV equidistribuída)
    para definir o range de varredura automaticamente.
    """
    ret_min = rf.retorno_periodo

    w_eq        = np.ones(n) / n
    ret_rv      = monteCarlo(params, w_eq, diasInvest, n_sim, diasRebal, z_fixo)
    ret_max_rv  = float(np.percentile(ret_rv, 75))  # P75 como teto realista
    ret_max     = _retorno_medio_portfolio(ret_rv, 0.0, rf.retorno_periodo)

    # Garante range com alguma amplitude
    if ret_max <= ret_min:
        ret_max = ret_min * 1.5 + 0.01

    return ret_min, ret_max


def calcular_fronteira(
    params:       ParametrosCalibrados,
    tickers:      list[str],
    rf:           ParametrosRF,
    diasInvest:   int,
    confianca:    float         = 0.95,
    n_pontos:     int           = 15,
    n_sim:        int           = 50_000,
    diasRebal:    int | None    = None,
    bounds:       BoundsAtivo | None = None,
    retornos_alvo: list[float] | None = None,
) -> FronteiraEficiente:
    """
    Calcula a fronteira eficiente CVaR varrendo níveis de retorno-alvo.

    Parâmetros
    ----------
    params        : parâmetros calibrados (t-Student + GARCH + cópula)
    tickers       : ativos RV
    rf            : parâmetros da renda fixa
    diasInvest    : horizonte em dias úteis
    confianca     : nível de confiança do CVaR (ex: 0.95)
    n_pontos      : número de pontos na fronteira
    n_sim         : simulações por ponto (menor que simulacaoPortifolio para viabilidade)
    diasRebal     : rebalanceamento periódico dentro da RV
    bounds        : limites de alocação por ativo
    retornos_alvo : se fornecido, usa esses valores em vez de varrer automaticamente
    """
    n      = len(tickers)
    z_fixo = np.random.standard_normal((n_sim, diasInvest, n))

    if retornos_alvo is None:
        ret_min, ret_max = _estimar_range_retorno(
            params, rf, diasInvest, n_sim, z_fixo, diasRebal, n,
        )
        retornos_alvo = list(np.linspace(ret_min, ret_max, n_pontos))

    print(f"  Fronteira CVaR: {len(retornos_alvo)} pontos, {n_sim} simulações cada")

    pontos: list[PontoFronteira] = []
    for i, alvo in enumerate(retornos_alvo):
        print(f"  Ponto {i+1}/{len(retornos_alvo)}: retorno-alvo={alvo:.4f}", end="\r")
        ponto = _otimizar_ponto(
            alvo, params, rf, diasInvest, confianca,
            n_sim, diasRebal, bounds, z_fixo, tickers,
        )
        if ponto:
            pontos.append(ponto)

    print()
    if not pontos:
        raise ValueError("Nenhum ponto convergiu. Verifique bounds e retornos_alvo.")

    pontos.sort(key=lambda p: p.retorno_alvo)
    return FronteiraEficiente(pontos=pontos, tickers=tickers)