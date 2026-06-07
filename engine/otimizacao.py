import numpy as np
from scipy import optimize

from modelos.params import ParametrosCalibrados
from monte_carlo import monteCarlo
from metricas import _cvar

def _normalizar_pesos(w: np.ndarray) -> np.ndarray:
    """Garante pesos positivos normalizados para somar 1."""
    return np.abs(w) / np.abs(w).sum()

def otimizar_pesos(
    params:              ParametrosCalibrados,
    diasInvestimento:    int,
    confianca:           float,
    numSimulacoes:       int,
    diasRebalanceamento: int | None,
    poupaTempo:          bool,
) -> np.ndarray:
    """
    Pesos da carteira RV que minimizam CVaR via Nelder-Mead.

    z_fixo pré-gerado garante superfície objetivo determinística —
    evita que o otimizador persiga ruído entre iterações.
    poupaTempo reduz n_sim e afrouxa tolerâncias mantendo z fixo.
    """
    n     = len(params.mus)
    n_sim = numSimulacoes // 20 if poupaTempo else numSimulacoes // 5
    opts  = (
        {"maxiter": 150, "xatol": 1e-2, "fatol": 1e-3} if poupaTempo
        else {"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4}
    )
    z_fixo   = np.random.standard_normal((n_sim, diasInvestimento, n))
    contador = [0]

    print("  Otimizando pesos (minimização de CVaR)...")

    def objetivo(w: np.ndarray) -> float:
        contador[0] += 1
        print(f"  Iteração {contador[0]}", end="\r")
        ret = monteCarlo(params, _normalizar_pesos(w), diasInvestimento, n_sim, diasRebalanceamento, z_fixo)
        return _cvar(ret, confianca)

    res   = optimize.minimize(objetivo, np.ones(n) / n, method="Nelder-Mead", options=opts)
    w_opt = _normalizar_pesos(res.x)

    print()
    print(f"  Pesos otimizados: { {i: round(float(w), 4) for i, w in enumerate(w_opt)} }")
    return w_opt