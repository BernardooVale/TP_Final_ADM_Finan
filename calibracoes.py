import numpy as np
from scipy import stats, optimize
import pandas as pd
from scipy.optimize import minimize_scalar

from defs import ParametrosCalibrados

def _calibrar_garch(residuos: np.ndarray, sigma2_bar: float) -> tuple[float, float, float]:
    """
    Calibra GARCH(1,1) sobre resíduos padronizados via MLE.

    GARCH(1,1): sigma2_t = omega + alpha * epsilon2_{t-1} + beta * sigma2_{t-1}

    - omega : variância base (piso)
    - alpha : peso do choque recente (reatividade)
    - beta  : peso da variância anterior (persistência)
    - alpha + beta < 1 garante estacionariedade

    sigma2_bar é a variância incondicional de longo prazo, usada para
    inicializar sigma2_0 e como ponto de partida da otimização.

    Retorna (omega, alpha, beta).
    """
    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        T        = len(residuos)
        sigma2   = np.empty(T)
        sigma2[0] = sigma2_bar

        for t in range(1, T):
            sigma2[t] = omega + alpha * residuos[t-1]**2 + beta * sigma2[t-1]

        ll = -0.5 * np.sum(np.log(sigma2) + residuos**2 / sigma2)
        return -ll

    w0     = [sigma2_bar * 0.05, 0.1, 0.85]
    bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.9999)]

    res = optimize.minimize(
        neg_ll, w0,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    return tuple(res.x if res.success else (sigma2_bar * 0.05, 0.1, 0.85))


def _calibrar_t_student(serie: pd.Series) -> tuple[float, float, float, float, float, float]:
    """
    Ajusta t-Student via MLE e calibra GARCH(1,1) sobre os resíduos.

    Retorna (nu, mu, sigma, omega, alpha, beta).
    sigma é o desvio incondicional (longo prazo), usado como sigma2_0.
    """
    nu, mu, sigma = stats.t.fit(serie)
    residuos      = (serie.values - mu) / sigma
    sigma2_bar    = sigma ** 2
    omega, alpha, beta = _calibrar_garch(residuos, sigma2_bar)
    return nu, mu, sigma, float(omega), float(alpha), float(beta)


def _calibrar_nu_copula(retornos: pd.DataFrame) -> float:
    """
    Estima nu da cópula-t via MLE 1D sobre distâncias de Mahalanobis.

    Fluxo:
    1. Uniformes empíricas via rank
    2. Probability integral transform → normais padrão
    3. Distância de Mahalanobis: d_t = x_t^T Sigma^{-1} x_t
    4. MLE 1D sobre d_t para estimar nu

    Captura dependência conjunta de cauda sem custo de MLE multivariada exata.
    """
    n, k      = retornos.shape
    uniformes = retornos.rank() / (n + 1)
    normais   = stats.norm.ppf(uniformes.values)

    corr_emp  = np.corrcoef(normais, rowvar=False) + np.eye(k) * 1e-6
    corr_inv  = np.linalg.inv(corr_emp)
    maha      = np.einsum('ti,ij,tj->t', normais, corr_inv, normais)

    def neg_ll(nu: float) -> float:
        scale = nu / k
        return -float(np.sum(stats.chi2.logpdf(maha * scale, df=nu)))

    res       = minimize_scalar(neg_ll, bounds=(3.0, 50.0), method="bounded")
    nu_copula = float(res.x) if res.success else 10.0
    print(f"  nu cópula calibrado: {nu_copula:.2f} {'(caudas pesadas)' if nu_copula < 10 else '(próximo gaussiana)'}")
    return nu_copula


def calibrar_todos(retornos: pd.DataFrame, tickers: list[str]) -> ParametrosCalibrados:
    """
    Orquestra calibração marginal (t-Student + GARCH) e da cópula-t.

    Retorna ParametrosCalibrados com todos os arrays necessários para simulação.
    """
    fitted = [_calibrar_t_student(retornos[t]) for t in tickers]

    params = ParametrosCalibrados(
        nus      = np.array([f[0] for f in fitted]),
        mus      = np.array([f[1] for f in fitted]),
        sigmas   = np.array([f[2] for f in fitted]),
        omegas   = np.array([f[3] for f in fitted]),
        alphas   = np.array([f[4] for f in fitted]),
        betas    = np.array([f[5] for f in fitted]),
        corr     = retornos.corr().values,
        nu_copula= _calibrar_nu_copula(retornos),
    )

    for t, o, a, b in zip(tickers, params.omegas, params.alphas, params.betas):
        print(f"  GARCH {t}: omega={o:.2e}  alpha={a:.4f}  beta={b:.4f}  persistência={a+b:.4f}")

    return params