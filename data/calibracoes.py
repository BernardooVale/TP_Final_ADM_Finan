import numpy as np
from scipy import stats
import pandas as pd
from scipy.optimize import minimize_scalar

from modelos.params import ParametrosCalibrados

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
    
    fitted = [stats.t.fit(retornos[t]) for t in tickers] # t_Student

    params = ParametrosCalibrados(
        nus      = np.array([f[0] for f in fitted]),
        mus      = np.array([f[1] for f in fitted]),
        sigmas   = np.array([f[2] for f in fitted]),
        corr     = retornos.corr().values,
        nu_copula= _calibrar_nu_copula(retornos),
    )

    return params