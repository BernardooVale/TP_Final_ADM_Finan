import numpy as np
from scipy import stats
import pandas as pd
from scipy.optimize import minimize_scalar

from modelos.params import ParametrosCalibrados

def _calibrar_nu_copula(
    retornos: pd.DataFrame,
    fitted:   list[tuple[float, float, float]],  # (nu, mu, sigma) por ativo
) -> float:
    """
    Estima nu da cópula-t via MLE 1D sobre distâncias de Mahalanobis.

    Ordem correta pelo Teorema de Sklar:
    1. Passa retornos pela CDF das marginais t-Student já ajustadas
       → uniformes "limpas" de escala e volatility clustering
    2. Inverte para normais padrão via PPF normal
    3. Calcula distâncias de Mahalanobis sobre essas normais
    4. MLE 1D sobre as distâncias para estimar nu

    Usar retornos brutos (rank empírico) confunde agrupamento de volatilidade
    macroeconômica com dependência de cauda — empurra nu para o bound inferior.
    """
    n, k = retornos.shape

    # ── 1. Uniformes via CDF das marginais ajustadas (Sklar correto) ──
    uniformes = np.empty((n, k))
    for i, (col, (nu_i, mu_i, sigma_i)) in enumerate(zip(retornos.columns, fitted)):
        uniformes[:, i] = stats.t.cdf(retornos[col].values, df=nu_i, loc=mu_i, scale=sigma_i)

    # Clipa para evitar 0 e 1 exatos que produzem ±inf no PPF
    uniformes = np.clip(uniformes, 1e-6, 1 - 1e-6)

    # ── 2. Normais padrão via PPF ──
    normais = stats.norm.ppf(uniformes)

    # ── 3. Distâncias de Mahalanobis ──
    corr_emp = np.corrcoef(normais, rowvar=False) + np.eye(k) * 1e-6
    corr_inv = np.linalg.inv(corr_emp)
    maha     = np.einsum('ti,ij,tj->t', normais, corr_inv, normais)

    # ── 4. MLE 1D sobre nu ──
    # d²/k ~ F(k, nu) para dados vindos de t-multivariada — não chi2
    # Usar chi2 era matematicamente incorreto e empurrava nu para o bound inferior
    f_stat = maha / k

    def neg_ll(nu: float) -> float:
        return -float(np.sum(stats.f.logpdf(f_stat, dfn=k, dfd=nu)))

    res       = minimize_scalar(neg_ll, bounds=(4.0, 50.0), method="bounded")
    nu_copula = float(res.x) if res.success else 10.0
    print(f"  nu cópula calibrado: {nu_copula:.2f} {'(caudas pesadas)' if nu_copula < 10 else '(próximo gaussiana)'}")
    return nu_copula

def _calibrar_t_student(serie: pd.Series) -> tuple[float, float, float]:
    """Ajusta t-Student via MLE. Retorna (nu, mu, sigma)."""
    nu, mu, sigma = stats.t.fit(serie)
    return nu, mu, sigma

def calibrar_todos(retornos: pd.DataFrame, tickers: list[str]) -> ParametrosCalibrados:
    fitted = [_calibrar_t_student(retornos[t]) for t in tickers]
    return ParametrosCalibrados(
        nus       = np.array([f[0] for f in fitted]),
        mus       = np.array([f[1] for f in fitted]),
        sigmas    = np.array([f[2] for f in fitted]),
        corr      = retornos.corr().values,
        nu_copula = _calibrar_nu_copula(retornos, fitted),  # ← passa fitted
    )