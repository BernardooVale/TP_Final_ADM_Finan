import numpy as np
from scipy import stats

from modelos.params import ParametrosCalibrados
from engine.kernels import _gerar_inovacoes_njit

_T_GRID_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}

_T_GRID_N      = 20_000   # pontos na grade — erro de interpolação < 1e-6
_T_GRID_TAIL   = 1 - 1e-7 # cobre até P(0.0000001) e P(0.9999999)

def _t_grid(df: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (x_grid, u_grid) para a t-Student com `df` graus de liberdade.

    x_grid : valores de x uniformemente espaçados entre os quantis extremos
    u_grid : CDF(x_grid, df) — permite CDF via np.interp(x, x_grid, u_grid)
                              e PPF via np.interp(u, u_grid, x_grid)

    Grade construída uma vez e cacheada por df (arredondado a 2 casas).
    """
    key = round(df, 2)
    if key not in _T_GRID_CACHE:
        x_lo = stats.t.ppf(1 - _T_GRID_TAIL, df)
        x_hi = stats.t.ppf(_T_GRID_TAIL,     df)
        x_grid = np.linspace(x_lo, x_hi, _T_GRID_N)
        u_grid = stats.t.cdf(x_grid, df)
        _T_GRID_CACHE[key] = (x_grid, u_grid)
    return _T_GRID_CACHE[key]


def _gerar_inovacoes(
    z:         np.ndarray,
    chol:      np.ndarray,
    nu_copula: float,
    nus:       np.ndarray,
) -> np.ndarray:
    """
    Transforma ruído normal em inovações marginais t-Student correlacionadas.

    Fluxo:
    1. Cópula-t: z / sqrt(chi2 / nu_copula) — dependência de cauda conjunta
    2. Correlaciona via Cholesky: y = t_copula @ L^T
    3. Uniformes via CDF_t(nu_copula) — lookup table, sem scipy no caminho quente
    4. PPF marginal t-Student por ativo — lookup table, sem scipy no caminho quente

    CDF e PPF aproximados por interpolação linear sobre grade fina (20k pontos).
    Erro máximo < 1e-6 para |x| dentro dos quantis P(1e-7) a P(1-1e-7).
    Valores fora da grade são clipados aos extremos (eventos < 1e-7 de prob).
    """
    S, D, A = z.shape

    chi2_shared = np.random.chisquare(nu_copula, size=(S, D, 1)) / nu_copula
    t_copula    = z / np.sqrt(chi2_shared)
    y           = t_copula @ chol.T

    y_flat = y.reshape(-1, A)
    u_flat = np.empty_like(y_flat)

    # CDF da cópula — mesma df para todos os ativos neste passo
    x_grid_cop, u_grid_cop = _t_grid(nu_copula)
    for i in range(A):
        u_flat[:, i] = np.interp(y_flat[:, i], x_grid_cop, u_grid_cop)

    x_flat = np.empty_like(u_flat)

    # PPF marginal — df pode diferir por ativo
    for i in range(A):
        x_grid_i, u_grid_i = _t_grid(float(nus[i]))
        # np.interp com (u, u_grid, x_grid) inverte a grade → PPF
        x_flat[:, i] = np.interp(u_flat[:, i], u_grid_i, x_grid_i)

    return x_flat.reshape(S, D, A)

def _higham_nearest_corr(corr: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """
    Nearest Correlation Matrix via projeções alternadas de Dykstra (Higham 2002).

    Garante:
    - Simétrica
    - Diagonal exatamente 1
    - Positiva semi-definida (menor autovalor >= 0)

    Referência: Higham, N.J. (2002). "Computing the nearest correlation matrix —
    a problem from finance." IMA Journal of Numerical Analysis, 22(3), 329-343.
    """
    n    = corr.shape[0]
    Y    = corr.copy()
    dS   = np.zeros_like(Y)   # correção de Dykstra acumulada

    for _ in range(max_iter):
        R = Y - dS

        # ── Projeção S: positiva semi-definida ──
        vals, vecs = np.linalg.eigh(R)
        vals       = np.maximum(vals, 0)
        X          = vecs @ np.diag(vals) @ vecs.T
        dS         = X - R

        # ── Projeção U: diagonal unitária ──
        Y_prev = Y.copy()
        Y      = X.copy()
        np.fill_diagonal(Y, 1.0)

        # ── Convergência ──
        diff = np.max(np.abs(Y - Y_prev))
        if diff < tol:
            break

    # Garante simetria exata e diagonal 1 após convergência
    Y = (Y + Y.T) / 2
    np.fill_diagonal(Y, 1.0)
    return Y

def _cholesky_seguro(corr: np.ndarray) -> np.ndarray:
    """
    Cholesky com fallback via Higham (2002) — Nearest Correlation Matrix.

    Preserva diagonal unitária e simetria exata, ao contrário da truncagem
    espectral simples que destrói a estrutura de correlação nas caudas.
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        print("  ⚠ Correlação não-positiva-definida — aplicando Higham (2002)")
        corr_fixed = _higham_nearest_corr(corr)
        return np.linalg.cholesky(corr_fixed)


def _iterar_chunks(n_sim: int, chunk_size: int):
    """Generator de intervalos (start, end) para chunking do Monte Carlo."""
    for start in range(0, n_sim, chunk_size):
        yield start, min(start + chunk_size, n_sim)


def _simular_retornos_diarios(
    params:           ParametrosCalibrados,
    chol:             np.ndarray,
    n_sim:            int,
    diasInvestimento: int,
    z_fixo:           np.ndarray | None,
) -> np.ndarray:
    A = len(params.mus)
    z = z_fixo if z_fixo is not None else np.random.standard_normal((n_sim, diasInvestimento, A))

    # Grades montadas uma vez aqui — não dentro do njit
    x_grid_cop, u_grid_cop = _t_grid(params.nu_copula)
    x_grids_mar = np.empty((A, len(x_grid_cop)))
    u_grids_mar = np.empty((A, len(x_grid_cop)))
    for a in range(A):
        x_grids_mar[a], u_grids_mar[a] = _t_grid(float(params.nus[a]))

    epsilon = _gerar_inovacoes_njit(
        z, chol, params.nu_copula,
        x_grid_cop, u_grid_cop,
        x_grids_mar, u_grids_mar,
    )

    return epsilon * params.sigmas[np.newaxis, np.newaxis, :] + params.mus[np.newaxis, np.newaxis, :]

def _acumular_retornos_chunk(
    r_diario:            np.ndarray,
    proporcaoAcao:       np.ndarray,
    diasInvestimento:    int,
    diasRebalanceamento: int | None,
    n_chunk:             int,
) -> np.ndarray:
    """
    Computa retorno cumulativo do portfólio para um chunk de cenários.

    Sem rebalanceamento: produto direto dos retornos ponderados.
    Com rebalanceamento: evolui pesos dia a dia, resetando a cada N dias.
    """
    if diasRebalanceamento is None:
        return np.prod(1 + r_diario @ proporcaoAcao, axis=1) - 1

    pesos = np.tile(proporcaoAcao, (n_chunk, 1)).astype(float)
    valor = np.ones(n_chunk)
    for d in range(diasInvestimento):
        r_d    = r_diario[:, d, :]
        valor *= 1 + (pesos * r_d).sum(axis=1)
        pesos  = pesos * (1 + r_d)
        pesos /= pesos.sum(axis=1, keepdims=True)
        if (d + 1) % diasRebalanceamento == 0:
            pesos[:] = proporcaoAcao
    return valor - 1


def monteCarlo(
    params:              ParametrosCalibrados,
    proporcaoAcao:       np.ndarray,
    diasInvestimento:    int,
    numSimulacoes:       int = 1_000_000,
    diasRebalanceamento: int | None = None,
    z_fixo:              np.ndarray | None = None,
    chunk_size:          int = 50_000,
) -> np.ndarray:
    """
    Simula retornos cumulativos do portfólio via Monte Carlo em chunks.

    Chunks evitam pressão de memória para numSimulacoes grandes.
    z_fixo garante superfície determinística na otimização de pesos.
    """
    chol       = _cholesky_seguro(params.corr)
    resultados = []

    for start, end in _iterar_chunks(numSimulacoes, chunk_size):
        n_chunk = end - start
        z_chunk = z_fixo[start:end] if z_fixo is not None else None

        r_diario = _simular_retornos_diarios(params, chol, n_chunk, diasInvestimento, z_chunk)
        resultados.append(
            _acumular_retornos_chunk(r_diario, proporcaoAcao, diasInvestimento, diasRebalanceamento, n_chunk)
        )

    return np.concatenate(resultados)