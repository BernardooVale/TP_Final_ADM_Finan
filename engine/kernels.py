import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _kernel_tempo_meta(
    mus:                np.ndarray,
    sigmas:             np.ndarray,
    pesos:              np.ndarray,
    chol:               np.ndarray,  # (A, A)
    nu_copula:          float,
    x_grid_cop:         np.ndarray,  # grade CDF cópula
    u_grid_cop:         np.ndarray,
    x_grids_mar:        np.ndarray,  # (A, N_GRID) grades PPF marginais
    u_grids_mar:        np.ndarray,  # (A, N_GRID)
    fracao_rf:          float,
    crescimento_rf_dia: float,
    aporte_dia:         float,
    intervalo_aporte:   int,
    capital_inicial:    float,
    meta:               float,
    n_sim:              int,
    n_dias:             int,
) -> np.ndarray:
    A         = len(mus)
    resultado = np.full(n_sim, -1.0)

    for s in prange(n_sim):
        pat_rf = capital_inicial * fracao_rf
        pat_rv = capital_inicial * (1.0 - fracao_rf)

        for d in range(n_dias):
            # ── Gera inovações para um único dia ──
            # 1. Normal padrão
            z = np.empty(A)
            for a in range(A):
                z[a] = np.random.standard_normal()

            # 2. Cópula-t: z / sqrt(chi2 / nu)
            chi2 = 0.0
            for _ in range(int(nu_copula)):
                g = np.random.standard_normal()
                chi2 += g * g
            chi2 /= nu_copula
            t_cop = np.empty(A)
            for a in range(A):
                t_cop[a] = z[a] / np.sqrt(chi2)

            # 3. Cholesky
            y = np.zeros(A)
            for a in range(A):
                for b in range(A):
                    y[a] += chol[a, b] * t_cop[b]

            # 4. CDF cópula → uniformes
            u = np.empty(A)
            for a in range(A):
                u[a] = np.interp(y[a], x_grid_cop, u_grid_cop)

            # 5. PPF marginal por ativo
            eps = np.empty(A)
            for a in range(A):
                eps[a] = np.interp(u[a], u_grids_mar[a], x_grids_mar[a])

            # ── Evolui patrimônio ──
            ret_rv_dia = 0.0
            for a in range(A):
                ret_rv_dia += pesos[a] * (mus[a] + sigmas[a] * eps[a])

            pat_rf *= crescimento_rf_dia
            pat_rv *= (1.0 + ret_rv_dia)

            if intervalo_aporte > 0 and (d + 1) % intervalo_aporte == 0:
                pat_rf += aporte_dia

            if pat_rf + pat_rv >= meta:
                resultado[s] = float(d + 1)
                break

    return resultado


@njit(parallel=True, cache=True)
def _kernel_desacumulacao(
    epsilon:            np.ndarray,
    mus:                np.ndarray,
    sigmas:             np.ndarray,
    pesos:              np.ndarray,
    fracao_rf:          float,
    crescimento_rf_dia: float,
    saque_por_dia:      float,
    intervalo_saque:    int,
    capital_inicial:    float,
) -> tuple[np.ndarray, np.ndarray]:
    S, D, A          = epsilon.shape
    dia_ruina        = np.full(S, -1.0)
    patrimonio_final = np.zeros(S)

    for s in prange(S):
        pat_rf   = capital_inicial * fracao_rf
        pat_rv   = capital_inicial * (1.0 - fracao_rf)
        arruinou = False

        for d in range(D):
            ret_rv_dia = 0.0
            for a in range(A):
                ret_rv_dia += pesos[a] * (mus[a] + sigmas[a] * epsilon[s, d, a])

            pat_rf *= crescimento_rf_dia
            pat_rv *= (1.0 + ret_rv_dia)

            if intervalo_saque > 0 and (d + 1) % intervalo_saque == 0:
                pat_rf -= saque_por_dia
                if pat_rf < 0.0:
                    pat_rv += pat_rf
                    pat_rf  = 0.0

            if pat_rf + pat_rv <= 0.0:
                dia_ruina[s] = float(d + 1)
                arruinou     = True
                break

        if not arruinou:
            patrimonio_final[s] = pat_rf + pat_rv

    return dia_ruina, patrimonio_final

@njit(parallel=True, cache=True)
def _gerar_inovacoes_njit(
    z:           np.ndarray,   # (S, D, A) normal padrão
    chol:        np.ndarray,   # (A, A)
    nu_copula:   float,
    x_grid_cop:  np.ndarray,   # (N_GRID,) grade CDF cópula
    u_grid_cop:  np.ndarray,   # (N_GRID,)
    x_grids_mar: np.ndarray,   # (A, N_GRID) grades PPF marginais
    u_grids_mar: np.ndarray,   # (A, N_GRID)
) -> np.ndarray:
    """
    Versão Numba de _gerar_inovacoes — paraleliza sobre cenários via prange.

    Fluxo por cenário s e dia d:
    1. t_copula = z[s,d] / sqrt(chi2 / nu_copula)
    2. y = chol @ t_copula  (correlaciona)
    3. u = CDF_t(y, nu_copula) via interp nas grades
    4. x = PPF_t(u, nu_marginal[a]) via interp nas grades
    """
    S, D, A  = z.shape
    epsilon  = np.empty((S, D, A))

    for s in prange(S):
        for d in range(D):
            # ── 1. Cópula-t: divide por sqrt(chi2/nu) ──
            chi2 = np.random.gamma(nu_copula / 2.0, 2.0)
            chi2 /= nu_copula

            t_cop = np.empty(A)
            for a in range(A):
                t_cop[a] = z[s, d, a] / np.sqrt(chi2)

            # ── 2. Cholesky ──
            y = np.zeros(A)
            for a in range(A):
                for b in range(A):
                    y[a] += chol[a, b] * t_cop[b]

            # ── 3. CDF cópula → uniformes ──
            u = np.empty(A)
            for a in range(A):
                u[a] = np.interp(y[a], x_grid_cop, u_grid_cop)

            # ── 4. PPF marginal por ativo ──
            for a in range(A):
                epsilon[s, d, a] = np.interp(u[a], u_grids_mar[a], x_grids_mar[a])

    return epsilon