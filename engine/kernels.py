import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _garch_scan(
    epsilon: np.ndarray,
    mus:     np.ndarray,
    sigmas:  np.ndarray,
) -> np.ndarray:
    S, D, A  = epsilon.shape
    retornos = np.empty((S, D, A))

    for s in prange(S):
        for d in range(D):
            for a in range(A):
                retornos[s, d, a] = mus[a] + sigmas[a] * epsilon[s, d, a]

    return retornos


@njit(parallel=True, cache=True)
def _garch_scan_tempo_meta(
    epsilon:            np.ndarray,
    mus:                np.ndarray,
    sigmas:             np.ndarray,
    pesos:              np.ndarray,
    fracao_rf:          float,
    crescimento_rf_dia: float,
    aporte_dia:         float,
    intervalo_aporte:   int,
    capital_inicial:    float,
    meta:               float,
) -> np.ndarray:
    S, D, A   = epsilon.shape
    resultado = np.full(S, -1.0)

    for s in prange(S):
        pat_rf = capital_inicial * fracao_rf
        pat_rv = capital_inicial * (1.0 - fracao_rf)

        for d in range(D):
            ret_rv_dia = 0.0
            for a in range(A):
                ret_rv_dia += pesos[a] * (mus[a] + sigmas[a] * epsilon[s, d, a])

            pat_rf *= crescimento_rf_dia
            pat_rv *= (1.0 + ret_rv_dia)

            if intervalo_aporte > 0 and (d + 1) % intervalo_aporte == 0:
                pat_rf += aporte_dia

            if pat_rf + pat_rv >= meta:
                resultado[s] = float(d + 1)
                break

    return resultado


@njit(parallel=True, cache=True)
def _garch_scan_desacumulacao(
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