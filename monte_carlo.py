import numpy as np
from numba import njit, prange
from scipy import stats
from math import sqrt

from defs import ParametrosCalibrados

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
    3. Uniformes via CDF_t(nu_copula)
    4. PPF marginal t-Student por ativo → epsilon (S, D, A)

    Separado do GARCH para reutilização em bootstrap e fronteira eficiente.
    """
    S, D, A = z.shape

    chi2_shared = np.random.chisquare(nu_copula, size=(S, D, 1)) / nu_copula
    t_copula    = z / np.sqrt(chi2_shared)
    y           = t_copula @ chol.T

    y_flat = y.reshape(-1, A)
    u_flat = np.empty_like(y_flat)
    for i in range(A):
        u_flat[:, i] = stats.t.cdf(y_flat[:, i], df=nu_copula)

    x_flat = np.empty_like(u_flat)
    for i in range(A):
        x_flat[:, i] = stats.t.ppf(u_flat[:, i], df=nus[i], loc=0.0, scale=1.0)

    return x_flat.reshape(S, D, A)

@njit(parallel=True, cache=True)
def _garch_scan(
    epsilon: np.ndarray,
    omegas:  np.ndarray,
    alphas:  np.ndarray,
    betas:   np.ndarray,
    mus:     np.ndarray,
    sigmas:  np.ndarray,
) -> np.ndarray:
    """
    Evolução GARCH(1,1) vetorizada via Numba.

    Loop externo (cenários) paralelizado com prange — independentes entre si.
    Loop interno (dias) sequencial — sigma2[t] depende de sigma2[t-1].

    sigma2_0 = omega / (1 - alpha - beta)  [variância incondicional]
    Compilado em C na primeira chamada (cache=True evita recompilação).
    """
    S, D, A  = epsilon.shape
    retornos = np.empty((S, D, A))

    for s in prange(S):
        sigma2 = np.empty(A)
        for a in range(A):
            denom     = 1.0 - alphas[a] - betas[a]
            sigma2[a] = omegas[a] / denom if denom > 1e-8 else sigmas[a] ** 2

        for d in range(D):
            for a in range(A):
                sigma_t           = sqrt(sigma2[a])
                retornos[s, d, a] = mus[a] + sigma_t * epsilon[s, d, a]
                choque            = (sigma_t * epsilon[s, d, a]) ** 2
                sigma2[a]         = omegas[a] + alphas[a] * choque + betas[a] * sigma2[a]

    return retornos

@njit(parallel=True, cache=True)
def _garch_scan_tempo_meta(
    epsilon:             np.ndarray,   # (S, D, A)
    omegas:              np.ndarray,
    alphas:              np.ndarray,
    betas:               np.ndarray,
    mus:                 np.ndarray,
    sigmas:              np.ndarray,
    pesos:               np.ndarray,   # (A,) pesos RV normalizados
    fracao_rf:           float,        # fração do capital total em RF
    crescimento_rf_dia:  float,        # fator diário RF: (1 + taxa_diaria)
    aporte_dia:          float,        # R$ adicionado a cada dia útil (0 = sem aporte)
    intervalo_aporte:    int,          # dias úteis entre aportes (0 = sem aporte)
    capital_inicial:     float,
    meta:                float,
) -> np.ndarray:
    """
    GARCH(1,1) + acúmulo de patrimônio dia a dia; retorna dia de cruzamento da meta.

    Para cada cenário s:
      - Mantém patrimônio_rf e patrimônio_rv separados
      - A cada dia: aplica retorno RF no componente RF, retorno RV ponderado no RV
      - Se intervalo_aporte > 0: adiciona aporte_dia ao componente RF nos dias certos
      - Quando patrimônio_rf + patrimônio_rv >= meta: registra dia e para

    Retorna array (S,) de float:
      - dia de cruzamento (1-indexed) se atingiu
      - -1.0 se não atingiu dentro do horizonte D
    """
    S, D, A = epsilon.shape
    resultado = np.full(S, -1.0)

    for s in prange(S):
        # Inicializa volatilidade GARCH
        sigma2 = np.empty(A)
        for a in range(A):
            denom     = 1.0 - alphas[a] - betas[a]
            sigma2[a] = omegas[a] / denom if denom > 1e-8 else sigmas[a] ** 2

        pat_rf = capital_inicial * fracao_rf
        pat_rv = capital_inicial * (1.0 - fracao_rf)

        for d in range(D):
            # Retorno RV ponderado do dia
            ret_rv_dia = 0.0
            for a in range(A):
                sigma_t    = sqrt(sigma2[a])
                r_at       = mus[a] + sigma_t * epsilon[s, d, a]
                ret_rv_dia += pesos[a] * r_at
                choque     = (sigma_t * epsilon[s, d, a]) ** 2
                sigma2[a]  = omegas[a] + alphas[a] * choque + betas[a] * sigma2[a]

            # Evolução do patrimônio
            pat_rf *= crescimento_rf_dia
            pat_rv *= (1.0 + ret_rv_dia)

            # Aporte periódico (adicionado à parcela RF)
            if intervalo_aporte > 0 and (d + 1) % intervalo_aporte == 0:
                pat_rf += aporte_dia

            # Verifica cruzamento da meta
            if pat_rf + pat_rv >= meta:
                resultado[s] = float(d + 1)   # 1-indexed
                break

    return resultado

@njit(parallel=True, cache=True)
def _garch_scan_desacumulacao(
    epsilon:            np.ndarray,   # (S, D, A)
    omegas:             np.ndarray,
    alphas:             np.ndarray,
    betas:              np.ndarray,
    mus:                np.ndarray,
    sigmas:             np.ndarray,
    pesos:              np.ndarray,   # (A,) pesos RV normalizados
    fracao_rf:          float,
    crescimento_rf_dia: float,        # (1 + taxa_diaria_rf)
    saque_por_dia:      float,        # R$ retirado a cada `intervalo_saque` dias
    intervalo_saque:    int,          # dias úteis entre saques
    capital_inicial:    float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GARCH(1,1) com saques periódicos; detecta ruína cenário a cenário.

    Evolução por dia d:
        pat_rf *= crescimento_rf_dia
        pat_rv *= (1 + retorno_rv_ponderado_d)
        se (d+1) % intervalo_saque == 0: subtrai saque_por_dia de pat_rf
            se pat_rf < 0: transfere déficit de pat_rv (venda de RV para cobrir saque)
        se pat_rf + pat_rv <= 0: registra ruína e para

    Retorna
    -------
    dia_ruina       : (S,) float — dia da ruína (1-indexed) ou -1.0 se sobreviveu
    patrimonio_final: (S,) float — patrimônio no último dia (0.0 se arruinou)
    """
    S, D, A = epsilon.shape
    dia_ruina        = np.full(S, -1.0)
    patrimonio_final = np.zeros(S)

    for s in prange(S):
        sigma2 = np.empty(A)
        for a in range(A):
            denom     = 1.0 - alphas[a] - betas[a]
            sigma2[a] = omegas[a] / denom if denom > 1e-8 else sigmas[a] ** 2

        pat_rf = capital_inicial * fracao_rf
        pat_rv = capital_inicial * (1.0 - fracao_rf)
        arruinou = False

        for d in range(D):
            # Retorno RV ponderado
            ret_rv_dia = 0.0
            for a in range(A):
                sigma_t    = sqrt(sigma2[a])
                r_at       = mus[a] + sigma_t * epsilon[s, d, a]
                ret_rv_dia += pesos[a] * r_at
                choque     = (sigma_t * epsilon[s, d, a]) ** 2
                sigma2[a]  = omegas[a] + alphas[a] * choque + betas[a] * sigma2[a]

            pat_rf *= crescimento_rf_dia
            pat_rv *= (1.0 + ret_rv_dia)

            # Saque periódico: debita da parcela RF; se insuficiente, liquida RV
            if intervalo_saque > 0 and (d + 1) % intervalo_saque == 0:
                pat_rf -= saque_por_dia
                if pat_rf < 0.0:
                    pat_rv += pat_rf   # transfere déficit (pat_rf negativo)
                    pat_rf  = 0.0

            if pat_rf + pat_rv <= 0.0:
                dia_ruina[s] = float(d + 1)
                arruinou     = True
                break

        if not arruinou:
            patrimonio_final[s] = pat_rf + pat_rv

    return dia_ruina, patrimonio_final

def _cholesky_seguro(corr: np.ndarray) -> np.ndarray:
    """
    Cholesky com fallback via correção de Higham.

    Se a matriz não for positiva-definida (outliers ou multicolinearidade),
    clipa autovalores negativos e reconstrói. Renormaliza diagonal para 1.
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        print("  ⚠ Correlação não-positiva-definida — aplicando correção de Higham")
        vals, vecs = np.linalg.eigh(corr)
        vals       = np.maximum(vals, 1e-8)
        corr_fixed = vecs @ np.diag(vals) @ vecs.T
        d          = np.sqrt(np.diag(corr_fixed))
        corr_fixed = corr_fixed / np.outer(d, d)
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
    """
    Gera retornos diários correlacionados com cópula-t e volatilidade GARCH(1,1).

    Delega geração de inovações a _gerar_inovacoes e evolução GARCH a _garch_scan.
    Sem GARCH (omegas None): sigma fixo, caminho numpy puro (retrocompatível).
    """
    A = len(params.mus)
    z = z_fixo if z_fixo is not None else np.random.standard_normal((n_sim, diasInvestimento, A))

    epsilon = _gerar_inovacoes(z, chol, params.nu_copula, params.nus)

    if params.omegas is None or params.alphas is None or params.betas is None:
        return epsilon * params.sigmas[np.newaxis, np.newaxis, :] + params.mus[np.newaxis, np.newaxis, :]

    return _garch_scan(
        epsilon,
        params.omegas.astype(np.float64),
        params.alphas.astype(np.float64),
        params.betas.astype(np.float64),
        params.mus.astype(np.float64),
        params.sigmas.astype(np.float64),
    )


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