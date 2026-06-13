import numpy as np
from numpy import ndarray
from dataclasses import dataclass

@dataclass
class ParametrosCalibrados:
    nus:       np.ndarray
    mus:       np.ndarray
    sigmas:    np.ndarray
    corr:      np.ndarray
    nu_copula: float

    def __post_init__(self) -> None:
        self._validar()

    def _validar(self) -> None:
        erros: list[str] = []

        # ── Graus de liberdade marginais ──
        # nu > 2 exigido para variância finita da t-Student.
        # nu <= 2 produz PPF divergente nas caudas — inovações infinitas no kernel.
        for i, nu in enumerate(self.nus):
            if nu <= 2.0:
                erros.append(
                    f"nus[{i}]={nu:.4f}: t-Student requer nu > 2 para variância finita"
                )

        # ── Grau de liberdade da cópula ──
        if self.nu_copula <= 2.0:
            erros.append(
                f"nu_copula={self.nu_copula:.4f}: cópula-t requer nu > 2 para variância finita"
            )

        # ── Matriz de correlação ──
        A = len(self.mus)
        if self.corr.shape != (A, A):
            erros.append(
                f"corr.shape={self.corr.shape}: esperado ({A}, {A})"
            )
        else:
            # Simetria
            if not np.allclose(self.corr, self.corr.T, atol=1e-8):
                erros.append("corr não é simétrica")

            # Diagonal unitária
            diag = np.diag(self.corr)
            if not np.allclose(diag, 1.0, atol=1e-6):
                erros.append(f"corr diagonal não é unitária: min={diag.min():.6f} max={diag.max():.6f}")

            # Positiva semi-definida — autovalores não-negativos
            eigvals = np.linalg.eigvalsh(self.corr)
            if eigvals.min() < -1e-6:
                erros.append(
                    f"corr não é positiva semi-definida: menor autovalor={eigvals.min():.2e} "
                    "(_cholesky_seguro aplicará correção de Higham)"
                )
                import warnings
                warnings.warn(
                    "Matriz de correlação não é positiva semi-definida — "
                    "correção de Higham será aplicada automaticamente no Cholesky",
                    UserWarning,
                    stacklevel=3,
                )

        if erros:
            raise ValueError(
                "ParametrosCalibrados inválidos:\n" +
                "\n".join(f"  • {e}" for e in erros)
            )

@dataclass
class ParametrosRF:
    crescimento:     float
    retorno_periodo: float
    taxa_diaria:     float
    

@dataclass
class BoundsAtivo:
    """
    Limites de alocação por ativo.

    tickers_rv  : bounds para cada ação (min, max), na mesma ordem de `tickers`
    rf          : bounds para a fração total em RF (min, max); None = sem restrição
    
    Exemplo:
        BoundsAtivo(
            tickers_rv = [(0.1, 0.5), (0.1, 0.4), (0.0, 0.3)],
            rf         = (0.2, 0.8),
        )
    """
    tickers_rv: list[tuple[float, float]]         # [(min, max), ...] para cada ativo RV
    rf:         tuple[float, float] | None = None  # (min, max) fração RF no portfólio total
    
@dataclass
class Simulacao:
    
    tipoSimulacao: int
    resultadoCumulativo: ndarray
