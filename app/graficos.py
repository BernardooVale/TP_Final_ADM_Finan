"""Gráficos matplotlib reutilizáveis pela interface."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend headless — Streamlit renderiza a figura
import matplotlib.pyplot as plt
import numpy as np

_COR = "#2563eb"
_COR2 = "#16a34a"
_COR3 = "#dc2626"


def histograma_patrimonio(
    distribuicao: np.ndarray,
    capitalTotal: float,
    meta: float | None = None,
):
    """Histograma da distribuição do patrimônio final com linhas de referência."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    dados = np.asarray(distribuicao)
    # Recorta cauda extrema (P99.5) só para a escala do gráfico não estourar.
    lim = np.percentile(dados, 99.5)
    ax.hist(dados[dados <= lim], bins=80, color=_COR, alpha=0.8, edgecolor="none")

    ax.axvline(capitalTotal, color=_COR3, linestyle="--", linewidth=1.5,
               label="Capital inicial")
    ax.axvline(np.median(dados), color="#111827", linestyle="-", linewidth=1.5,
               label="Mediana")
    if meta is not None:
        ax.axvline(meta, color=_COR2, linestyle="--", linewidth=1.5, label="Meta")

    ax.set_xlabel("Patrimônio final (R$)")
    ax.set_ylabel("Cenários")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fronteira_cvar(pontos: list):
    """Dispersão risco (CVaR) × retorno médio da fronteira eficiente."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [p.cvar * 100 for p in pontos]
    y = [p.retorno_medio * 100 for p in pontos]
    ax.plot(x, y, "-o", color=_COR, markersize=6)
    for p in pontos:
        ax.annotate(f"{p.fracao_rv*100:.0f}% RV",
                    (p.cvar * 100, p.retorno_medio * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("Risco — CVaR (%)")
    ax.set_ylabel("Retorno médio (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def pareto_piso_meta(fronteira: list, capitalTotal: float):
    """Trade-off P(piso) × P(meta) ao longo da fronteira de Pareto."""
    fig, ax = plt.subplots(figsize=(7, 4))
    rv_pct = [p.alocRV / capitalTotal * 100 for p in fronteira]
    ax.plot(rv_pct, [p.prob_piso * 100 for p in fronteira],
            "-o", color=_COR2, label="P(≥ piso)", markersize=5)
    ax.plot(rv_pct, [p.prob_meta * 100 for p in fronteira],
            "-o", color=_COR, label="P(≥ meta)", markersize=5)
    ax.set_xlabel("Alocação em RV (%)")
    ax.set_ylabel("Probabilidade (%)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def barras_alocacao(alocRF: float, alocRV: float):
    """Barra horizontal empilhada RF × RV."""
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.barh([0], [alocRF], color=_COR2, label="Renda Fixa")
    ax.barh([0], [alocRV], left=[alocRF], color=_COR, label="Renda Variável")
    ax.set_yticks([])
    ax.set_xlabel("R$")
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(top=0.75, bottom=0.35)
    return fig
