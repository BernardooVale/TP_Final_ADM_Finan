"""Gráficos matplotlib reutilizáveis pela interface."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend headless — Streamlit renderiza a figura
import matplotlib.pyplot as plt
import numpy as np

# Paleta zinc dark (shadcn-like): claros sobre fundo transparente + um acento.
_TINTA   = "#e4e4e7"  # zinc-200 — série/linha principal (clara)
_GRAFITE = "#52525b"  # zinc-600 — preenchimentos
_CINZA   = "#71717a"  # zinc-500 — referências/secundário
_CLARO   = "#3f3f46"  # zinc-700 — RF na barra empilhada
_ACENTO  = "#3b82f6"  # azul que destaca no escuro

# Compat com nomes antigos
_COR, _COR2, _COR3 = _TINTA, _ACENTO, _CINZA

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "text.color":       "#e4e4e7",
    "axes.edgecolor":   "#3f3f46",
    "axes.labelcolor":  "#a1a1aa",
    "xtick.color":      "#a1a1aa",
    "ytick.color":      "#a1a1aa",
    "axes.grid":        True,
    "grid.color":       "#27272a",
    "grid.linewidth":   0.8,
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "savefig.facecolor": "none",
})


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
    ax.hist(dados[dados <= lim], bins=80, color=_GRAFITE, alpha=0.85, edgecolor="none")

    ax.axvline(capitalTotal, color=_CINZA, linestyle="--", linewidth=1.4,
               label="Capital inicial")
    ax.axvline(np.median(dados), color=_TINTA, linestyle="-", linewidth=1.6,
               label="Mediana")
    if meta is not None:
        ax.axvline(meta, color=_ACENTO, linestyle="--", linewidth=1.6, label="Meta")

    ax.set_xlabel("Patrimônio final (R$)")
    ax.set_ylabel("Cenários")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fronteira_cvar(pontos: list):
    """Dispersão risco (CVaR) × retorno médio da fronteira eficiente."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [p.cvar * 100 for p in pontos]
    y = [p.retorno_medio * 100 for p in pontos]
    ax.plot(x, y, "-o", color=_TINTA, markersize=6, markerfacecolor="white",
            markeredgecolor=_TINTA, markeredgewidth=1.4)
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
            "-o", color=_TINTA, label="P(≥ piso)", markersize=5)
    ax.plot(rv_pct, [p.prob_meta * 100 for p in fronteira],
            "-o", color=_ACENTO, label="P(≥ meta)", markersize=5)
    ax.set_xlabel("Alocação em RV (%)")
    ax.set_ylabel("Probabilidade (%)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def barras_alocacao(alocRF: float, alocRV: float):
    """Barra horizontal empilhada RF × RV."""
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.barh([0], [alocRF], color=_CLARO, label="Renda Fixa")
    ax.barh([0], [alocRV], left=[alocRF], color=_TINTA, label="Renda Variável")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("R$")
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(top=0.75, bottom=0.35)
    return fig
