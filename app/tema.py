"""
Tema visual dark inspirado no shadcn/ui (paleta zinc).

Centraliza a injeção de CSS e pequenos componentes de marcação (badge,
callout, chip) usados no lugar dos alertas coloridos padrão do Streamlit,
para um visual sóbrio e consistente.
"""
from __future__ import annotations

import streamlit as st

# ── Design tokens (zinc, dark) ──
BG        = "#09090b"   # zinc-950 — fundo
SURFACE   = "#18181b"   # zinc-900 — cards/inputs
SURFACE_2 = "#27272a"   # zinc-800 — hover/ativo
BORDER    = "#27272a"   # zinc-800
BORDER_2  = "#3f3f46"   # zinc-700
FG        = "#fafafa"   # zinc-50 — texto
MUTED     = "#a1a1aa"   # zinc-400 — texto secundário
MUTED_2   = "#71717a"   # zinc-500

# Variantes de badge/callout: (fundo, texto, borda)
_VARIANTS = {
    "default":     (FG,                     "#18181b",  FG),
    "secondary":   (SURFACE_2,              "#e4e4e7",  BORDER_2),
    "outline":     ("transparent",          "#d4d4d8",  BORDER_2),
    "success":     ("rgba(34,197,94,0.12)", "#4ade80",  "rgba(34,197,94,0.30)"),
    "warning":     ("rgba(245,158,11,0.12)","#fbbf24",  "rgba(245,158,11,0.30)"),
    "destructive": ("rgba(239,68,68,0.12)", "#f87171",  "rgba(239,68,68,0.30)"),
}

_CSS = f"""
<style>
:root {{
  --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
               Inter, Helvetica, Arial, sans-serif;
}}
html, body, [class*="css"], .stApp, button, input, textarea, select {{
  font-family: var(--font-sans);
}}

/* Densidade e largura */
.block-container {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1080px; }}

/* Tipografia */
h1 {{ font-weight: 650; letter-spacing: -0.02em; font-size: 1.7rem; }}
h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED}; }}

/* Botões — primário sólido claro, demais como outline */
.stButton > button, .stFormSubmitButton > button {{
  border-radius: 0.5rem;
  border: 1px solid {BORDER_2};
  font-weight: 500;
  transition: background-color .15s ease, border-color .15s ease;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: {FG}; color: {BG}; border-color: {FG};
}}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
  background: #e4e4e7; border-color: #e4e4e7;
}}
.stButton > button[kind="secondary"] {{ background: transparent; color: {FG}; }}
.stButton > button[kind="secondary"]:hover {{ border-color: {MUTED_2}; background: {SURFACE}; }}

/* Métricas como cards discretos */
[data-testid="stMetric"] {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 0.6rem;
  padding: 0.85rem 1rem;
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-weight: 600; letter-spacing: -0.01em; }}

/* Contêineres com borda (st.container(border=True)) viram cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 0.75rem; }}

/* Navegação (radio horizontal) como abas/segmented control */
div[role="radiogroup"] {{ gap: 0.25rem; flex-wrap: wrap; }}
div[role="radiogroup"] > label {{
  border: 1px solid transparent;
  border-radius: 0.5rem;
  padding: 0.3rem 0.75rem;
  margin: 0;
  color: {MUTED};
}}
div[role="radiogroup"] > label:hover {{ background: {SURFACE}; color: {FG}; }}
div[role="radiogroup"] > label:has(input:checked) {{
  background: {SURFACE_2}; color: {FG}; border-color: {BORDER_2};
  font-weight: 500;
}}
div[role="radiogroup"] svg, div[role="radiogroup"] > label > div:first-child {{ display: none; }}

/* Inputs / selects / expanders com cantos suaves */
[data-testid="stExpander"], .stDataFrame {{ border-radius: 0.5rem; }}

/* Tags do multiselect — viram badges escuros (em vez do branco do primaryColor) */
span[data-baseweb="tag"] {{
  background-color: {SURFACE_2} !important;
  border: 1px solid {BORDER_2};
  border-radius: 0.5rem !important;
  color: {FG} !important;
}}
span[data-baseweb="tag"] span {{ color: {FG} !important; }}
span[data-baseweb="tag"] svg {{ color: {MUTED} !important; fill: {MUTED} !important; }}
span[data-baseweb="tag"] [role="button"]:hover {{ background-color: {BORDER_2} !important; }}

/* Divisores mais sutis */
hr {{ border-color: {BORDER}; }}

/* Esconde a barra lateral (setup foi para a página) */
section[data-testid="stSidebar"] {{ display: none; }}
</style>
"""


def aplicar_tema() -> None:
    """Injeta o CSS do tema. Chamar uma vez, logo após set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(label: str, variant: str = "secondary") -> str:
    """HTML de um badge shadcn-like (usar com st.markdown(..., unsafe_allow_html=True))."""
    bg, fg, bd = _VARIANTS.get(variant, _VARIANTS["secondary"])
    return (
        f'<span style="display:inline-block;padding:0.12rem 0.55rem;font-size:0.78rem;'
        f'font-weight:500;line-height:1.5;border-radius:0.5rem;background:{bg};'
        f'color:{fg};border:1px solid {bd};white-space:nowrap;">{label}</span>'
    )


def callout(texto: str, variant: str = "secondary") -> None:
    """Caixa de aviso discreta (substitui st.info/warning/success padrão)."""
    bg, fg, bd = _VARIANTS.get(variant, _VARIANTS["secondary"])
    st.markdown(
        f'<div style="padding:0.7rem 0.95rem;border:1px solid {bd};background:{bg};'
        f'color:{fg};border-radius:0.6rem;font-size:0.9rem;line-height:1.45;">{texto}</div>',
        unsafe_allow_html=True,
    )
