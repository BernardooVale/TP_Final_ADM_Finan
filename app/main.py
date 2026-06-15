"""
Interface Streamlit da Engine Quantitativa de Portfólio.

Execute a partir da raiz do repositório:

    streamlit run app/main.py

Fluxo: o usuário monta a carteira na própria página (sem barra lateral) e
calibra uma vez (passo caro: download + MLE). Em seguida navega pelas 8
funcionalidades. Alocação/comparação/meta/duplo objetivo compartilham o mesmo
Monte Carlo via cache (ver app/estado.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garante a raiz do repo no path para importar engine/modelos/data/interface.
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np
import streamlit as st

from modelos.defs import FrequenciaRentabilidadeRendaFix
from app import estado
from app.tema import aplicar_tema, badge, callout, MUTED
from app.formatacao import reais
from app.paginas import (
    alocacao, comparacao, meta, duplo_objetivo,
    fronteira, alocacao_otimizada, desacumulacao, tempo_meta,
)

st.set_page_config(page_title="Engine de Portfólio", layout="wide")

PAGINAS = {
    "Alocação":          alocacao.render,
    "Comparador":        comparacao.render,
    "Meta":              meta.render,
    "Duplo objetivo":    duplo_objetivo.render,
    "Fronteira":         fronteira.render,
    "Otimizar carteira": alocacao_otimizada.render,
    "Desacumulação":     desacumulacao.render,
    "Tempo para meta":   tempo_meta.render,
}

# Ativos populares da B3 + ETFs (multiselect aceita também tickers digitados).
POPULARES = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA",
    "B3SA3.SA", "WEGE3.SA", "ITSA4.SA", "PETR3.SA", "BPAC11.SA", "RENT3.SA",
    "SUZB3.SA", "PRIO3.SA", "ELET3.SA", "GGBR4.SA", "JBSS3.SA", "RADL3.SA",
    "MGLU3.SA", "VBBR3.SA", "BOVA11.SA", "IVVB11.SA",
]

PRAZOS = {"6 meses": 126, "1 ano": 252, "2 anos": 504, "3 anos": 756, "5 anos": 1260}

# Frequência de rebalanceamento da Renda Variável (em dias úteis; None = não rebalanceia).
REBALANCEAMENTO = {
    "Nunca (buy and hold)": None,
    "Mensal":     21,
    "Trimestral": 63,
    "Semestral":  126,
    "Anual":      252,
}

FREQ_RF_LABEL = {
    FrequenciaRentabilidadeRendaFix.ANUAL: "ao ano",
    FrequenciaRentabilidadeRendaFix.TRIMESTRAL: "ao trimestre",
    FrequenciaRentabilidadeRendaFix.MENSAL: "ao mês",
    FrequenciaRentabilidadeRendaFix.DIARIO: "ao dia",
}


# ──────────────────────────── setup (onboarding / edição) ────────────────────────────

def _editor_pesos(tickers: list[str]) -> list[float]:
    """Grade de pesos por ativo (number_input), com botão de pesos iguais."""
    n = len(tickers)
    iguais = round(100 / n, 1) if n else 0.0
    for t in tickers:
        st.session_state.setdefault(f"w_{t}", iguais)

    lin1, lin2 = st.columns([3, 1])
    lin1.markdown("**Distribuição da Renda Variável**")
    if lin2.button("Pesos iguais", width="stretch"):
        for t in tickers:
            st.session_state[f"w_{t}"] = iguais
        st.rerun()

    pesos: list[float] = []
    cols = st.columns(min(n, 4)) if n else []
    for i, t in enumerate(tickers):
        with cols[i % 4]:
            pesos.append(st.number_input(t, min_value=0.0, step=5.0, key=f"w_{t}"))
    soma = sum(pesos)
    st.caption(
        f"Soma dos pesos: {soma:.0f}% — será normalizada para 100% automaticamente."
        if soma else "Defina ao menos um peso positivo."
    )
    return pesos


def render_setup() -> None:
    primeira_vez = not estado.carteira_pronta()
    st.title("Engine Quantitativa de Portfólio")
    st.caption("Simulação estocástica de carteiras — Monte Carlo, t-Student e cópula-t.")
    st.write("")

    cart = estado.carteira()
    tickers_default = cart.tickers if cart else ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]

    with st.container(border=True):
        st.markdown("##### Ativos e pesos")
        tickers = st.multiselect(
            "Ativos de Renda Variável",
            options=sorted(set(POPULARES) | set(tickers_default)),
            default=tickers_default,
            accept_new_options=True,
            help="Selecione da lista ou digite um ticker do yfinance (ex.: TAEE11.SA).",
        )
        st.write("")
        pesos = _editor_pesos(tickers) if tickers else []

    with st.container(border=True):
        st.markdown("##### Parâmetros")
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input("Capital total (R$)", 1000.0, step=1000.0,
                                  value=float(cart.capitalTotal) if cart else 100_000.0)
        prazo_lbl = c2.selectbox("Prazo do investimento", list(PRAZOS),
                                 index=_indice_prazo(cart))
        qualidade = c3.selectbox("Qualidade da simulação", list(estado.QUALIDADES), index=1,
                                 help="Mais cenários = mais preciso e mais lento.")

        c4, c5, c6 = st.columns(3)
        rf_pct = c4.number_input("Taxa da Renda Fixa (%)", 0.0, 100.0,
                                 value=float(cart.rentabilidadeRF * 100) if cart else 14.5, step=0.1)
        freq_rf = c5.selectbox("Frequência da taxa RF", list(FrequenciaRentabilidadeRendaFix),
                               index=list(FrequenciaRentabilidadeRendaFix).index(
                                   cart.freqRF if cart else FrequenciaRentabilidadeRendaFix.ANUAL),
                               format_func=lambda f: f"{f.value} ({FREQ_RF_LABEL[f]})")
        periodo = c6.selectbox("Histórico para calibração", ["1y", "2y", "3y", "5y", "10y"],
                               index=2)

        with st.expander("Avançado"):
            rebal_lbl = st.selectbox(
                "Rebalanceamento da carteira",
                list(REBALANCEAMENTO),
                index=_indice_rebal(),
                help="Com que frequência os pesos da Renda Variável voltam ao alvo. "
                     "'Nunca' deixa os pesos correrem com o mercado (buy and hold); "
                     "as demais opções devolvem a carteira aos pesos definidos a cada período.",
            )

    st.session_state["numSimulacoes"] = estado.QUALIDADES[qualidade]
    st.session_state["diasRebalanceamento"] = REBALANCEAMENTO[rebal_lbl]

    b1, b2 = st.columns([1, 4])
    calibrar = b1.button("Calibrar carteira", type="primary", width="stretch",
                         disabled=not tickers)
    if not primeira_vez:
        if b2.button("Cancelar", width="content"):
            st.session_state["editando"] = False
            st.rerun()

    if calibrar:
        if sum(pesos) <= 0:
            callout("Defina ao menos um peso positivo para a Renda Variável.", "warning")
            return
        try:
            with st.spinner("Baixando histórico e calibrando as distribuições..."):
                estado.calibrar(
                    tickers=tickers,
                    pesos_rv=pesos,
                    periodo=periodo,
                    capitalTotal=capital,
                    rentabilidadeRF=rf_pct / 100.0,
                    freqRF=freq_rf,
                    diasInvestimento=PRAZOS[prazo_lbl],
                )
            st.session_state["editando"] = False
            st.toast("Carteira calibrada.")
            st.rerun()
        except Exception as e:  # noqa: BLE001 — feedback direto ao usuário
            callout(f"Falha na calibração: {e}", "destructive")


def _indice_prazo(cart) -> int:
    if cart is None:
        return 1
    dias = cart.diasInvestimento
    vals = list(PRAZOS.values())
    return vals.index(dias) if dias in vals else 1


def _indice_rebal() -> int:
    atual = st.session_state.get("diasRebalanceamento")
    vals = list(REBALANCEAMENTO.values())
    return vals.index(atual) if atual in vals else 0


# ──────────────────────────── header da carteira (calibrada) ────────────────────────────

def header_carteira() -> None:
    cart = estado.carteira()
    chips = " ".join(
        badge(f"{t.replace('.SA','')} {p*100:.0f}%", "secondary")
        for t, p in zip(cart.tickers, cart.pesos_rv)
    )
    prazo_lbl = next((k for k, v in PRAZOS.items() if v == cart.diasInvestimento),
                     f"{cart.diasInvestimento}d")
    meta_info = (
        f'<span style="color:{MUTED};font-size:0.85rem;">'
        f'{reais(cart.capitalTotal)} &nbsp;·&nbsp; {prazo_lbl} &nbsp;·&nbsp; '
        f'RF {cart.rentabilidadeRF*100:.1f}%</span>'
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center;">'
        f'{chips} &nbsp; {meta_info}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    qual_atual = _label_qualidade(st.session_state["numSimulacoes"])
    qualidade = c1.selectbox("Qualidade da simulação", list(estado.QUALIDADES),
                             index=list(estado.QUALIDADES).index(qual_atual),
                             label_visibility="collapsed")
    st.session_state["numSimulacoes"] = estado.QUALIDADES[qualidade]
    if c3.button("Editar carteira", width="stretch"):
        st.session_state["editando"] = True
        st.rerun()


def _label_qualidade(n_sim: int) -> str:
    for k, v in estado.QUALIDADES.items():
        if v == n_sim:
            return k
    return list(estado.QUALIDADES)[1]


# ──────────────────────────── main ────────────────────────────

def main() -> None:
    estado.init_estado()
    aplicar_tema()
    st.session_state.setdefault("editando", False)

    if not estado.carteira_pronta() or st.session_state["editando"]:
        render_setup()
        return

    header_carteira()
    st.divider()
    escolha = st.radio("Funcionalidade", list(PAGINAS), horizontal=True,
                       label_visibility="collapsed")
    st.write("")
    PAGINAS[escolha]()


main()
