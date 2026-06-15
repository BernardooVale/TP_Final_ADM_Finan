"""Otimização de carteira: pesos da RV que minimizam o CVaR (Nelder-Mead)."""
from __future__ import annotations

import numpy as np
import streamlit as st

import interface
from app import estado, componentes as ui
from app.formatacao import pct

FUNC = "alocacaoOtimizada"


def render() -> None:
    ui.cabecalho(
        "Otimização da carteira (mín. CVaR)",
        "Encontra os pesos da Renda Variável que minimizam a perda esperada "
        "condicional (CVaR), isolando a parcela de bolsa.",
    )
    cart = estado.carteira()

    c1, c2 = st.columns(2)
    confianca = c1.slider("Confiança do CVaR", 0.80, 0.99, 0.95, 0.01)
    poupa = c2.toggle("Modo rápido (poupa tempo)", value=True,
                      help="Reduz o rigor estatístico para resposta rápida na UI.")
    n_sim = c1.select_slider("Simulações", [50_000, 100_000, 250_000, 500_000],
                             value=100_000)

    ui.status_cache(chave=None, do_grupo=False)

    if ui.botao_calcular(FUNC, em_cache=False):
        pesos = ui.executar_com_spinner(
            False,
            lambda: interface.alocacaoOtimizada(
                params=cart.params,
                diasInvestimento=cart.diasInvestimento,
                confianca=confianca,
                numSimulacoes=n_sim,
                diasRebalanceamento=st.session_state["diasRebalanceamento"],
                poupaTempo=poupa,
            ),
        )
        ui.guardar_resultado(FUNC, np.asarray(pesos))

    res = ui.resultado_anterior(FUNC)
    if res is not None:
        _exibir(res, cart)


def _exibir(pesos: np.ndarray, cart) -> None:
    st.divider()
    st.markdown("**Pesos ótimos da Renda Variável**")
    st.bar_chart({t: float(w) for t, w in zip(cart.tickers, pesos)})
    st.dataframe(
        [{"Ticker": t, "Peso ótimo": pct(float(w)),
          "Peso atual": pct(float(pa))}
         for t, w, pa in zip(cart.tickers, pesos, cart.pesos_rv)],
        hide_index=True, width='stretch',
    )

    if st.button("Aplicar pesos ótimos à carteira", key="aplicar_pesos"):
        cart.pesos_rv = np.asarray(pesos, dtype=float)
        st.toast("Pesos aplicados à carteira.")
        st.rerun()
