"""Meta de patrimônio: exposição à RV necessária para atingir um alvo."""
from __future__ import annotations

import numpy as np
import streamlit as st

import interface
from app import estado, componentes as ui
from app.formatacao import reais, pct
from app.graficos import histograma_patrimonio
from app.tema import badge, callout

FUNC = "meta"


def render() -> None:
    ui.cabecalho(
        "Meta de patrimônio",
        "Busca binária pela divisão RF/RV que atinge um patrimônio-alvo com a "
        "probabilidade exigida.",
    )
    cart = estado.carteira()

    c1, c2 = st.columns(2)
    meta_val = c1.number_input("Meta de patrimônio (R$)", 0.0, step=1000.0,
                               value=float(cart.capitalTotal * 1.5))
    prob = c2.slider("Probabilidade exigida", 0.50, 0.99, 0.80, 0.01,
                     help="Certeza mínima de atingir a meta.")

    n_sim = st.session_state["numSimulacoes"]
    rebal = st.session_state["diasRebalanceamento"]
    chave = estado.chave_simulacao(cart.pesos_rv, cart.diasInvestimento, rebal, n_sim)
    em_cache = ui.status_cache(chave, do_grupo=True)

    if ui.botao_calcular(FUNC, em_cache):
        cached = estado.obter_simulacao(chave)
        resultado, sim = ui.executar_com_spinner(
            em_cache,
            lambda: interface.meta(
                capitalTotal=cart.capitalTotal,
                meta=meta_val,
                probabilidade=prob,
                proporcaoAcao=cart.pesos_rv,
                params=cart.params,
                rf=cart.paramsRF,
                diasInvestimento=cart.diasInvestimento,
                numSimulacoes=n_sim,
                diasRebalanceamento=rebal,
                retornosCumulativos=cached,
            ),
        )
        estado.guardar_simulacao(chave, sim.resultadoCumulativo)
        ui.guardar_resultado(FUNC, resultado)

    res = ui.resultado_anterior(FUNC)
    if res is not None:
        _exibir(res)


def _exibir(res) -> None:
    st.divider()
    if res.atingivel:
        callout(f"{badge('Atingível', 'success')} &nbsp; Meta alcançável com a alocação encontrada.",
                variant="success")
    else:
        callout(f"{badge('Inatingível', 'destructive')} &nbsp; Nem 100% em RV atinge a meta com a "
                "probabilidade exigida.", variant="destructive")

    m1, m2, m3 = st.columns(3)
    m1.metric("Alocar em RF", reais(res.alocadoRendaFixa),
              pct(res.alocadoRendaFixa / res.capitalTotal))
    m2.metric("Alocar em RV", reais(res.alocadoRendaVariavel),
              pct(res.alocadoRendaVariavel / res.capitalTotal))
    m3.metric("Prob. real de atingir", pct(res.probabilidade_real),
              f"alvo {pct(res.probabilidade_alvo)}")

    if len(res.distribuicaoPatrimonio):
        st.markdown("**Distribuição do patrimônio final**")
        st.pyplot(histograma_patrimonio(res.distribuicaoPatrimonio,
                                        res.capitalTotal, meta=res.meta))
