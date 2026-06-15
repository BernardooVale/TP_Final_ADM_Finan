"""Duplo objetivo: piso de segurança + meta de crescimento (fronteira de Pareto)."""
from __future__ import annotations

import streamlit as st

import interface
from modelos.pareto import RestricaoPiso, RestricaoMeta
from app import estado, componentes as ui
from app.formatacao import reais, pct
from app.graficos import pareto_piso_meta

FUNC = "duploObjetivo"


def render() -> None:
    ui.cabecalho(
        "Duplo objetivo (piso + meta)",
        "Encontra alocações que respeitam simultaneamente um piso (não perder "
        "além de X com confiança α) e uma meta (atingir Y com confiança β).",
    )
    ui.info_carteira_sidebar()
    cart = estado.carteira()

    st.markdown("**Piso de segurança**")
    p1, p2 = st.columns(2)
    piso_valor = p1.number_input("Patrimônio mínimo (R$)", 0.0, step=1000.0,
                                 value=float(cart.capitalTotal * 0.95))
    piso_conf = p2.slider("Confiança do piso", 0.50, 0.99, 0.95, 0.01)

    st.markdown("**Meta de crescimento**")
    m1, m2 = st.columns(2)
    meta_valor = m1.number_input("Patrimônio-alvo (R$)", 0.0, step=1000.0,
                                 value=float(cart.capitalTotal * 1.25))
    meta_conf = m2.slider("Confiança da meta", 0.50, 0.99, 0.60, 0.01)

    n_pontos = st.slider("Pontos na fronteira de Pareto", 3, 11, 5)

    n_sim = st.session_state["numSimulacoes"]
    rebal = st.session_state["diasRebalanceamento"]
    chave = estado.chave_simulacao(cart.pesos_rv, cart.diasInvestimento, rebal, n_sim)
    em_cache = ui.status_cache(chave, do_grupo=True)

    if ui.botao_calcular(FUNC, em_cache):
        cached = estado.obter_simulacao(chave)
        resultado, sim = ui.executar_com_spinner(
            em_cache,
            lambda: interface.duploObjetivo(
                capitalTotal=cart.capitalTotal,
                piso=RestricaoPiso(piso_valor, piso_conf),
                meta=RestricaoMeta(meta_valor, meta_conf),
                proporcaoAcao=cart.pesos_rv,
                tickers=cart.tickers,
                params=cart.params,
                rf=cart.paramsRF,
                diasInvestimento=cart.diasInvestimento,
                numSimulacoes=n_sim,
                diasRebalanceamento=rebal,
                n_pontos_pareto=n_pontos,
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
    if not res.viavel or res.ponto_minimo_rv is None:
        st.error(f"Sem solução viável. {res.mensagem}")
        return

    st.success(res.mensagem)
    cap = res.capitalTotal

    def card(col, ponto, titulo):
        col.markdown(f"**{titulo}**")
        col.metric("RV", reais(ponto.alocRV), pct(ponto.alocRV / cap))
        col.metric("RF", reais(ponto.alocRF), pct(ponto.alocRF / cap))
        col.caption(f"P(≥ piso): {pct(ponto.prob_piso)} · P(≥ meta): {pct(ponto.prob_meta)}")

    c1, c2 = st.columns(2)
    card(c1, res.ponto_minimo_rv, "Conservador (mín. RV)")
    if res.ponto_maximo_rv is not None:
        card(c2, res.ponto_maximo_rv, "Agressivo (máx. RV)")

    if len(res.fronteira) > 1:
        st.markdown("**Fronteira de Pareto**")
        st.pyplot(pareto_piso_meta(res.fronteira, cap))
        st.dataframe(
            [{"RV": reais(p.alocRV), "RF": reais(p.alocRF),
              "P(≥ piso)": pct(p.prob_piso), "P(≥ meta)": pct(p.prob_meta)}
             for p in res.fronteira],
            hide_index=True, width='stretch',
        )
