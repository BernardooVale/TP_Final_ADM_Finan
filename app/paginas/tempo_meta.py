"""Tempo para meta: em quanto tempo o patrimônio cruza o alvo."""
from __future__ import annotations

import streamlit as st

import interface
from app import estado, componentes as ui
from app.formatacao import reais, pct, anos_dias

FUNC = "tempoMeta"


def render() -> None:
    ui.cabecalho(
        "Tempo para atingir meta",
        "Distribui o tempo (anos/dias úteis) até o patrimônio cruzar a meta, "
        "considerando aportes opcionais.",
    )
    ui.info_carteira_sidebar()
    cart = estado.carteira()

    c1, c2 = st.columns(2)
    meta_val = c1.number_input("Meta de patrimônio (R$)", 0.0, step=1000.0,
                               value=float(cart.capitalTotal * 2.0))
    fracao_rv = c2.slider("Exposição à RV", 0.0, 1.0, 0.6, 0.05)

    with st.expander("Aportes periódicos (opcional)"):
        valorAporte, freq = ui.inputs_aporte(FUNC)

    ui.status_cache(chave=None, do_grupo=False)

    if ui.botao_calcular(FUNC, em_cache=False):
        resultado = ui.executar_com_spinner(
            False,
            lambda: interface.tempoMeta(
                capitalTotal=cart.capitalTotal,
                meta=meta_val,
                fracao_rv=fracao_rv,
                proporcaoAcao=cart.pesos_rv,
                tickers=cart.tickers,
                params=cart.params,
                rf=cart.paramsRF,
                diasInvestimento=cart.diasInvestimento,
                numSimulacoes=st.session_state["numSimulacoes"],
                diasRebalanceamento=st.session_state["diasRebalanceamento"],
                valorAporte=valorAporte,
                frequenciaAporte=freq,
            ),
        )
        ui.guardar_resultado(FUNC, resultado)

    res = ui.resultado_anterior(FUNC)
    if res is not None:
        _exibir(res)


def _exibir(res) -> None:
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Prob. de atingir a meta", pct(res.prob_atingir),
              f"no horizonte de {res.max_dias} dias úteis")
    m2.metric("Cenários que não atingiram", f"{res.dias_nao_atingido:,}".replace(",", "."))

    if res.percentis_anos:
        st.markdown("**Tempo até atingir a meta (cenários que atingem)**")
        st.dataframe(
            [{"Percentil": f"P{p}", "Tempo": anos_dias(res.percentis_dias[p])}
             for p in res.percentis_anos],
            hide_index=True, width='stretch',
        )
    else:
        st.caption("Nenhum cenário atingiu a meta no horizonte simulado.")
