"""Alocação: quanto colocar em RF vs RV para cobrir o capital no cenário alvo."""
from __future__ import annotations

import numpy as np
import streamlit as st

import interface
from modelos.defs import RiscoAlvo
from app import estado, componentes as ui
from app.formatacao import reais, pct
from app.graficos import histograma_patrimonio, barras_alocacao

FUNC = "alocacao"


def render() -> None:
    ui.cabecalho(
        "Alocação RF × RV",
        "Resolve quanto alocar em Renda Fixa e Renda Variável para que o "
        "patrimônio final cubra o capital inicial no cenário de estresse (CVaR).",
    )
    ui.info_carteira_sidebar()
    cart = estado.carteira()

    c1, c2 = st.columns(2)
    confianca = c1.slider("Confiança do CVaR", 0.80, 0.99, 0.95, 0.01,
                          help="Nível de confiança da cauda de perda.")
    risco = c2.selectbox("Cenário de risco", list(RiscoAlvo),
                         format_func=lambda r: {"media": "Média da cauda (CVaR)",
                                                "pior": "Pior cenário absoluto"}[r.value])

    with st.expander("Aportes periódicos (opcional)"):
        valorAporte, freq = ui.inputs_aporte(FUNC)

    n_sim = st.session_state["numSimulacoes"]
    rebal = st.session_state["diasRebalanceamento"]
    chave = estado.chave_simulacao(cart.pesos_rv, cart.diasInvestimento, rebal, n_sim)
    em_cache = ui.status_cache(chave, do_grupo=True)

    if ui.botao_calcular(FUNC, em_cache):
        cached = estado.obter_simulacao(chave)
        resultado, sim = ui.executar_com_spinner(
            em_cache,
            lambda: interface.alocacao(
                capitalTotal=cart.capitalTotal,
                tickers=cart.tickers,
                proporcaoAcao=list(cart.pesos_rv),
                paramsRF=cart.paramsRF,
                params=cart.params,
                riscoAlvo=risco,
                diasInvestimento=cart.diasInvestimento,
                confianca=confianca,
                numSimulacoes=n_sim,
                diasRebalanceamento=rebal,
                valorAporte=valorAporte,
                frequenciaAporte=freq,
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
    cobre = res.patrimonioFinal >= res.capitalTotal
    m1, m2, m3 = st.columns(3)
    m1.metric("Alocado em RF", reais(res.alocadoRendaFixa),
              pct(res.alocadoRendaFixa / res.capitalTotal))
    m2.metric("Alocado em RV", reais(res.alocadoRendaVariavel),
              pct(res.alocadoRendaVariavel / res.capitalTotal))
    m3.metric("Patrimônio final (cenário alvo)", reais(res.patrimonioFinal),
              "✓ cobre o capital" if cobre else "✗ insuficiente",
              delta_color="normal" if cobre else "inverse")

    st.pyplot(barras_alocacao(res.alocadoRendaFixa, res.alocadoRendaVariavel))

    m4, m5, m6 = st.columns(3)
    m4.metric("RF ao fim do prazo", reais(res.saldoFinalRendaFixa))
    m5.metric("Sharpe", f"{res.sharpe:.3f}")
    m6.metric("Sortino", f"{res.sortino:.3f}")

    if len(res.distribuicaoPatrimonio):
        st.markdown("**Distribuição do patrimônio final**")
        st.pyplot(histograma_patrimonio(res.distribuicaoPatrimonio, res.capitalTotal))
        ps = np.percentile(res.distribuicaoPatrimonio, [5, 25, 50, 75, 95])
        st.dataframe(
            {"Percentil": [f"P{p}" for p in (5, 25, 50, 75, 95)],
             "Patrimônio": [reais(v) for v in ps]},
            hide_index=True, width='stretch',
        )
