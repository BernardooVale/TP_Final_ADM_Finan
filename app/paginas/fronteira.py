"""Fronteira eficiente CVaR: trade-off risco de cauda × retorno."""
from __future__ import annotations

import streamlit as st

import interface
from app import estado, componentes as ui
from app.formatacao import pct, pct_sinal
from app.graficos import fronteira_cvar
from app.tema import badge, callout

FUNC = "fronteira"


def render() -> None:
    ui.cabecalho(
        "Fronteira eficiente (CVaR)",
        "Gera a curva de carteiras que minimizam o risco de cauda (CVaR) para "
        "cada nível de retorno-alvo.",
    )
    cart = estado.carteira()

    c1, c2, c3 = st.columns(3)
    confianca = c1.slider("Confiança do CVaR", 0.80, 0.99, 0.95, 0.01)
    n_pontos = c2.slider("Resolução (pontos)", 2, 7, 3)
    n_sim = c3.select_slider("Simulações por ponto",
                             [8_000, 15_000, 25_000, 50_000], value=8_000,
                             help="Cada ponto roda uma otimização completa — valores altos demoram.")
    poupa = st.toggle("Modo rápido (recomendado)", value=True,
                      help="Reduz o orçamento do otimizador. Cada ponto cai de ~minutos "
                           "para ~segundos, com uma fronteira mais grosseira.")

    ui.status_cache(chave=None, do_grupo=False)
    callout(
        f"{badge('Mais pesada', 'warning')} &nbsp; Roda uma otimização CVaR por ponto "
        f"(~{n_pontos} pontos). Com o modo rápido desligado pode levar vários minutos por ponto.",
        variant="warning",
    )

    if ui.botao_calcular(FUNC, em_cache=False):
        resultado = ui.executar_com_spinner(
            False,
            lambda: interface.fronteira(
                params=cart.params,
                tickers=cart.tickers,
                rf=cart.paramsRF,
                diasInvest=cart.diasInvestimento,
                confianca=confianca,
                numPontos=n_pontos,
                numSimulacoes=n_sim,
                diasRebalanceamento=st.session_state["diasRebalanceamento"],
                poupaTempo=poupa,
            ),
        )
        ui.guardar_resultado(FUNC, resultado)

    res = ui.resultado_anterior(FUNC)
    if res is not None:
        _exibir(res)


def _exibir(res) -> None:
    st.divider()
    if not res.pontos:
        callout("Nenhum ponto viável encontrado para os parâmetros informados.",
                variant="warning")
        return

    st.pyplot(fronteira_cvar(res.pontos))

    linhas = []
    for p in res.pontos:
        linha = {
            "Retorno médio": pct_sinal(p.retorno_medio),
            "CVaR": pct(p.cvar),
            "RV": pct(p.fracao_rv),
            "RF": pct(p.fracao_rf),
        }
        for t, w in zip(p.tickers, p.pesos_rv):
            linha[t] = pct(float(w))
        linhas.append(linha)
    st.dataframe(linhas, hide_index=True, width='stretch')
    st.caption("Pesos por ticker são as proporções dentro da parcela de RV.")
