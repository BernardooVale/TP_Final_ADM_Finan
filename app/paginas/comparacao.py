"""Comparação lado a lado de estratégias estáticas de alocação."""
from __future__ import annotations

import numpy as np
import streamlit as st

import interface
from modelos.estrategias import TipoEstrategiaBase, EstrategiaUsuario
from app import estado, componentes as ui
from app.formatacao import reais, pct, pct_sinal
from app.tema import badge, callout

FUNC = "comparacao"


def render() -> None:
    ui.cabecalho(
        "Comparador de estratégias",
        "Mede Q1/Mediana/Q3, retorno médio, probabilidade de perda e de meta "
        "para estratégias fixas (100% RF, 100% RV, mistas) sobre o mesmo capital.",
    )
    cart = estado.carteira()

    bases = st.multiselect(
        "Estratégias a comparar", list(TipoEstrategiaBase),
        default=list(TipoEstrategiaBase),
        format_func=lambda e: e.value,
    )
    c1, c2 = st.columns(2)
    usar_meta = c1.checkbox("Definir meta de patrimônio")
    meta = c2.number_input("Meta (R$)", 0.0, step=1000.0,
                           value=float(cart.capitalTotal * 1.3),
                           disabled=not usar_meta)

    incluir_usuario = st.checkbox(
        "Incluir minha carteira (usa a última distribuição da aba Alocação)",
        value=False,
        help="Disponível depois de calcular uma Alocação com os pesos atuais.",
    )

    n_sim = st.session_state["numSimulacoes"]
    rebal = st.session_state["diasRebalanceamento"]
    chave = estado.chave_simulacao(cart.pesos_rv, cart.diasInvestimento, rebal, n_sim)
    em_cache = ui.status_cache(chave, do_grupo=True)

    if ui.botao_calcular(FUNC, em_cache):
        cached = estado.obter_simulacao(chave)
        est_usuario = _estrategia_usuario() if incluir_usuario else None
        resultado, sim = ui.executar_com_spinner(
            em_cache,
            lambda: interface.comparacao(
                capitalTotal=cart.capitalTotal,
                propocaoAcao=cart.pesos_rv,
                params=cart.params,
                rf=cart.paramsRF,
                diasInvestimento=cart.diasInvestimento,
                estrategia_usuario=est_usuario,
                estrategias_base=bases or list(TipoEstrategiaBase),
                meta=meta if usar_meta else None,
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


def _estrategia_usuario() -> EstrategiaUsuario | None:
    aloc = ui.resultado_anterior("alocacao")
    if aloc is None or not len(getattr(aloc, "distribuicaoPatrimonio", [])):
        callout("Calcule a aba Alocação primeiro para incluir sua carteira.", variant="warning")
        return None
    frac_rv = aloc.alocadoRendaVariavel / aloc.capitalTotal
    return EstrategiaUsuario("Minha carteira", aloc.distribuicaoPatrimonio, frac_rv)


def _exibir(res) -> None:
    st.divider()
    linhas = []
    for e in res.estrategias:
        linhas.append({
            "Estratégia": e.nome,
            "Q1": reais(e.q1),
            "Mediana": reais(e.mediana),
            "Q3": reais(e.q3),
            "Retorno médio": pct_sinal(e.retorno_medio),
            "P(meta)": pct(e.prob_meta) if e.prob_meta is not None else "—",
            "P(perda)": pct(e.prob_perda),
        })
    st.dataframe(linhas, hide_index=True, width='stretch')
    st.caption("Ordenado por mediana decrescente. P(perda) = chance de terminar abaixo do capital inicial.")

    melhor = max(res.estrategias, key=lambda e: e.mediana)
    callout(f"{badge('Maior mediana', 'default')} &nbsp; <strong>{melhor.nome}</strong> — "
            f"{reais(melhor.mediana)}", variant="secondary")
