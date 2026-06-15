"""Desacumulação: probabilidade de ruína na fase de saques (aposentadoria)."""
from __future__ import annotations

import streamlit as st

import interface
from modelos.defs import FrequenciaAporte
from app import estado, componentes as ui
from app.formatacao import reais, pct, anos_dias
from app.tema import badge, callout

FUNC = "desacumulacao"


def render() -> None:
    ui.cabecalho(
        "Desacumulação (saques periódicos)",
        "Simula a fase de usufruto: calcula a probabilidade de ruína para um saque "
        "programado e o maior saque sustentável dentro do limite de ruína.",
    )
    cart = estado.carteira()

    c1, c2 = st.columns(2)
    saque = c1.number_input("Saque por ciclo (R$)", 0.0, step=100.0,
                            value=float(cart.capitalTotal * 0.01))
    freq = c2.selectbox("Frequência do saque", list(FrequenciaAporte),
                        format_func=lambda f: f.value)
    c3, c4 = st.columns(2)
    fracao_rv = c3.slider("Exposição à RV", 0.0, 1.0, 0.5, 0.05,
                          help="Fração do capital exposta à bolsa durante a desacumulação.")
    limite_ruina = c4.slider("Limite de ruína aceitável", 0.01, 0.20, 0.05, 0.01,
                             help="Ruína tolerada na busca do saque sustentável (engine exige > 0).")

    ui.status_cache(chave=None, do_grupo=False)

    if ui.botao_calcular(FUNC, em_cache=False):
        resultado = ui.executar_com_spinner(
            False,
            lambda: interface.desacumulacao(
                capitalTotal=cart.capitalTotal,
                saque=saque,
                frequenciaSaque=freq,
                fracao_rv=fracao_rv,
                propocaoAcao=cart.pesos_rv,
                tickers=cart.tickers,
                params=cart.params,
                rf=cart.paramsRF,
                diasInvestimento=cart.diasInvestimento,
                numSimulacoes=st.session_state["numSimulacoes"],
                diasRebalanceamento=st.session_state["diasRebalanceamento"],
                limite_ruina=limite_ruina,
            ),
        )
        ui.guardar_resultado(FUNC, resultado)

    res = ui.resultado_anterior(FUNC)
    if res is not None:
        _exibir(res)


def _exibir(res) -> None:
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Prob. de ruína", pct(res.prob_ruina),
              delta_color="inverse")
    m2.metric("Prob. de sobreviver", pct(res.prob_sobreviver))
    if res.patrimonio_mediano is not None:
        m3.metric("Patrimônio mediano (sobreviventes)", reais(res.patrimonio_mediano))

    if res.saque_sustentavel is not None:
        callout(
            f"{badge('Saque sustentável', 'success')} &nbsp; "
            f"<strong>{reais(res.saque_sustentavel)}</strong> / {res.frequencia_saque.value} "
            f"(ruína ≤ {pct(res.limite_ruina_alvo)}).",
            variant="success",
        )
    else:
        callout(f"{badge('Sem saque sustentável', 'destructive')} &nbsp; Nenhum valor mantém a "
                f"ruína ≤ {pct(res.limite_ruina_alvo)}.", variant="destructive")

    dias = res.percentis_duracao.get("dias", {})
    if dias:
        st.markdown("**Duração do patrimônio (cenários que arruínam)**")
        st.dataframe(
            [{"Percentil": f"P{p}", "Duração": anos_dias(d)} for p, d in dias.items()],
            hide_index=True, width='stretch',
        )
    else:
        st.caption("Nenhum cenário arruinou no horizonte simulado.")
