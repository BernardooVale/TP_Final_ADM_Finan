"""
Interface Streamlit da Engine Quantitativa de Portfólio.

Execute a partir da raiz do repositório:

    streamlit run app/main.py

Fluxo: o usuário define a carteira na barra lateral e calibra uma única vez
(passo caro: download + MLE). Em seguida navega pelas 8 funcionalidades. As
de alocação/comparação/meta/duplo objetivo compartilham o mesmo Monte Carlo
via cache (ver app/estado.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que a raiz do repo (pai de app/) esteja no path para importar
# engine/modelos/data/interface, independentemente do diretório de execução.
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import streamlit as st

from modelos.defs import FrequenciaRentabilidadeRendaFix
from app import estado
from app.paginas import (
    alocacao, comparacao, meta, duplo_objetivo,
    fronteira, alocacao_otimizada, desacumulacao, tempo_meta,
)

st.set_page_config(page_title="Engine de Portfólio", page_icon="📈", layout="wide")

# Registro de páginas: rótulo -> (função render, é do grupo de cache?)
PAGINAS = {
    "📊 Alocação RF × RV":        (alocacao.render, True),
    "⚖️ Comparador":              (comparacao.render, True),
    "🎯 Meta de patrimônio":      (meta.render, True),
    "🔀 Duplo objetivo":          (duplo_objetivo.render, True),
    "📈 Fronteira eficiente":     (fronteira.render, False),
    "🧮 Otimizar carteira":       (alocacao_otimizada.render, False),
    "💸 Desacumulação":           (desacumulacao.render, False),
    "⏱️ Tempo para meta":         (tempo_meta.render, False),
}

FREQ_RF_LABEL = {
    FrequenciaRentabilidadeRendaFix.ANUAL: "ao ano",
    FrequenciaRentabilidadeRendaFix.TRIMESTRAL: "ao trimestre",
    FrequenciaRentabilidadeRendaFix.MENSAL: "ao mês",
    FrequenciaRentabilidadeRendaFix.DIARIO: "ao dia",
}


def _parse_lista(texto: str) -> list[str]:
    return [t.strip().upper() for t in texto.replace(";", ",").split(",") if t.strip()]


def _parse_pesos(texto: str, n: int) -> list[float]:
    """Lê pesos separados por vírgula; se não casar com n tickers, usa equal-weight."""
    try:
        vals = [float(x.strip().replace(",", ".")) for x in texto.split(",") if x.strip()]
    except ValueError:
        vals = []
    if len(vals) != n or not vals:
        return [1.0 / n] * n if n else []
    return vals


def sidebar_setup() -> None:
    st.sidebar.title("📈 Carteira")
    st.sidebar.caption("Defina os ativos e calibre uma vez. A calibração baixa o "
                       "histórico e ajusta as distribuições (passo lento).")

    with st.sidebar.form("setup"):
        tickers_txt = st.text_input("Tickers (yfinance, separados por vírgula)",
                                    value="PETR4.SA, VALE3.SA, ITUB4.SA")
        pesos_txt = st.text_input("Pesos da RV (mesma ordem; vazio = iguais)",
                                  value="0.4, 0.35, 0.25")
        periodo = st.selectbox("Histórico", ["1y", "2y", "3y", "5y", "10y"], index=2)

        st.divider()
        capital = st.number_input("Capital total (R$)", 1000.0, step=1000.0, value=100_000.0)
        dias = st.number_input("Prazo (dias úteis)", 21, 5040, 252, step=21,
                               help="252 dias úteis ≈ 1 ano.")

        st.divider()
        rf_pct = st.number_input("Taxa da Renda Fixa (%)", 0.0, 100.0, 14.5, 0.1)
        freq_rf = st.selectbox("Frequência da taxa RF", list(FrequenciaRentabilidadeRendaFix),
                               index=list(FrequenciaRentabilidadeRendaFix).index(
                                   FrequenciaRentabilidadeRendaFix.ANUAL),
                               format_func=lambda f: f"{f.value} ({FREQ_RF_LABEL[f]})")

        calibrar = st.form_submit_button("🔄 Calibrar carteira", type="primary",
                                         width="stretch")

    # Controles globais fora do form (afetam o cache, não exigem recalibração).
    st.sidebar.divider()
    qualidade = st.sidebar.selectbox("Qualidade da simulação", list(estado.QUALIDADES),
                                     index=1,
                                     help="Mais simulações = mais preciso e mais lento.")
    st.session_state["numSimulacoes"] = estado.QUALIDADES[qualidade]

    rebal = st.sidebar.number_input("Rebalancear a cada N dias (0 = nunca)", 0, 252, 0, step=21)
    st.session_state["diasRebalanceamento"] = rebal if rebal > 0 else None

    if estado.carteira_pronta():
        st.sidebar.button("🗑️ Limpar cache de simulações", on_click=estado.limpar_cache,
                          width="stretch")

    if calibrar:
        tickers = _parse_lista(tickers_txt)
        if not tickers:
            st.sidebar.error("Informe ao menos um ticker.")
            return
        pesos = _parse_pesos(pesos_txt, len(tickers))
        try:
            with st.spinner("Baixando histórico e calibrando..."):
                estado.calibrar(
                    tickers=tickers,
                    pesos_rv=pesos,
                    periodo=periodo,
                    capitalTotal=capital,
                    rentabilidadeRF=rf_pct / 100.0,
                    freqRF=freq_rf,
                    diasInvestimento=int(dias),
                )
            st.sidebar.success("Carteira calibrada!")
        except Exception as e:  # noqa: BLE001 — feedback direto ao usuário
            st.sidebar.error(f"Falha na calibração: {e}")


def main() -> None:
    estado.init_estado()
    sidebar_setup()

    st.title("Engine Quantitativa de Portfólio")

    if not estado.carteira_pronta():
        st.info("👈 Defina a carteira na barra lateral e clique em **Calibrar carteira** "
                "para começar.")
        st.markdown(
            "Esta interface expõe as 8 funcionalidades do motor de simulação:\n"
            "- **Alocação, Comparador, Meta e Duplo objetivo** compartilham uma mesma "
            "simulação de Monte Carlo (cache) enquanto pesos/prazo/qualidade não mudarem.\n"
            "- **Fronteira, Otimizar carteira, Desacumulação e Tempo para meta** rodam "
            "simulações próprias a cada cálculo."
        )
        return

    escolha = st.radio("Funcionalidade", list(PAGINAS), horizontal=True,
                       label_visibility="collapsed")
    st.divider()
    render, _ = PAGINAS[escolha]
    render()


main()
