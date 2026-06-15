"""Componentes de UI reutilizados pelas páginas das funcionalidades."""
from __future__ import annotations

from typing import Callable

import numpy as np
import streamlit as st

from modelos.defs import FrequenciaAporte
from app import estado


def cabecalho(titulo: str, descricao: str) -> None:
    st.subheader(titulo)
    st.caption(descricao)


def status_cache(chave: tuple, do_grupo: bool) -> bool:
    """
    Mostra se a simulação será reaproveitada do cache ou rodada do zero.

    Retorna True se já existe simulação em cache para a chave atual.
    Para funcionalidades fora do grupo de cache, apenas avisa que a
    simulação é própria e sempre executada.
    """
    if not do_grupo:
        st.info(
            "ℹ️ Esta funcionalidade roda uma **simulação própria** a cada cálculo "
            "(não compartilha o cache de alocação/comparação/meta). Pode demorar."
        )
        return False

    em_cache = estado.obter_simulacao(chave) is not None
    if em_cache:
        st.success(
            "♻️ **Simulação em cache** para esta carteira/horizonte — o resultado "
            "sai em milissegundos, sem rodar o Monte Carlo de novo."
        )
    else:
        st.warning(
            "⚠️ Não há simulação em cache para esta combinação de **pesos / prazo / "
            "qualidade**. Calcular agora vai rodar um novo Monte Carlo (pode demorar)."
        )
    return em_cache


def botao_calcular(funcionalidade: str, em_cache: bool) -> bool:
    """Botão de execução, com rótulo coerente ao estado do cache."""
    rotulo = "Calcular (instantâneo — em cache)" if em_cache else "Rodar simulação"
    return st.button(rotulo, type="primary", key=f"btn_{funcionalidade}")


def guardar_resultado(funcionalidade: str, resultado) -> None:
    st.session_state[f"res_{funcionalidade}"] = resultado


def resultado_anterior(funcionalidade: str):
    return st.session_state.get(f"res_{funcionalidade}")


def executar_com_spinner(em_cache: bool, fn: Callable):
    """Roda `fn` exibindo spinner condizente com o custo esperado."""
    msg = "Recuperando do cache..." if em_cache else "Rodando Monte Carlo — aguarde..."
    with st.spinner(msg):
        return fn()


_SEM_APORTE = "Sem aporte"


def inputs_aporte(chave: str) -> tuple[float, FrequenciaAporte | None]:
    """
    Inputs de aporte periódico (valor + frequência).

    Usa um rótulo textual em vez de `None` como opção do selectbox — evita um
    bug de serialização do AppTest e deixa a opção "sem aporte" explícita.
    """
    a1, a2 = st.columns(2)
    valor = a1.number_input("Valor do aporte (R$)", 0.0, step=100.0, value=0.0,
                            key=f"aporte_val_{chave}")
    opcoes = [_SEM_APORTE] + [f.value for f in FrequenciaAporte]
    escolha = a2.selectbox("Frequência do aporte", opcoes, key=f"aporte_freq_{chave}")
    freq = None if escolha == _SEM_APORTE else FrequenciaAporte(escolha)
    return valor, freq


def info_carteira_sidebar() -> None:
    """Resumo compacto da carteira ativa, exibido no topo de cada página."""
    cart = estado.carteira()
    if cart is None:
        return
    pesos = "  ".join(
        f"`{t}` {p*100:.0f}%" for t, p in zip(cart.tickers, cart.pesos_rv)
    )
    st.caption(
        f"**Carteira ativa:** {pesos}  ·  capital "
        f"{_reais(cart.capitalTotal)}  ·  prazo {cart.diasInvestimento}d  ·  "
        f"qualidade {st.session_state['numSimulacoes']:,} sims".replace(",", ".")
    )


def _reais(v: float) -> str:
    from app.formatacao import reais
    return reais(v)
