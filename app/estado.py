"""
Gerência de estado da interface (session_state) e cache de simulações.

A calibração (download + MLE) e o Monte Carlo são caros. Para evitar
re-simular a cada interação, guardamos no `st.session_state`:

1. Os parâmetros calibrados da carteira (`params`, `tickers`, `paramsRF`...),
   gerados uma única vez por "rodada de calibração" (`calib_id`).
2. Um cache de arrays `retornosCumulativos` (objeto Simulacao), indexado por
   uma chave que captura TUDO que altera o resultado do Monte Carlo:
   pesos da RV, horizonte, rebalanceamento, nº de simulações e a calibração.

As funcionalidades que aceitam `retornosCumulativos` (alocação, comparação,
meta e duplo objetivo) compartilham esse cache — basta a chave bater. As
demais (fronteira, desacumulação, otimização e tempo-meta) rodam simulações
próprias internas e não usam esse cache.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from modelos.params import ParametrosCalibrados, ParametrosRF
from modelos.defs import FrequenciaRentabilidadeRendaFix
from data.mineracao import baixar_retornos
from data.calibracoes import calibrar_todos
from renda_fixa import preparar_parametros_rf


# Funcionalidades que compartilham o cache de `retornosCumulativos`.
GRUPO_CACHE = {"alocacao", "comparacao", "meta", "duploObjetivo"}

# Mapeia o seletor de qualidade para o nº de simulações de Monte Carlo.
QUALIDADES = {
    "Rápido (100 mil)":   100_000,
    "Padrão (500 mil)":   500_000,
    "Preciso (1 milhão)": 1_000_000,
}


@dataclass
class Carteira:
    """Tudo que define a carteira calibrada do usuário, fixado na calibração."""
    calib_id:         int
    tickers:          list[str]
    pesos_rv:         np.ndarray
    params:           ParametrosCalibrados
    paramsRF:         ParametrosRF
    capitalTotal:     float
    diasInvestimento: int
    rentabilidadeRF:  float
    freqRF:           FrequenciaRentabilidadeRendaFix
    periodo:          str


# ──────────────────────────── inicialização ────────────────────────────

def init_estado() -> None:
    """Garante as chaves base do session_state."""
    ss = st.session_state
    ss.setdefault("carteira", None)        # Carteira | None
    ss.setdefault("calib_id", 0)           # incrementa a cada calibração
    ss.setdefault("cache_sim", {})         # chave -> np.ndarray (retornosCumulativos)
    ss.setdefault("numSimulacoes", 500_000)
    ss.setdefault("diasRebalanceamento", None)


def carteira() -> Carteira | None:
    return st.session_state.get("carteira")


def carteira_pronta() -> bool:
    return st.session_state.get("carteira") is not None


# ──────────────────────────── calibração ────────────────────────────

def calibrar(
    tickers:          list[str],
    pesos_rv:         list[float],
    periodo:          str,
    capitalTotal:     float,
    rentabilidadeRF:  float,
    freqRF:           FrequenciaRentabilidadeRendaFix,
    diasInvestimento: int,
) -> Carteira:
    """
    Roda o passo caro (download + MLE) e fixa a carteira no estado.

    Reordena os pesos para acompanhar os tickers efetivamente válidos
    devolvidos por `baixar_retornos` (alguns podem ser descartados).
    Invalida o cache de simulações ao trocar de calibração.
    """
    pesos_por_ticker = dict(zip(tickers, pesos_rv))

    retornos, tickers_validos = baixar_retornos(tickers, periodo)
    params = calibrar_todos(retornos, tickers_validos)

    pesos = np.array([pesos_por_ticker.get(t, 0.0) for t in tickers_validos], dtype=float)
    soma = pesos.sum()
    pesos = pesos / soma if soma > 0 else np.full(len(tickers_validos), 1 / len(tickers_validos))

    paramsRF = preparar_parametros_rf(rentabilidadeRF, freqRF, diasInvestimento)

    st.session_state["calib_id"] += 1
    st.session_state["cache_sim"] = {}   # nova calibração invalida o cache

    cart = Carteira(
        calib_id=st.session_state["calib_id"],
        tickers=tickers_validos,
        pesos_rv=pesos,
        params=params,
        paramsRF=paramsRF,
        capitalTotal=capitalTotal,
        diasInvestimento=diasInvestimento,
        rentabilidadeRF=rentabilidadeRF,
        freqRF=freqRF,
        periodo=periodo,
    )
    st.session_state["carteira"] = cart
    return cart


# ──────────────────────────── cache de simulação ────────────────────────────

def chave_simulacao(
    pesos_rv:            np.ndarray,
    diasInvestimento:    int,
    diasRebalanceamento: int | None,
    numSimulacoes:       int,
) -> tuple:
    """
    Chave determinística para o array `retornosCumulativos`.

    Inclui `calib_id` para que recalibrar (trocar ativos/período) descarte
    automaticamente caches antigos. Os pesos são arredondados para evitar
    misses por ruído de ponto flutuante na UI.
    """
    cart = carteira()
    calib_id = cart.calib_id if cart else -1
    pesos = tuple(round(float(p), 6) for p in np.asarray(pesos_rv).ravel())
    return (calib_id, pesos, int(diasInvestimento), diasRebalanceamento, int(numSimulacoes))


def obter_simulacao(chave: tuple) -> np.ndarray | None:
    return st.session_state["cache_sim"].get(chave)


def guardar_simulacao(chave: tuple, retornosCumulativos: np.ndarray) -> None:
    st.session_state["cache_sim"][chave] = retornosCumulativos


def limpar_cache() -> None:
    st.session_state["cache_sim"] = {}
