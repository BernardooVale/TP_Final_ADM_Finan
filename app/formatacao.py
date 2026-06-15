"""Helpers de formatação para exibição na interface (R$, %, etc.)."""
from __future__ import annotations


def reais(valor: float) -> str:
    """Formata em Real com separador de milhar pt-BR. Ex.: R$ 1.234.567,89"""
    s = f"{valor:,.2f}"                      # 1,234,567.89
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def pct(fracao: float, casas: int = 1) -> str:
    """Formata uma fração [0,1] como porcentagem. Ex.: 0.873 -> 87,3%"""
    return f"{fracao * 100:.{casas}f}%".replace(".", ",")


def pct_sinal(fracao: float, casas: int = 1) -> str:
    """Porcentagem com sinal explícito (para retornos). Ex.: +12,4%"""
    return f"{fracao * 100:+.{casas}f}%".replace(".", ",")


def anos_dias(dias: float) -> str:
    """Converte dias úteis em texto 'X,X anos (N dias úteis)'."""
    anos = dias / 252
    return f"{anos:.1f} anos ({dias:,.0f} dias úteis)".replace(",", ".")
