from dataclasses import dataclass
import numpy as np
from enum import Enum
    
class TipoEstrategiaBase(str, Enum):
    """Estratégias fixas disponíveis para comparação."""
    RF_100       = "100% RF"
    RV_100       = "100% RV"
    RV75_RF25    = "75% RV / 25% RF"
    RV25_RF75    = "25% RV / 75% RF"


@dataclass
class EstrategiaUsuario:
    """
    Estratégia customizada do usuário para o comparador.

    Aceita qualquer resultado que contenha uma distribuição de patrimônio
    já simulada (AlocacaoResultado ou ResultadoMeta).

    Parâmetros
    ----------
    nome                  : rótulo exibido na tabela comparativa
    distribuicaoPatrimonio: array (n_sim,) de patrimônio final em R$
                            — extraído de resultado.distribuicaoPatrimonio
    fracao_rv             : fração em RV usada (para exibição); None = desconhecido
    """
    nome:                   str
    distribuicaoPatrimonio: np.ndarray
    fracao_rv:              float | None = None


@dataclass
class MetricasEstrategia:
    """Métricas comparativas de uma estratégia."""
    nome:          str
    fracao_rv:     float | None
    q1:            float   # P25 do patrimônio final (R$)
    mediana:       float   # P50
    q3:            float   # P75
    media:         float
    prob_meta:     float | None   # P(patrimônio >= meta); None se meta não definida
    prob_perda:    float          # P(patrimônio < capitalTotal)
    retorno_medio: float          # (media / capitalTotal) - 1