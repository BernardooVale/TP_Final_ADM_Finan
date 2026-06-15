# Documentação da Engine Quantitativa de Portfólio

> **Front-end:** há uma interface Streamlit em [`app/`](app/README.md) que
> expõe as 8 funcionalidades. Para rodar:
> `.venv/bin/pip install -r requirements-app.txt && .venv/bin/streamlit run app/main.py`

## 1. Visão Geral da Arquitetura
Este repositório contém um motor de simulação estocástica de alta performance para análise e otimização de portfólios financeiros. A arquitetura foi desenhada para superar as limitações da teoria clássica de Markowitz, utilizando **Simulações de Monte Carlo**, **Distribuições Marginais t-Student** e **Cópulas-t** para modelar adequadamente assimetrias e eventos de cauda (cisnes negros).

O sistema está dividido em quatro camadas principais:
1. **Mineração e Calibração (`data/`):** Coleta da série histórica via `yfinance` e ajuste estatístico via Máxima Verossimilhança (MLE).
2. **Motor de Simulação (`engine/`):** Núcleo paralelizado em C via `Numba` (`kernels.py`) para geração massiva de inovações e trajetórias de forma otimizada.
3. **Modelos de Domínio (`modelos/`):** Estruturas estritas de dados (DataClasses e Enums) garantindo a tipagem e os contratos entre as camadas.
4. **Interface de Integração (`interface.py`):** Camada de fachada (Facade Pattern) projetada para comunicação limpa com o Front-end ou APIs (REST/GraphQL).

---

## 2. Modelos de Dados (Payloads de Comunicação)
*(Atenção Desenvolvedor Front-end / Agente de IA: Estes são os objetos que devem ser serializados/desserializados na comunicação)*

### Enums Base (`defs.py`)
* `RiscoAlvo`: String. Opções: `"media"` (CVaR clássico), `"pior"` (mínimo absoluto).
* `FrequenciaRentabilidadeRendaFix`: String. Opções: `"diario"`, `"mensal"`, `"trimestral"`, `"anual"`.
* `FrequenciaAporte`: String. Opções: `"mensal"`, `"trimestral"`, `"semestral"`.

### Objetos de Parâmetros (`params.py`)
* **`ParametrosCalibrados`**: Contém as matrizes matemáticas geradas pela etapa de calibração (`nus`, `mus`, `sigmas`, `corr`, `nu_copula`).
* **`ParametrosRF`**: Objeto carregando os juros equivalentes. Contém `crescimento` (float), `retorno_periodo` (float) e `taxa_diaria` (float). **Atenção:** Deve ser gerado exclusivamente pela função `preparar_parametros_rf`.

---

## 3. Contratos da Camada de Integração (`interface.py`)

Esta seção detalha as funções disponíveis para consumo externo. A maioria das funções retorna uma tupla estruturada contendo o **Resultado de Negócio** (exibição na tela) e o **Objeto de Simulação Prévia** (para uso em cache e otimização de tempo de CPU em chamadas subsequentes).

### 3.1. `alocacao`
**Objetivo:** Executa a simulação base para um dado conjunto de pesos estáticos, calculando a cobertura do capital inicial e extraindo métricas de risco (Sharpe, Sortino, CVaR).

* **Entradas (Inputs):**
  * `capitalTotal` *(float)*: Capital inicial investido (R$).
  * `tickers` *(list[str])*: Lista de símbolos dos ativos de Renda Variável (RV).
  * `proporcaoAcao` *(list[float])*: Pesos desejados para cada ativo (soma deve ser 1.0).
  * `paramsRF` *(ParametrosRF)*: Objeto instanciado de Renda Fixa.
  * `params` *(ParametrosCalibrados)*: Parâmetros calibrados dos ativos.
  * `riscoAlvo` *(RiscoAlvo)*: Métrica de estresse (média da cauda ou pior cenário).
  * `diasInvestimento` *(int)*: Prazo total em dias úteis.
  * `confianca` *(float, padrão=0.95)*: Nível de confiança para o CVaR.
  * `numSimulacoes` *(int, padrão=1_000_000)*: Quantidade de trajetórias.
  * `diasRebalanceamento` *(int | None, padrão=None)*: Frequência de rebalanceamento da carteira de RV.
  * `valorAporte` *(float, padrão=0.0)*: Valor do aporte em RF.
  * `frequenciaAporte` *(FrequenciaAporte | None, padrão=None)*: Frequência do aporte.
  * `retornosCumulativos` *(np.ndarray | None, padrão=None)*: Matriz preexistente para evitar re-simulação (Cache).
* **Retornos (Outputs):**
  * **[0] `AlocacaoResultado`**: Objeto contendo os dados nominais alocados, saldos projetados, métricas de risco e percentis de distribuição.
  * **[1] `Simulacao`**: Objeto contendo os retornos puros gerados (cacheável).

### 3.2. `comparacao`
**Objetivo:** Testa lado a lado múltiplas estratégias estáticas de distribuição (ex: 100% RF, 75% RV) contra a distribuição atual/sugerida do usuário.

* **Entradas (Inputs):**
  * `capitalTotal` *(float)*: Capital base.
  * `propocaoAcao` *(np.ndarray)*: Pesos da RV do usuário.
  * `params` *(ParametrosCalibrados)*.
  * `rf` *(ParametrosRF)*.
  * `diasInvestimento` *(int)*.
  * `estrategia_usuario` *(EstrategiaUsuario | None)*: Dados pré-calculados do usuário.
  * `estrategias_base` *(list[TipoEstrategiaBase])*: Enumerações das estratégias a testar.
  * `meta` *(float | None)*: Patrimônio alvo para cálculo de probabilidade de sucesso.
* **Retornos (Outputs):**
  * **[0] `ResultadoComparador`**: Tabela/Matriz contendo Q1, Mediana, Q3, Probabilidade de Perda e Probabilidade de Meta para cada estratégia mapeada.
  * **[1] `Simulacao`**: Retornos cumulativos empacotados.

### 3.3. `desacumulacao`
**Objetivo:** Simula a fase de usufruto/aposentadoria. Calcula a probabilidade de ruína (zerar o patrimônio) dado um fluxo de saques programados.

* **Entradas (Inputs):**
  * `capitalTotal` *(float)*.
  * `saque` *(float)*: Valor financeiro a ser retirado por ciclo.
  * `frequenciaSaque` *(FrequenciaAporte)*: Periodicidade do saque.
  * `fracao_rv` *(float)*: Percentual do capital exposto à bolsa [0, 1].
  * `propocaoAcao` *(np.ndarray)*: Pesos internos da bolsa.
  * `tickers`, `params`, `rf`, `diasInvestimento`.
  * `limite_ruina` *(float, padrão=0.0)*: Tolerância aceitável de ruína. Usado na busca pela taxa segura máxima.
  * `percentis_duracao` *(list[int])*: Quantis de tempo de sobrevida.
* **Retornos (Outputs):**
  * **[0] `ResultadoDesacumulacao`**: Contém a probabilidade exata de ruína, o saque máximo sustentável e a duração do patrimônio em anos/dias úteis.

### 3.4. `fronteira`
**Objetivo:** Gera a curva da Fronteira Eficiente baseada na minimização do Risco de Cauda (CVaR).

* **Entradas (Inputs):**
  * `params`, `tickers`, `rf`, `diasInvest`.
  * `confianca` *(float)*: Limite de confiança da cauda (ex: 0.95).
  * `numPontos` *(int, padrão=5)*: Resolução da curva.
  * `bounds` *(BoundsAtivo | None)*: Limites mínimos e máximos permitidos de alocação por ativo.
* **Retornos (Outputs):**
  * **[0] `FronteiraEficiente`**: Array de objetos `PontoFronteira`, contendo risco, retorno médio e os pesos de cada coordenada (X=Risco, Y=Retorno).

### 3.5. `meta` (Meta de Patrimônio Simples)
**Objetivo:** Descobre através de busca binária a exposição exata à Renda Variável necessária para atingir uma meta financeira com uma probabilidade informada.

* **Entradas (Inputs):**
  * `capitalTotal` *(float)*.
  * `meta` *(float)*: Patrimônio alvo desejado (R$).
  * `probabilidade` *(float)*: Certeza mínima exigida (ex: 0.80).
  * `proporcaoAcao`, `params`, `rf`, `diasInvestimento`, `capitalAportes`.
* **Retornos (Outputs):**
  * **[0] `ResultadoMeta`**: Separação financeira (R$) exigida entre RF e RV para otimizar o cenário, e a flag booleana `atingivel`.
  * **[1] `Simulacao`**: Objeto de cache.

### 3.6. `duploObjetivo`
**Objetivo:** Otimização complexa. Encontra a fronteira de Pareto (Trade-off) entre duas restrições: Um piso de segurança (não perder capital X) e Uma meta de crescimento (atingir capital Y).

* **Entradas (Inputs):**
  * `capitalTotal` *(float)*.
  * `piso` *(RestricaoPiso)*: Objeto contendo `valor` (R$) e `confianca` (%).
  * `meta` *(RestricaoMeta)*: Objeto contendo `valor` (R$) e `confianca` (%).
  * `proporcaoAcao`, `tickers`, `params`, `rf`, `diasInvestimento`.
* **Retornos (Outputs):**
  * **[0] `ResultadoDuploObjetivo`**: Modelagem conservadora (Ponto Mínimo), modelagem agressiva (Ponto Máximo) e a lista de pontos intermediários (`fronteira`) mesclando os níveis de probabilidade.
  * **[1] `Simulacao`**: Objeto de cache.

### 3.7. `alocacaoOtimizada`
**Objetivo:** Retorna os pesos ótimos dos ativos da carteira (RV isolada) minimizando estritamente a Perda Esperada Condicional (CVaR) via algoritmo Nelder-Mead.

* **Entradas (Inputs):**
  * `params`, `diasInvestimento`, `confianca`, `numSimulacoes`, `diasRebalanceamento`.
  * `poupaTempo` *(bool)*: Reduz rigor estatístico para devolução rápida na UI.
* **Retornos (Outputs):**
  * **[0] `np.ndarray`**: Array float de pesos ideais padronizados (ex: `[0.45, 0.30, 0.25]`).

### 3.8. `tempoMeta`
**Objetivo:** Simula o avanço do patrimônio no tempo e calcula em quantos dias/anos o capital cruzará a linha de chegada estipulada pela `meta`.

* **Entradas (Inputs):**
  * `capitalTotal` *(float)*, `meta` *(float)*.
  * `fracao_rv` *(float)*: Exposição total em RV [0, 1].
  * `proporcaoAcao`, `tickers`, `params`, `rf`, `diasInvestimento`, `valorAporte`, `frequenciaAporte`.
  * `percentis` *(list[int])*: Quantis de tempo (ex: `[10, 50, 90]`).
* **Retornos (Outputs):**
  * **[0] `ResultadoTempoMeta`**: Probabilidade de sucesso no horizonte máximo, e dicionários mapeando os anos e dias úteis para cada percentil.

---

## Dicas de Implementação (Front-End & API)

1. **Imutabilidade Matemática da RF:** O back-end foi unificado para capitalização diária contínua. O objeto `ParametrosRF` deve ser gerado pelo método `preparar_parametros_rf`. A Interface/UI não deve tentar calcular juros compostos ou ciclos de pagamento por conta própria, mas sim repassar o valor de aporte e confiar nos cálculos de `_valor_futuro_aportes` do motor.
2. **Cache de Simulações (State Management):** A variável `Simulacao` retornada (que encapsula o array `retornosCumulativos`) existe porque Monte Carlo em Python custa CPU. Se o usuário transitar da aba de *Alocação* para a aba de *Duplo Objetivo* sem alterar os ativos, o front-end **deve** devolver o array preexistente de retornos no payload. O back-end utilizará o parâmetro opcional `retornosCumulativos` (onde suportado) para bypassar o Monte Carlo e entregar respostas da UI em milissegundos.