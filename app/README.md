# Interface Streamlit

Front-end da Engine Quantitativa de Portfólio. Expõe as 8 funcionalidades da
camada de integração (`interface.py`) em uma UI web local.

## Como rodar

A partir da **raiz do repositório**:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-app.txt
.venv/bin/streamlit run app/main.py
```

A interface abre em ` `.

## Fluxo de uso

1. **Montar a carteira** na própria página inicial:
   - Selecione os **ativos** no campo de busca (tickers populares da B3 já
     listados; é possível digitar qualquer ticker do yfinance).
   - Defina os **pesos** da Renda Variável (botão "Pesos iguais" disponível; a
     soma é normalizada para 100%).
   - Ajuste **capital, prazo, taxa da Renda Fixa, qualidade** e, em *Avançado*,
     o **rebalanceamento**.
2. Clique em **Calibrar carteira** — passo caro (download + ajuste estatístico),
   roda **uma vez** e fica guardado na sessão.
3. **Navegar pelas funcionalidades** pelo seletor no topo. A faixa da carteira
   mostra os ativos, permite trocar a **qualidade** e tem **Editar carteira**
   para recalibrar.

## Cache de simulações (otimização de CPU)

Monte Carlo em Python custa CPU, então a interface reaproveita simulações:

- **Alocação, Comparador, Meta e Duplo objetivo** compartilham o **mesmo** array
  de retornos cumulativos (objeto `Simulacao`). Enquanto **pesos da RV, prazo,
  rebalanceamento e qualidade** não mudarem, a segunda funcionalidade em diante
  sai em milissegundos — a UI exibe o badge "Em cache".
- Mudar qualquer um desses parâmetros (ou recalibrar) invalida o cache e a
  próxima execução roda um novo Monte Carlo (a UI avisa antes).
- **Fronteira, Otimizar carteira, Desacumulação e Tempo para meta** rodam
  simulações próprias internas a cada cálculo (não usam esse cache).

O seletor de **Qualidade da simulação** (na faixa da carteira) ajusta o nº de
cenários — mais cenários = mais preciso e mais lento.

> **Fronteira eficiente** é a funcionalidade mais pesada: roda uma otimização
> CVaR por ponto. Use o **Modo rápido** (ligado por padrão) e poucos pontos.

## Tema

Visual dark inspirado no shadcn/ui (paleta zinc). O tema base fica em
`.streamlit/config.toml` e os ajustes finos (badges, callouts, navegação como
abas, tags do multiselect) em `app/tema.py`.

## Estrutura

```
app/
  main.py          # entrada: setup na página, faixa da carteira e navegação
  tema.py          # tema dark (CSS) + badges e callouts shadcn-like
  estado.py        # session_state, calibração e cache de simulação
  componentes.py   # widgets reutilizáveis (status de cache, aporte, etc.)
  formatacao.py    # R$, %, anos/dias
  graficos.py      # histograma, fronteira, Pareto, barras (matplotlib)
  paginas/         # uma página por funcionalidade
```
