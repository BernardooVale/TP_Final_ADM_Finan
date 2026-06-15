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

A interface abre em `http://localhost:8501`.

## Fluxo de uso

1. **Calibrar carteira** (barra lateral): informe tickers (yfinance), pesos da
   RV, período histórico, capital, prazo e a taxa da Renda Fixa. Clique em
   **Calibrar carteira**. Esse é o passo caro (download + ajuste estatístico) e
   roda **uma única vez** — fica guardado na sessão.
2. **Navegar pelas funcionalidades** pelo seletor no topo.

## Cache de simulações (otimização de CPU)

Monte Carlo em Python custa CPU, então a interface reaproveita simulações:

- **Alocação, Comparador, Meta e Duplo objetivo** compartilham o **mesmo** array
  de retornos cumulativos (objeto `Simulacao`). Enquanto **pesos da RV, prazo,
  rebalanceamento e qualidade** não mudarem, a segunda funcionalidade em diante
  sai em milissegundos — a UI mostra "♻️ Simulação em cache".
- Mudar qualquer um desses parâmetros (ou recalibrar) invalida o cache e a
  próxima execução roda um novo Monte Carlo (a UI avisa antes).
- **Fronteira, Otimizar carteira, Desacumulação e Tempo para meta** rodam
  simulações próprias internas a cada cálculo (não usam esse cache).

O controle de **Qualidade da simulação** (barra lateral) ajusta o nº de
cenários — mais cenários = mais preciso e mais lento.

> ⚠️ **Fronteira eficiente** é a funcionalidade mais pesada: roda uma otimização
> CVaR por ponto. Use o **Modo rápido** (ligado por padrão) e poucos pontos.

## Estrutura

```
app/
  main.py          # entrada: setup na sidebar + navegação
  estado.py        # session_state, calibração e cache de simulação
  componentes.py   # widgets reutilizáveis (status de cache, aporte, etc.)
  formatacao.py    # R$, %, anos/dias
  graficos.py      # histograma, fronteira, Pareto, barras (matplotlib)
  paginas/         # uma página por funcionalidade
```
