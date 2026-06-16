# Autores
Bernardo Vale dos Santos Bento
Gustavo Henrique Silva Paiva

# Motor de Simulação e Otimização Quantitativa de Portfólios

Bem-vindo ao repositório do nosso Motor Quantitativo de Portfólios. Este sistema foi desenhado para modelar, simular e otimizar carteiras de investimento compostas por Renda Fixa (RF) e Renda Variável (RV), indo muito além das premissas tradicionais da Teoria Moderna do Portfólio (Markowitz).

Enquanto a maioria das ferramentas de mercado assume que os retornos das ações seguem uma distribuição Normal (Curva de Sino), este motor foi construído para lidar com a realidade do mercado financeiro: **assimetria, eventos extremos (cisnes negros) e correlações que convergem para 1 durante crises.**

---

## 🔬 O Motor Matemático: Por que somos diferentes?

Para garantir projeções e testes de estresse realistas, a arquitetura estatística deste projeto utiliza métodos quantitativos de ponta:

1. **Distribuições t-Student Marginais:** Em vez de usar a Volatilidade (Desvio Padrão) clássica, calibramos cada ativo em uma distribuição *t-Student*. Isso permite capturar as famosas "caudas pesadas" (fat tails), ou seja, a probabilidade real de quedas bruscas no mercado que a distribuição Normal subestima.
2. **Cópulas-t Multivariadas:** Diversificação falha quando você mais precisa dela. Em crashes de mercado, os ativos tendem a cair juntos. Utilizamos Cópulas-t para modelar a *dependência de cauda*, garantindo que nossos cenários de estresse reflitam crises sistêmicas reais.
3. **Otimização por CVaR (Expected Shortfall):** Não otimizamos carteiras olhando para a variância (que penaliza retornos positivos da mesma forma que os negativos). Nosso otimizador minimiza o **CVaR** (Conditional Value at Risk), focando estritamente em proteger o capital nos piores cenários (ex: a média dos 5% piores cenários simulados).
4. **Capitalização Contínua em Renda Fixa:** Tratamos a Renda Fixa com precisão matemática diária. Os aportes são capitalizados dia a dia (considerando dias úteis), eliminando distorções de "dinheiro parado" entre ciclos de pagamento.

---

## 📊 Funcionalidades para Gestão de Patrimônio (Wealth Management)

O motor está dividido em módulos focados em responder às perguntas mais críticas do planejamento financeiro:

### 1. Fronteira Eficiente CVaR
Calcula a curva de portfólios ótimos. Para cada nível de retorno esperado, o algoritmo encontra a exata proporção entre Renda Fixa e ativos de Renda Variável que **minimiza o Risco de Cauda (CVaR)**. 

### 2. Duplo Objetivo (Proteção vs. Crescimento)
O "Santo Graal" da alocação baseada em objetivos (Goal-based Investing). Encontra o *Trade-off* perfeito (Fronteira de Pareto) entre duas restrições concorrentes:
* **Piso (Floor):** "Quero ter 95% de certeza de que não perderei mais do que X% do meu capital."
* **Meta (Upside):** "Quero maximizar a probabilidade de atingir Y reais no fim do prazo."

### 3. Fase de Desacumulação (Aposentadoria / Usufruto)
Simula saques periódicos no portfólio. A Renda Fixa age como colchão de liquidez; se ela secar, o sistema simula a venda forçada de ações a mercado.
* **Outputs:** Calcula a probabilidade exata de ruína (zerar o dinheiro antes de morrer), a duração do patrimônio e descobre qual é a **Taxa de Saque Sustentável máxima**.

### 4. Tempo para Meta
Em vez de fixar um prazo de investimento, fixamos o patrimônio-alvo. O motor avança os dias úteis no simulador e gera um histograma mostrando: *"Há 90% de chance de você atingir R$ 1 Milhão entre 7,5 e 9,2 anos"*.

### 5. Comparador de Estratégias
Permite fazer o benchmarking (teste lado a lado) de estratégias estáticas. Compara uma alocação customizada do cliente (ex: 60% Ações, 40% Títulos) contra estratégias fixas (100% RF, 100% RV, 75/25, etc.), avaliando o Índice de Sharpe, Índice de Sortino (risco de *downside*), e percentis de patrimônio final.

---

## 💻 Guia de Instalação e Uso (Para Não-Programadores)

Para interagir com o motor, construímos uma interface visual em seu navegador. Siga os passos abaixo para preparar o ambiente e rodar a plataforma no seu computador.

### Pré-requisitos
* Ter o [Python](https://www.python.org/downloads/) (versão 3.10 ou superior) instalado no seu computador. Ao instalar no Windows, **certifique-se de marcar a caixa "Add Python to PATH"**.

### Passo 1: Preparar a Pasta do Projeto
Abra o seu Terminal (no Mac/Linux) ou o Prompt de Comando / PowerShell (no Windows) e navegue até a pasta onde os arquivos do projeto estão salvos.

### Passo 2: Criar um Ambiente Virtual
Um ambiente virtual é como uma "caixa isolada" para o projeto não interferir nos outros programas do seu computador.

* **No Windows:**

python -m venv venv
venv\Scripts\activate

* **No Mac ou Linux:**

python3 -m venv venv
source venv/bin/activate

*(Nota: Você saberá que deu certo se o nome `(venv)` aparecer no início da linha do seu terminal).*

### Passo 3: Instalar as Dependências
Com o ambiente ativado, instale as bibliotecas financeiras e matemáticas (Pandas, Numpy, Scipy, YFinance, Streamlit) que o motor precisa para rodar:

pip install -r requirements.txt

*(Esse processo pode levar alguns minutos, pois ele fará o download da base matemática do projeto).*

### Passo 4: Iniciar a Plataforma
Agora é só "ligar" o motor e iniciar a interface gráfica construída pela nossa equipe de Front-end:

streamlit run app/main.py

O seu navegador padrão abrirá automaticamente na página da plataforma. Caso não abra, basta digitar na barra de endereços do seu navegador:
👉 **`http://localhost:8501`**

Para desligar o sistema depois de usar, basta voltar à tela do terminal e pressionar `Ctrl + C`.