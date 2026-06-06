import pandas as pd
import yfinance as yf

def baixar_retornos(tickers: list[str], periodo: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Baixa histórico de preços via yfinance e calcula retornos diários.

    Remove tickers sem dados suficientes (< 30 observações) e avisa o usuário.
    Retorna o DataFrame de retornos e a lista de tickers válidos.
    """
    print("Baixando e validando tickers...")
    precos = yf.download(tickers, period=periodo, auto_adjust=True, progress=False)["Close"]

    # yfinance retorna Series quando há apenas um ticker — força DataFrame
    if isinstance(precos, pd.Series):
        precos = precos.to_frame(name=tickers[0])

    # Filtra tickers com dados insuficientes para calibração confiável
    validos   = [t for t in tickers if t in precos.columns and precos[t].notna().sum() > 30]
    invalidos = set(tickers) - set(validos)

    if invalidos:
        print(f"  ⚠ Tickers ignorados (sem dados suficientes): {invalidos}")
    if not validos:
        raise ValueError("Nenhum ticker válido encontrado. Verifique os símbolos informados.")

    retornos = precos[validos].pct_change().dropna()
    return retornos, validos