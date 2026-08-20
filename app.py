import os
import heapq
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Tradesphere",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Data and technical indicators
# -----------------------------
@st.cache_data(ttl=300)
def load_market_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(column).title() for column in data.columns]
    return data.dropna()


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]

    result["SMA_20"] = close.rolling(20).mean()
    result["SMA_50"] = close.rolling(50).mean()
    result["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + relative_strength))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema_12 - ema_26
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()

    middle_band = close.rolling(20).mean()
    standard_deviation = close.rolling(20).std()
    result["BB_Upper"] = middle_band + 2 * standard_deviation
    result["BB_Lower"] = middle_band - 2 * standard_deviation

    return result


# -----------------------------
# Monotonic stack
# -----------------------------
def find_support_resistance(
    data: pd.DataFrame,
    window: int = 3,
) -> tuple[list[float], list[float]]:
    lows = data["Low"].to_numpy()
    highs = data["High"].to_numpy()

    support_stack: list[tuple[int, float]] = []
    resistance_stack: list[tuple[int, float]] = []

    supports: list[float] = []
    resistances: list[float] = []

    for index in range(window, len(data) - window):
        local_lows = lows[index - window:index + window + 1]
        local_highs = highs[index - window:index + window + 1]

        if lows[index] == np.min(local_lows):
            while support_stack and support_stack[-1][1] >= lows[index]:
                support_stack.pop()
            support_stack.append((index, float(lows[index])))

        if highs[index] == np.max(local_highs):
            while resistance_stack and resistance_stack[-1][1] <= highs[index]:
                resistance_stack.pop()
            resistance_stack.append((index, float(highs[index])))

    supports = [price for _, price in support_stack[-5:]]
    resistances = [price for _, price in resistance_stack[-5:]]

    return supports, resistances


# -----------------------------
# Priority queue alerts
# -----------------------------
def build_alert_queue(data: pd.DataFrame) -> list[tuple[int, str]]:
    if len(data) < 2:
        return []

    latest = data.iloc[-1]
    previous = data.iloc[-2]
    alerts: list[tuple[int, str]] = []

    if latest["RSI"] < 30:
        heapq.heappush(alerts, (1, "RSI indicates an oversold condition"))
    elif latest["RSI"] > 70:
        heapq.heappush(alerts, (1, "RSI indicates an overbought condition"))

    if latest["MACD"] > latest["MACD_Signal"] and previous["MACD"] <= previous["MACD_Signal"]:
        heapq.heappush(alerts, (2, "MACD bullish crossover detected"))

    if latest["MACD"] < latest["MACD_Signal"] and previous["MACD"] >= previous["MACD_Signal"]:
        heapq.heappush(alerts, (2, "MACD bearish crossover detected"))

    if latest["Close"] > latest["BB_Upper"]:
        heapq.heappush(alerts, (3, "Price is above the upper Bollinger Band"))

    if latest["Close"] < latest["BB_Lower"]:
        heapq.heappush(alerts, (3, "Price is below the lower Bollinger Band"))

    if latest["SMA_20"] > latest["SMA_50"]:
        heapq.heappush(alerts, (4, "Short-term trend is above the long-term trend"))
    else:
        heapq.heappush(alerts, (4, "Short-term trend is below the long-term trend"))

    return alerts


# -----------------------------
# News and sentiment
# -----------------------------
def get_news_sentiment(ticker: str) -> dict[str, Any]:
    news_api_key = os.getenv("NEWS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not news_api_key:
        return {
            "score": 0.0,
            "label": "Unavailable",
            "summary": "Set NEWS_API_KEY to enable news sentiment analysis.",
            "articles": [],
        }

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": news_api_key,
        },
        timeout=15,
    )
    response.raise_for_status()

    articles = response.json().get("articles", [])
    headlines = [article.get("title", "") for article in articles if article.get("title")]

    if not headlines:
        return {
            "score": 0.0,
            "label": "Neutral",
            "summary": "No recent news was found.",
            "articles": [],
        }

    if openai_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)
            prompt = (
                f"Analyze these headlines for {ticker}. Return a short summary and "
                "a sentiment score from -1 to 1.\n\n"
                + "\n".join(f"- {headline}" for headline in headlines)
            )

            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cautious financial news sentiment analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            summary = completion.choices[0].message.content
        except Exception as error:
            summary = f"LLM sentiment unavailable: {error}"
    else:
        positive_words = ("growth", "beat", "surge", "profit", "upgrade", "strong")
        negative_words = ("loss", "fall", "drop", " downgrade", "weak", "lawsuit")

        positive_count = sum(
            any(word in headline.lower() for word in positive_words)
            for headline in headlines
        )
        negative_count = sum(
            any(word in headline.lower() for word in negative_words)
            for headline in headlines
        )

        score = (positive_count - negative_count) / max(len(headlines), 1)
        label = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"

        return {
            "score": round(score, 2),
            "label": label,
            "summary": f"Rule-based sentiment: {label}. Add OPENAI_API_KEY for LLM analysis.",
            "articles": articles,
        }

    return {
        "score": 0.0,
        "label": "LLM analyzed",
        "summary": summary,
        "articles": articles,
    }


# -----------------------------
# Stripe payment interface
# -----------------------------
def create_checkout_url() -> str | None:
    stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
    stripe_price_id = os.getenv("STRIPE_PRICE_ID")

    if not stripe_secret_key or not stripe_price_id:
        return None

    import stripe

    stripe.api_key = stripe_secret_key

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": stripe_price_id, "quantity": 1}],
        success_url="http://localhost:8501/?payment=success",
        cancel_url="http://localhost:8501/?payment=cancelled",
    )
    return session.url


# -----------------------------
# Streamlit interface
# -----------------------------
st.title("📈 AI Stock & Trading Analytics")
st.caption("Educational analytics only — not financial advice.")

with st.sidebar:
    st.header("Market Settings")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    period = st.selectbox("Historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("Interval", ["1d", "1h", "1wk"], index=0)
    run_analysis = st.button("Run Analysis", type="primary")

    st.divider()
    st.header("Premium Access")

    if st.button("Subscribe with Stripe"):
        try:
            checkout_url = create_checkout_url()

            if checkout_url:
                st.markdown(f"[Continue to secure checkout]({checkout_url})")
            else:
                st.warning(
                    "Configure STRIPE_SECRET_KEY and STRIPE_PRICE_ID "
                    "to enable live checkout."
                )
        except Exception as error:
            st.error(f"Payment setup error: {error}")

if run_analysis or "market_data" not in st.session_state:
    with st.spinner("Loading market data..."):
        market_data = load_market_data(ticker, period, interval)

    if market_data.empty:
        st.error("No market data found. Check the ticker and selected interval.")
        st.stop()

    st.session_state.market_data = calculate_indicators(market_data)
    st.session_state.analysis_ticker = ticker

data = st.session_state.market_data
active_ticker = st.session_state.analysis_ticker

latest = data.iloc[-1]
previous_close = data["Close"].iloc[-2] if len(data) > 1 else latest["Close"]
price_change = latest["Close"] - previous_close
percent_change = price_change / previous_close * 100

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Latest Price", f"${latest['Close']:.2f}", f"{percent_change:.2f}%")
metric_2.metric("RSI", f"{latest['RSI']:.2f}")
metric_3.metric("MACD", f"{latest['MACD']:.2f}")
metric_4.metric("Volume", f"{latest['Volume']:,.0f}")

st.subheader(f"{active_ticker} Price Chart")

figure = go.Figure()
figure.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price",
    )
)
figure.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20"))
figure.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50"))
figure.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper"))
figure.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower"))

figure.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
)
st.plotly_chart(figure, use_container_width=True)

support_levels, resistance_levels = find_support_resistance(data)
alerts = build_alert_queue(data)

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Support and Resistance")
    st.write("Support:", [round(level, 2) for level in support_levels])
    st.write("Resistance:", [round(level, 2) for level in resistance_levels])

with right_column:
    st.subheader("Priority Alerts")
    if alerts:
        for priority, alert in alerts:
            st.write(f"Priority {priority}: {alert}")
    else:
        st.info("No alerts detected.")

st.subheader("Technical Indicator Panels")

indicator_figure = go.Figure()
indicator_figure.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
indicator_figure.add_hline(y=70, line_dash="dash", line_color="red")
indicator_figure.add_hline(y=30, line_dash="dash", line_color="green")
indicator_figure.update_layout(height=300, yaxis_title="RSI", template="plotly_dark")
st.plotly_chart(indicator_figure, use_container_width=True)

st.subheader("AI and News Sentiment")

if st.button("Analyze News Sentiment"):
    try:
        with st.spinner("Analyzing recent news..."):
            sentiment = get_news_sentiment(active_ticker)

        sentiment_col_1, sentiment_col_2 = st.columns(2)
        sentiment_col_1.metric("Sentiment Score", sentiment["score"])
        sentiment_col_2.metric("Sentiment Label", sentiment["label"])
        st.info(sentiment["summary"])

        for article in sentiment["articles"]:
            title = article.get("title", "Untitled article")
            url = article.get("url", "#")
            st.markdown(f"- [{title}]({url})")
    except Exception as error:
        st.error(f"Sentiment request failed: {error}")

st.divider()
st.caption(
    "Data source: Yahoo Finance. Configure NEWS_API_KEY, OPENAI_API_KEY, "
    "STRIPE_SECRET_KEY, and STRIPE_PRICE_ID for optional integrations."
)
