import heapq
import os
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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #243047;
    }

    .hero {
        padding: 1.5rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #172554, #0f766e);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #dbeafe;
        margin: 0;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1rem 0 0.7rem;
        color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #243047;
        padding: 1rem;
        border-radius: 14px;
    }

    .disclaimer {
        color: #94a3b8;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_config(name: str, default: str = "") -> str:
    """Read configuration from Streamlit Secrets or environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


@st.cache_data(ttl=300)
def load_market_data(
    ticker: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
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
    result["BB_Middle"] = middle_band
    result["BB_Upper"] = middle_band + 2 * standard_deviation
    result["BB_Lower"] = middle_band - 2 * standard_deviation

    return result


def find_support_resistance(
    data: pd.DataFrame,
    window: int = 3,
) -> tuple[list[float], list[float]]:
    if len(data) < (window * 2 + 1):
        return [], []

    lows = data["Low"].to_numpy()
    highs = data["High"].to_numpy()

    support_stack: list[tuple[int, float]] = []
    resistance_stack: list[tuple[int, float]] = []

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


def build_alert_queue(data: pd.DataFrame) -> list[tuple[int, str]]:
    if len(data) < 2:
        return []

    latest = data.iloc[-1]
    previous = data.iloc[-2]
    alerts: list[tuple[int, str]] = []

    if pd.notna(latest["RSI"]):
        if latest["RSI"] < 30:
            heapq.heappush(alerts, (1, "RSI indicates an oversold condition"))
        elif latest["RSI"] > 70:
            heapq.heappush(alerts, (1, "RSI indicates an overbought condition"))

    if (
        pd.notna(latest["MACD"])
        and pd.notna(latest["MACD_Signal"])
        and latest["MACD"] > latest["MACD_Signal"]
        and previous["MACD"] <= previous["MACD_Signal"]
    ):
        heapq.heappush(alerts, (2, "MACD bullish crossover detected"))

    if (
        pd.notna(latest["MACD"])
        and pd.notna(latest["MACD_Signal"])
        and latest["MACD"] < latest["MACD_Signal"]
        and previous["MACD"] >= previous["MACD_Signal"]
    ):
        heapq.heappush(alerts, (2, "MACD bearish crossover detected"))

    if pd.notna(latest["BB_Upper"]) and latest["Close"] > latest["BB_Upper"]:
        heapq.heappush(alerts, (3, "Price is above the upper Bollinger Band"))

    if pd.notna(latest["BB_Lower"]) and latest["Close"] < latest["BB_Lower"]:
        heapq.heappush(alerts, (3, "Price is below the lower Bollinger Band"))

    if pd.notna(latest["SMA_20"]) and pd.notna(latest["SMA_50"]):
        message = (
            "Short-term trend is above the long-term trend"
            if latest["SMA_20"] > latest["SMA_50"]
            else "Short-term trend is below the long-term trend"
        )
        heapq.heappush(alerts, (4, message))

    return alerts


def get_news_sentiment(ticker: str) -> dict[str, Any]:
    news_api_key = get_config("NEWS_API_KEY")
    openai_api_key = get_config("OPENAI_API_KEY")
    openai_model = get_config("OPENAI_MODEL", "gpt-4o-mini")

    if not news_api_key or news_api_key == "your-news-api-key":
        return {
            "score": 0.0,
            "label": "Unavailable",
            "summary": "Add a real NEWS_API_KEY to enable news sentiment.",
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
    headlines = [
        article.get("title", "")
        for article in articles
        if article.get("title")
    ]

    if not headlines:
        return {
            "score": 0.0,
            "label": "Neutral",
            "summary": "No recent news was found.",
            "articles": [],
        }

    if openai_api_key and openai_api_key != "your-openai-api-key":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)
            prompt = (
                f"Analyze these headlines for {ticker}. Return a concise summary "
                "and classify the sentiment as Positive, Neutral, or Negative.\n\n"
                + "\n".join(f"- {headline}" for headline in headlines)
            )

            completion = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cautious financial news analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            summary = completion.choices[0].message.content
            return {
                "score": 0.0,
                "label": "AI analyzed",
                "summary": summary,
                "articles": articles,
            }
        except Exception as error:
            st.warning(f"AI analysis unavailable: {error}")

    positive_words = (
        "growth",
        "beat",
        "surge",
        "profit",
        "upgrade",
        "strong",
        "gain",
    )
    negative_words = (
        "loss",
        "fall",
        "drop",
        "downgrade",
        "weak",
        "lawsuit",
        "decline",
    )

    positive_count = sum(
        any(word in headline.lower() for word in positive_words)
        for headline in headlines
    )
    negative_count = sum(
        any(word in headline.lower() for word in negative_words)
        for headline in headlines
    )

    score = (positive_count - negative_count) / max(len(headlines), 1)
    label = (
        "Positive"
        if score > 0.15
        else "Negative"
        if score < -0.15
        else "Neutral"
    )

    return {
        "score": round(score, 2),
        "label": label,
        "summary": (
            f"Rule-based sentiment: {label}. "
            "Add OPENAI_API_KEY for AI analysis."
        ),
        "articles": articles,
    }


def create_checkout_url() -> str | None:
    stripe_secret_key = get_config("STRIPE_SECRET_KEY")
    stripe_price_id = get_config("STRIPE_PRICE_ID")
    app_url = get_config("APP_URL", "http://localhost:8501")

    if (
        not stripe_secret_key
        or stripe_secret_key == "your-stripe-secret-key"
        or not stripe_price_id
        or stripe_price_id == "your-stripe-price-id"
    ):
        return None

    import stripe

    stripe.api_key = stripe_secret_key

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": stripe_price_id, "quantity": 1}],
        success_url=f"{app_url}/?payment=success",
        cancel_url=f"{app_url}/?payment=cancelled",
    )

    return session.url


def create_price_chart(data: pd.DataFrame) -> go.Figure:
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

    for column, color in [
        ("SMA_20", "#38bdf8"),
        ("SMA_50", "#f59e0b"),
        ("BB_Upper", "#a78bfa"),
        ("BB_Lower", "#a78bfa"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                name=column.replace("_", " "),
                line={"color": color, "width": 1.5},
            )
        )

    figure.update_layout(
        height=560,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.02},
    )

    return figure


def create_indicator_chart(data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line={"color": "#22c55e", "width": 2},
        )
    )
    figure.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    figure.add_hline(y=30, line_dash="dash", line_color="#38bdf8")

    figure.update_layout(
        height=280,
        template="plotly_dark",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )

    return figure


# Sidebar
with st.sidebar:
    st.markdown("## 📈 Tradesphere")
    st.caption("Market intelligence dashboard")

    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    period = st.selectbox(
        "Historical period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )
    interval = st.selectbox(
        "Data interval",
        ["1d", "1h", "1wk"],
        index=0,
    )

    run_analysis = st.button(
        "🚀 Run analysis",
        type="primary",
        use_container_width=True,
    )

    if st.button("🔄 Clear cached data", use_container_width=True):
        load_market_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### Premium Access")

    if st.button("Subscribe with Stripe", use_container_width=True):
        try:
            checkout_url = create_checkout_url()

            if checkout_url:
                st.markdown(f"[Continue to secure checkout]({checkout_url})")
            else:
                st.warning(
                    "Add real STRIPE_SECRET_KEY and STRIPE_PRICE_ID "
                    "values to enable checkout."
                )
        except Exception as error:
            st.error(f"Payment setup error: {error}")


# Load data
if run_analysis or "market_data" not in st.session_state:
    with st.spinner(f"Loading {ticker} market data..."):
        raw_data = load_market_data(ticker, period, interval)

    if raw_data.empty:
        st.error("No market data found. Check the ticker and selected interval.")
        st.stop()

    st.session_state.market_data = calculate_indicators(raw_data)
    st.session_state.analysis_ticker = ticker

data = st.session_state.market_data
active_ticker = st.session_state.analysis_ticker
latest = data.iloc[-1]

previous_close = (
    data["Close"].iloc[-2]
    if len(data) > 1
    else latest["Close"]
)
price_change = latest["Close"] - previous_close
percent_change = (price_change / previous_close) * 100


# Header
st.markdown(
    f"""
    <div class="hero">
        <h1>📈 Tradesphere</h1>
        <p>{active_ticker} market analytics, technical signals, and news insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Educational analytics only — not financial advice.")

# KPI cards
metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Latest Price",
    f"${latest['Close']:.2f}",
    f"{percent_change:.2f}%",
)

metric_2.metric(
    "RSI",
    "N/A" if pd.isna(latest["RSI"]) else f"{latest['RSI']:.2f}",
)

metric_3.metric(
    "MACD",
    "N/A" if pd.isna(latest["MACD"]) else f"{latest['MACD']:.2f}",
)

metric_4.metric(
    "Volume",
    f"{latest['Volume']:,.0f}",
)

# Tabs
overview_tab, technical_tab, sentiment_tab = st.tabs(
    ["📊 Overview", "🧭 Technical Signals", "📰 News Sentiment"]
)

with overview_tab:
    st.markdown('<div class="section-title">Price Overview</div>', unsafe_allow_html=True)
    st.plotly_chart(
        create_price_chart(data),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    left_column, right_column = st.columns(2)

    support_levels, resistance_levels = find_support_resistance(data)

    with left_column:
        st.markdown(
            '<div class="section-title">Support and Resistance</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Support levels:",
            [round(level, 2) for level in support_levels] or "Not enough data",
        )
        st.write(
            "Resistance levels:",
            [round(level, 2) for level in resistance_levels]
            or "Not enough data",
        )

    with right_column:
        st.markdown(
            '<div class="section-title">Priority Alerts</div>',
            unsafe_allow_html=True,
        )

        alerts = build_alert_queue(data)

        if alerts:
            for priority, alert in alerts:
                if priority == 1:
                    st.error(f"Priority {priority}: {alert}")
                elif priority == 2:
                    st.warning(f"Priority {priority}: {alert}")
                else:
                    st.info(f"Priority {priority}: {alert}")
        else:
            st.success("No alerts detected.")

with technical_tab:
    st.markdown(
        '<div class="section-title">Relative Strength Index</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        create_indicator_chart(data),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        '<div class="section-title">Technical Data</div>',
        unsafe_allow_html=True,
    )

    technical_data = data[
        ["Close", "SMA_20", "SMA_50", "EMA_20", "RSI", "MACD", "MACD_Signal"]
    ].tail(20).sort_index(ascending=False)

    st.dataframe(
        technical_data.round(2),
        use_container_width=True,
        height=420,
    )

with sentiment_tab:
    st.markdown(
        '<div class="section-title">AI and News Sentiment</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔎 Analyze news sentiment", type="primary"):
        try:
            with st.spinner("Analyzing recent news..."):
                sentiment = get_news_sentiment(active_ticker)

            sentiment_col_1, sentiment_col_2 = st.columns(2)
            sentiment_col_1.metric(
                "Sentiment Score",
                sentiment["score"],
            )
            sentiment_col_2.metric(
                "Sentiment Label",
                sentiment["label"],
            )

            st.info(sentiment["summary"])

            if sentiment["articles"]:
                st.markdown("### Latest headlines")

                for article in sentiment["articles"]:
                    title = article.get("title", "Untitled article")
                    url = article.get("url", "#")
                    source = article.get("source", {}).get("name", "Unknown source")
                    st.markdown(f"- [{title}]({url}) — *{source}*")
            else:
                st.info("No articles available.")
        except Exception as error:
            st.error(f"Sentiment request failed: {error}")
    else:
        st.info("Click the button to analyze recent news for this ticker.")


st.markdown(
    """
    <div class="disclaimer">
        Data source: Yahoo Finance. This application is for educational purposes only.
        Market data may be delayed and should not be considered financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
