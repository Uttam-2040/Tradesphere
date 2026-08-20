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
    page_title="Tradesphere | Market Intelligence",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Configuration and authentication
# ---------------------------------------------------------------------------

DEFAULT_USERNAME = "tsadmin"
DEFAULT_PASSWORD = "TS2026!"


def get_config(name: str, default: str = "") -> str:
    """Read a configuration value from Streamlit Secrets or environment."""
    try:
        value = st.secrets.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, default).strip()


def is_placeholder(value: str) -> bool:
    value = value.strip().lower()
    return (
        not value
        or value.startswith("your-")
        or value in {"change-this-password", "your-real-password"}
    )


def get_login_credentials() -> tuple[str, str]:
    username = get_config("AUTH_USERNAME", DEFAULT_USERNAME)
    password = get_config("AUTH_PASSWORD", DEFAULT_PASSWORD)

    if is_placeholder(username):
        username = DEFAULT_USERNAME

    if is_placeholder(password):
        password = DEFAULT_PASSWORD

    return username, password


def authenticate_user(username: str, password: str) -> bool:
    configured_username, configured_password = get_login_credentials()

    return (
        username.strip().lower() == configured_username.strip().lower()
        and password == configured_password
    )


def show_login_page() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, #172554 0%, transparent 32%),
                radial-gradient(circle at 85% 80%, #064e3b 0%, transparent 28%),
                #060b16;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: transparent;
        }

        .login-shell {
            max-width: 460px;
            margin: 7vh auto 20px auto;
            padding: 38px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 28px;
            background: rgba(15, 23, 42, 0.82);
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(18px);
            text-align: center;
        }

        .ts-logo {
            width: 82px;
            height: 82px;
            margin: 0 auto 18px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 24px;
            color: white;
            font-size: 30px;
            font-weight: 900;
            letter-spacing: -2px;
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            box-shadow: 0 12px 30px rgba(20, 184, 166, 0.28);
        }

        .login-title {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .login-subtitle {
            color: #94a3b8;
            margin-bottom: 25px;
        }

        .login-footer {
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 20px;
            text-align: center;
        }

        .demo-credentials {
            max-width: 460px;
            margin: 0 auto;
            padding: 14px 18px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 14px;
            background: rgba(14, 116, 144, 0.12);
            color: #bae6fd;
            text-align: center;
            font-size: 0.85rem;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }
        </style>

        <div class="login-shell">
            <div class="ts-logo">TS</div>
            <div class="login-title">Welcome to Tradesphere</div>
            <div class="login-subtitle">
                Intelligent market insights for modern investors
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    configured_username, configured_password = get_login_credentials()

    left, center, right = st.columns([1, 2, 1])

    with center:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                value="",
                placeholder="Enter username",
                autocomplete="username",
            )

            password = st.text_input(
                "Password",
                value="",
                type="password",
                placeholder="Enter password",
                autocomplete="current-password",
            )

            submitted = st.form_submit_button(
                "🔐 Access Dashboard",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username.strip()
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = (
                    "Invalid login details. Check the username and password below."
                )

        if st.session_state.get("login_error"):
            st.error(st.session_state.login_error)

        st.markdown(
            f"""
            <div class="demo-credentials">
                <strong>Login credentials</strong><br>
                Username: <code>{configured_username}</code><br>
                Password: <code>{configured_password}</code>
            </div>
            <div class="login-footer">
                TS Secure Access · Educational analytics platform
            </div>
            """,
            unsafe_allow_html=True,
        )


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "login_error" not in st.session_state:
    st.session_state.login_error = ""

if not st.session_state.authenticated:
    show_login_page()
    st.stop()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(30, 64, 175, 0.16), transparent 26%),
            radial-gradient(circle at 100% 100%, rgba(13, 148, 136, 0.12), transparent 25%),
            #070d19;
        color: #e2e8f0;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 13, 25, 0.88);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220, #0a1020);
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 25px;
    }

    .mini-logo {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        color: white;
        font-weight: 900;
        background: linear-gradient(135deg, #2563eb, #14b8a6);
    }

    .brand-name {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .brand-caption {
        color: #64748b;
        font-size: 0.72rem;
    }

    .eyebrow {
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .dashboard-title {
        color: #f8fafc;
        font-size: 2.15rem;
        font-weight: 850;
        letter-spacing: -0.05em;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
    }

    .hero {
        padding: 28px 30px;
        margin: 20px 0;
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 24px;
        background: linear-gradient(135deg, #1e40af, #0f766e);
        box-shadow: 0 18px 45px rgba(2, 6, 23, 0.35);
    }

    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin: 0 0 4px 0;
    }

    .hero p {
        color: #dbeafe;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        min-height: 122px;
        padding: 20px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.72);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 750;
        margin: 20px 0 10px 0;
    }

    .status-card {
        padding: 15px 18px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.7);
    }

    .disclaimer {
        color: #64748b;
        font-size: 0.78rem;
        text-align: center;
        margin: 32px 0 10px 0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Market data and indicators
# ---------------------------------------------------------------------------

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
    if len(data) < window * 2 + 1:
        return [], []

    lows = data["Low"].to_numpy()
    highs = data["High"].to_numpy()
    supports = []
    resistances = []

    for index in range(window, len(data) - window):
        if lows[index] == np.min(lows[index - window:index + window + 1]):
            supports.append(float(lows[index]))

        if highs[index] == np.max(highs[index - window:index + window + 1]):
            resistances.append(float(highs[index]))

    return supports[-5:], resistances[-5:]


def build_alert_queue(data: pd.DataFrame) -> list[tuple[int, str]]:
    if len(data) < 2:
        return []

    latest = data.iloc[-1]
    previous = data.iloc[-2]
    alerts = []

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


# ---------------------------------------------------------------------------
# Charts and optional integrations
# ---------------------------------------------------------------------------

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
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
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
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)",
        )
    )

    figure.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    figure.add_hline(y=30, line_dash="dash", line_color="#38bdf8")

    figure.update_layout(
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )

    return figure


def get_news_sentiment(ticker: str) -> dict[str, Any]:
    news_api_key = get_config("NEWS_API_KEY")
    openai_api_key = get_config("OPENAI_API_KEY")
    openai_model = get_config("OPENAI_MODEL", "gpt-4o-mini")

    if is_placeholder(news_api_key):
        return {
            "score": 0.0,
            "label": "Unavailable",
            "summary": "Configure a real NEWS_API_KEY to enable news sentiment.",
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

    if not is_placeholder(openai_api_key):
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

            return {
                "score": 0.0,
                "label": "AI analyzed",
                "summary": completion.choices[0].message.content,
                "articles": articles,
            }
        except Exception:
            pass

    positive_words = (
        "growth", "beat", "surge", "profit", "upgrade", "strong", "gain"
    )
    negative_words = (
        "loss", "fall", "drop", "downgrade", "weak", "lawsuit", "decline"
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
    label = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"

    return {
        "score": round(score, 2),
        "label": label,
        "summary": f"Rule-based sentiment: {label}.",
        "articles": articles,
    }


def create_checkout_url() -> str | None:
    secret_key = get_config("STRIPE_SECRET_KEY")
    price_id = get_config("STRIPE_PRICE_ID")
    app_url = get_config("APP_URL", "http://localhost:8501")

    if is_placeholder(secret_key) or is_placeholder(price_id):
        return None

    import stripe

    stripe.api_key = secret_key

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{app_url}/?payment=success",
        cancel_url=f"{app_url}/?payment=cancelled",
    )

    return session.url


# ---------------------------------------------------------------------------
# Sidebar and controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="mini-logo">TS</div>
            <div>
                <div class="brand-name">Tradesphere</div>
                <div class="brand-caption">Market intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Signed in as **{st.session_state.get('username', 'User')}**")

    ticker = st.text_input(
        "Ticker symbol",
        value="AAPL",
        help="Examples: AAPL, MSFT, TSLA, NVDA",
    ).upper().strip()

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

    if st.button("🔄 Refresh market data", use_container_width=True):
        load_market_data.clear()
        st.session_state.pop("market_data", None)
        st.rerun()

    st.divider()

    st.markdown("### ⚡ Premium Access")

    if st.button("Subscribe with Stripe", use_container_width=True):
        try:
            checkout_url = create_checkout_url()

            if checkout_url:
                st.markdown(f"[Continue to secure checkout]({checkout_url})")
            else:
                st.warning(
                    "Configure STRIPE_SECRET_KEY and STRIPE_PRICE_ID "
                    "to enable checkout."
                )
        except Exception as error:
            st.error(f"Payment setup error: {error}")

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.pop("market_data", None)
        st.session_state.pop("username", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Load market data
# ---------------------------------------------------------------------------

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

previous_close = data["Close"].iloc[-2] if len(data) > 1 else latest["Close"]
price_change = latest["Close"] - previous_close
percent_change = (price_change / previous_close) * 100


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

left_header, right_header = st.columns([4, 1])

with left_header:
    st.markdown('<div class="eyebrow">Live workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-title">Market Command Center</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-subtitle">Analytics, signals and sentiment in one place</div>',
        unsafe_allow_html=True,
    )

with right_header:
    st.markdown(
        '<div class="status-card">🟢 Data engine online<br><small>Yahoo Finance</small></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="hero">
        <h1>TS · {active_ticker}</h1>
        <p>Technical intelligence and market context for your selected asset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Educational analytics only — not financial advice.")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Latest Price",
    f"${latest['Close']:.2f}",
    f"{price_change:.2f} ({percent_change:.2f}%)",
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

overview_tab, technical_tab, sentiment_tab = st.tabs(
    ["📊 Overview", "🧭 Technical Signals", "📰 News Intelligence"]
)

with overview_tab:
    st.markdown('<div class="section-title">Price Overview</div>', unsafe_allow_html=True)

    st.plotly_chart(
        create_price_chart(data),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    support_levels, resistance_levels = find_support_resistance(data)
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(
            '<div class="section-title">Support and Resistance</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "🟢 Support levels:",
            [round(level, 2) for level in support_levels] or "Not enough data",
        )
        st.write(
            "🔴 Resistance levels:",
            [round(level, 2) for level in resistance_levels] or "Not enough data",
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

    technical_columns = [
        "Close",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI",
        "MACD",
        "MACD_Signal",
    ]

    st.dataframe(
        data[technical_columns].tail(20).sort_index(ascending=False).round(2),
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
            sentiment_col_1.metric("Sentiment Score", sentiment["score"])
            sentiment_col_2.metric("Sentiment Label", sentiment["label"])
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
        TS Tradesphere · Data source: Yahoo Finance · Market data may be delayed.
        This application is for educational purposes only and is not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
