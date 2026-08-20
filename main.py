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
    page_title="Tradesphere | AI Market Intelligence",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_USERNAME = "tsadmin"
DEFAULT_PASSWORD = "TS2026!"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_config(name: str, default: str = "") -> str:
    try:
        secret = st.secrets.get(name, "")
        if secret and str(secret).strip():
            return str(secret).strip()
    except Exception:
        pass

    return os.getenv(name, default).strip()


def is_placeholder(value: str) -> bool:
    value = value.strip().lower()
    return (
        not value
        or value.startswith("your-")
        or value.startswith("your_")
        or value in {
            "change-this-password",
            "your-real-password",
            "replace-me",
        }
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
        username.strip().lower() == configured_username.lower()
        and password == configured_password
    )


def number(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(37,99,235,.2), transparent 28%),
            radial-gradient(circle at 100% 100%, rgba(20,184,166,.14), transparent 25%),
            #070d19;
        color: #e2e8f0;
    }

    [data-testid="stHeader"] {
        background: rgba(7,13,25,.88);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0b1220,#080f1c);
        border-right: 1px solid rgba(148,163,184,.16);
    }

    .login-box {
        max-width: 470px;
        margin: 8vh auto 20px;
        padding: 38px;
        border-radius: 28px;
        text-align: center;
        background: rgba(15,23,42,.9);
        border: 1px solid rgba(148,163,184,.2);
        box-shadow: 0 25px 80px rgba(0,0,0,.45);
    }

    .ts-logo,
    .mini-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(135deg,#2563eb,#14b8a6);
        box-shadow: 0 12px 30px rgba(20,184,166,.3);
    }

    .ts-logo {
        width: 82px;
        height: 82px;
        margin: 0 auto 18px;
        border-radius: 24px;
        font-size: 30px;
    }

    .mini-logo {
        width: 46px;
        height: 46px;
        border-radius: 14px;
    }

    .login-title {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 850;
    }

    .login-subtitle,
    .dashboard-subtitle {
        color: #94a3b8;
    }

    .credential-box {
        max-width: 470px;
        margin: 15px auto;
        padding: 14px;
        border-radius: 14px;
        background: rgba(14,116,144,.14);
        border: 1px solid rgba(56,189,248,.25);
        color: #bae6fd;
        text-align: center;
        font-size: .85rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 25px;
    }

    .brand-name {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .brand-caption {
        color: #64748b;
        font-size: .72rem;
    }

    .eyebrow {
        color: #38bdf8;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .dashboard-title {
        color: #f8fafc;
        font-size: 2.15rem;
        font-weight: 850;
        letter-spacing: -.05em;
    }

    .hero {
        padding: 28px 30px;
        margin: 20px 0;
        border: 1px solid rgba(96,165,250,.22);
        border-radius: 24px;
        background: linear-gradient(135deg,#1e40af,#0f766e);
        box-shadow: 0 18px 45px rgba(2,6,23,.35);
    }

    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin: 0 0 4px;
    }

    .hero p {
        color: #dbeafe;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        min-height: 120px;
        padding: 20px;
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 18px;
        background: rgba(15,23,42,.72);
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
        margin: 20px 0 10px;
    }

    .status-card,
    .ai-card {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(15,23,42,.72);
        border: 1px solid rgba(148,163,184,.15);
    }

    .ai-card {
        border-color: rgba(56,189,248,.3);
        background: linear-gradient(
            135deg,
            rgba(30,64,175,.2),
            rgba(15,118,110,.14)
        );
    }

    .disclaimer {
        color: #64748b;
        font-size: .78rem;
        text-align: center;
        margin: 32px 0 10px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_login_page() -> None:
    configured_username, configured_password = get_login_credentials()

    st.markdown(
        """
        <div class="login-box">
            <div class="ts-logo">TS</div>
            <div class="login-title">Welcome to Tradesphere</div>
            <div class="login-subtitle">
                AI-powered market intelligence for modern investors
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter username",
            autocomplete="username",
        )
        password = st.text_input(
            "Password",
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
            st.session_state.login_error = "Invalid username or password."

    if st.session_state.get("login_error"):
        st.error(st.session_state.login_error)

    st.markdown(
        f"""
        <div class="credential-box">
            <strong>Demo login</strong><br>
            Username: <code>{configured_username}</code><br>
            Password: <code>{configured_password}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "login_error" not in st.session_state:
    st.session_state.login_error = ""

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if not st.session_state.authenticated:
    show_login_page()
    st.stop()


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
    rs = gain / loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema_12 - ema_26
    result["MACD_Signal"] = result["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    result["BB_Middle"] = middle
    result["BB_Upper"] = middle + 2 * deviation
    result["BB_Lower"] = middle - 2 * deviation

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
        low_window = lows[index - window:index + window + 1]
        high_window = highs[index - window:index + window + 1]

        if lows[index] == np.min(low_window):
            supports.append(float(lows[index]))

        if highs[index] == np.max(high_window):
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
        trend = (
            "Short-term trend is above the long-term trend"
            if latest["SMA_20"] > latest["SMA_50"]
            else "Short-term trend is below the long-term trend"
        )
        heapq.heappush(alerts, (4, trend))

    return alerts


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
        plot_bgcolor="rgba(15,23,42,.6)",
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
            fillcolor="rgba(34,197,94,.08)",
        )
    )

    figure.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    figure.add_hline(y=30, line_dash="dash", line_color="#38bdf8")

    figure.update_layout(
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.6)",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )

    return figure


def build_market_context(ticker: str, data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    change = latest["Close"] - previous["Close"]
    change_percent = change / previous["Close"] * 100
    support, resistance = find_support_resistance(data)
    alerts = build_alert_queue(data)

    return f"""
Ticker: {ticker}
Latest close: {money(latest["Close"])}
Daily change: {money(change)} ({change_percent:.2f}%)
RSI: {number(latest["RSI"])}
MACD: {number(latest["MACD"])}
MACD signal: {number(latest["MACD_Signal"])}
SMA 20: {number(latest["SMA_20"])}
SMA 50: {number(latest["SMA_50"])}
Support levels: {[round(x, 2) for x in support] or "Unavailable"}
Resistance levels: {[round(x, 2) for x in resistance] or "Unavailable"}
Alerts: {[message for _, message in alerts] or "None"}
""".strip()


def ask_ai_assistant(
    question: str,
    ticker: str,
    data: pd.DataFrame,
) -> str:
    api_key = get_config("OPENROUTER_API_KEY")
    model = get_config("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if is_placeholder(api_key):
        return (
            "AI Assistant is not configured. Set the environment variable "
            "`OPENROUTER_API_KEY` or add it to Streamlit Secrets."
        )

    context = build_market_context(ticker, data)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Tradesphere AI, an educational market-analysis "
                "assistant. Use the supplied data, explain indicators clearly, "
                "never guarantee profits, and do not provide personalized "
                "financial advice. Keep responses practical and concise."
            ),
        }
    ]

    for message in st.session_state.ai_messages[-8:]:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                f"Current market context:\n{context}\n\n"
                f"User question:\n{question}"
            ),
        }
    )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": get_config(
                "APP_URL",
                "https://tradesphere.streamlit.app",
            ),
            "X-Title": "Tradesphere AI",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 700,
        },
        timeout=60,
    )

    if not response.ok:
        try:
            error = response.json().get("error", {}).get(
                "message",
                response.text,
            )
        except Exception:
            error = response.text

        raise RuntimeError(f"OpenRouter request failed: {error}")

    result = response.json()
    choices = result.get("choices", [])

    if not choices:
        raise RuntimeError("OpenRouter returned no response choices.")

    return choices[0]["message"]["content"]


def show_ai_assistant(ticker: str, data: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="ai-card">
            <strong>🤖 Tradesphere AI Assistant</strong><br>
            Ask about trends, indicators, alerts, support, resistance, or risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🧹 Clear AI conversation"):
        st.session_state.ai_messages = []
        st.rerun()

    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(f"Ask Tradesphere AI about {ticker}...")

    if question:
        st.session_state.ai_messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Tradesphere AI is analyzing the dashboard..."):
                try:
                    answer = ask_ai_assistant(question, ticker, data)
                except Exception as error:
                    answer = f"AI Assistant error: {error}"

            st.markdown(answer)

        st.session_state.ai_messages.append(
            {"role": "assistant", "content": answer}
        )


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="mini-logo">TS</div>
            <div>
                <div class="brand-name">Tradesphere</div>
                <div class="brand-caption">AI market intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Signed in as **{st.session_state.get('username', 'User')}**"
    )

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

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.ai_messages = []
        st.session_state.pop("market_data", None)
        st.session_state.pop("username", None)
        st.rerun()


if run_analysis or "market_data" not in st.session_state:
    with st.spinner(f"Loading {ticker} market data..."):
        raw_data = load_market_data(ticker, period, interval)

    if raw_data.empty:
        st.error("No market data found. Check the ticker or interval.")
        st.stop()

    st.session_state.market_data = calculate_indicators(raw_data)
    st.session_state.analysis_ticker = ticker

data = st.session_state.market_data
active_ticker = st.session_state.analysis_ticker
latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest

price_change = latest["Close"] - previous["Close"]
percent_change = price_change / previous["Close"] * 100


header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="eyebrow">Live workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-title">Market Command Center</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-subtitle">'
        "Analytics, signals and AI assistance in one place"
        "</div>",
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div class="status-card">
            🟢 Data engine online<br>
            <small>Yahoo Finance</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="hero">
        <h1>TS · {active_ticker}</h1>
        <p>Technical intelligence and AI market context for your selected asset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Educational analytics only — not financial advice.")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Latest Price",
    money(latest["Close"]),
    f"{price_change:.2f} ({percent_change:.2f}%)",
)

metric_2.metric("RSI", number(latest["RSI"]))
metric_3.metric("MACD", number(latest["MACD"]))
metric_4.metric("Volume", f"{latest['Volume']:,.0f}")

overview_tab, technical_tab, ai_tab, news_tab = st.tabs(
    [
        "📊 Overview",
        "🧭 Technical Signals",
        "🤖 AI Assistant",
        "📰 News Intelligence",
    ]
)


with overview_tab:
    st.markdown(
        '<div class="section-title">Price Overview</div>',
        unsafe_allow_html=True,
    )

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
            [round(level, 2) for level in support_levels]
            or "Not enough data",
        )
        st.write(
            "🔴 Resistance levels:",
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
        data[technical_columns].tail(20).sort_index(
            ascending=False
        ).round(2),
        use_container_width=True,
        height=420,
    )


with ai_tab:
    show_ai_assistant(active_ticker, data)


with news_tab:
    st.markdown(
        '<div class="section-title">News Intelligence</div>',
        unsafe_allow_html=True,
    )

    news_api_key = get_config("NEWS_API_KEY")

    if is_placeholder(news_api_key):
        st.info(
            "News intelligence is disabled. Add NEWS_API_KEY to Secrets "
            "to enable it."
        )
    elif st.button("🔎 Load latest news", type="primary"):
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": active_ticker,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": news_api_key,
                },
                timeout=20,
            )
            response.raise_for_status()
            articles = response.json().get("articles", [])

            if not articles:
                st.info("No recent news was found.")
            else:
                for article in articles:
                    title = article.get("title", "Untitled article")
                    url = article.get("url", "#")
                    source = article.get("source", {}).get(
                        "name",
                        "Unknown source",
                    )
                    st.markdown(f"- [{title}]({url}) — *{source}*")
        except Exception as error:
            st.error(f"News request failed: {error}")
    else:
        st.info("Click the button to load recent news.")


st.markdown(
    """
    <div class="disclaimer">
        TS Tradesphere · Data source: Yahoo Finance · Market data may be delayed.
        AI responses are educational and are not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
