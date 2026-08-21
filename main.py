import heapq
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from openai import OpenAI


st.set_page_config(
    page_title="Tradesphere | AI Market Intelligence",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_USERNAME = "tsadmin"
DEFAULT_PASSWORD = "TS2026!"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The model can be changed with OPENROUTER_MODEL.
DEFAULT_MODEL = "~deepseek/deepseek-v4-flash-latest"

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def get_config(name: str, default: str = "") -> str:
    """
    Environment variables take priority over Streamlit Secrets.

    This allows:
        export OPENROUTER_API_KEY="sk-or-v1-..."
    """
    environment_value = os.getenv(name)

    if environment_value and environment_value.strip():
        return environment_value.strip().strip('"').strip("'")

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value is not None and str(secret_value).strip():
        return str(secret_value).strip().strip('"').strip("'")

    return default.strip()


def is_placeholder(value: str) -> bool:
    value = str(value or "").strip().lower()

    return (
        not value
        or value.startswith("your-")
        or value.startswith("your_")
        or value.startswith("<")
        or value.endswith(">")
        or value in {
            "change-this-password",
            "your-real-password",
            "replace-me",
            "openrouter_api_key",
            "your-real-openrouter-key",
            "sk-or-v1-your-real-key",
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
    }

    .login-box {
        max-width: 470px;
        margin: 8vh auto 20px;
        padding: 38px;
        border-radius: 28px;
        text-align: center;
        background: rgba(15,23,42,.9);
        border: 1px solid rgba(148,163,184,.2);
    }

    .ts-logo,
    .mini-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 900;
        background: linear-gradient(135deg,#2563eb,#14b8a6);
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

    .login-title,
    .dashboard-title {
        color: #f8fafc;
        font-weight: 850;
    }

    .login-title {
        font-size: 2rem;
    }

    .login-subtitle,
    .dashboard-subtitle {
        color: #94a3b8;
    }

    .credential-box,
    .status-card,
    .ai-card {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(15,23,42,.72);
        border: 1px solid rgba(148,163,184,.15);
    }

    .credential-box {
        max-width: 470px;
        margin: 15px auto;
        text-align: center;
        color: #bae6fd;
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
        font-size: 2.15rem;
    }

    .hero {
        padding: 28px 30px;
        margin: 20px 0;
        border-radius: 24px;
        background: linear-gradient(135deg,#1e40af,#0f766e);
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

    .section-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 750;
        margin: 20px 0 10px;
    }

    .disclaimer {
        color: #64748b;
        font-size: .78rem;
        text-align: center;
        margin: 32px 0 10px;
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
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input(
            "Password",
            type="password",
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


def normalize_market_data(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    result = data.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in result.columns
        ]

    result.columns = [str(column).strip().title() for column in result.columns]
    result = result.loc[:, ~result.columns.duplicated()]

    missing = [column for column in REQUIRED_COLUMNS if column not in result]

    if missing:
        raise ValueError(
            "Yahoo Finance returned incomplete data. "
            f"Missing columns: {', '.join(missing)}"
        )

    result = result[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.replace([np.inf, -np.inf], np.nan).dropna()

    if result.empty:
        return pd.DataFrame()

    result.index = pd.to_datetime(result.index)
    return result.sort_index()


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(
    ticker: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    if interval == "1h" and period in {"2y", "5y"}:
        period = "1y"

    last_error = "Unknown Yahoo Finance error."

    for attempt in range(2):
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30,
            )

            normalized = normalize_market_data(data)

            if not normalized.empty:
                return normalized

            last_error = "Yahoo Finance returned no rows."

        except Exception as error:
            last_error = str(error)

        if attempt == 0:
            continue

    raise RuntimeError(f"Unable to load {ticker} data. {last_error}")


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]

    result["SMA_20"] = close.rolling(20, min_periods=1).mean()
    result["SMA_50"] = close.rolling(50, min_periods=1).mean()
    result["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema_12 - ema_26
    result["MACD_Signal"] = result["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    middle = close.rolling(20, min_periods=1).mean()
    deviation = close.rolling(20, min_periods=1).std()
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
        low_section = lows[index - window:index + window + 1]
        high_section = highs[index - window:index + window + 1]

        if lows[index] == np.min(low_section):
            supports.append(float(lows[index]))

        if highs[index] == np.max(high_section):
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

    if all(
        pd.notna(latest[column])
        for column in ["MACD", "MACD_Signal"]
    ) and all(
        pd.notna(previous[column])
        for column in ["MACD", "MACD_Signal"]
    ):
        if (
            latest["MACD"] > latest["MACD_Signal"]
            and previous["MACD"] <= previous["MACD_Signal"]
        ):
            heapq.heappush(alerts, (2, "MACD bullish crossover detected"))

        if (
            latest["MACD"] < latest["MACD_Signal"]
            and previous["MACD"] >= previous["MACD_Signal"]
        ):
            heapq.heappush(alerts, (2, "MACD bearish crossover detected"))

    if pd.notna(latest["BB_Upper"]) and latest["Close"] > latest["BB_Upper"]:
        heapq.heappush(alerts, (3, "Price is above the upper Bollinger Band"))

    if pd.notna(latest["BB_Lower"]) and latest["Close"] < latest["BB_Lower"]:
        heapq.heappush(alerts, (3, "Price is below the lower Bollinger Band"))

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
    )

    return figure


def build_market_context(ticker: str, data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest

    change = latest["Close"] - previous["Close"]
    change_percent = (
        change / previous["Close"] * 100
        if previous["Close"]
        else 0
    )

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


def get_openrouter_key() -> str:
    key = get_config("OPENROUTER_API_KEY")

    if is_placeholder(key) or not key.startswith("sk-or-v1-"):
        return ""

    return key


def extract_reasoning_details(message: Any) -> Any:
    if hasattr(message, "model_dump"):
        return message.model_dump().get("reasoning_details")

    return getattr(message, "reasoning_details", None)


def convert_message_for_api(message: dict[str, Any]) -> dict[str, Any]:
    result = {
        "role": message["role"],
        "content": message.get("content", ""),
    }

    reasoning_details = message.get("reasoning_details")

    if reasoning_details is not None:
        result["reasoning_details"] = reasoning_details

    return result


def describe_ai_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()

    if (
        "user not found" in lowered
        or "invalid api key" in lowered
        or "401" in lowered
        or "403" in lowered
    ):
        return (
            "OpenRouter rejected the API key. Create a new key at "
            "https://openrouter.ai/keys and set OPENROUTER_API_KEY "
            "in the same terminal used to start Streamlit."
        )

    if "402" in lowered:
        return (
            "OpenRouter has no available credits for this request. "
            "Add credits or choose a free model."
        )

    if "429" in lowered:
        return "OpenRouter rate limit reached. Please wait and try again."

    if "model" in lowered and (
        "not found" in lowered or "invalid" in lowered
    ):
        return (
            f"The selected model is unavailable: "
            f"{get_config('OPENROUTER_MODEL', DEFAULT_MODEL)}. "
            "Set OPENROUTER_MODEL to a valid OpenRouter model ID."
        )

    return text


def ask_ai_assistant(
    question: str,
    ticker: str,
    data: pd.DataFrame,
) -> str:
    api_key = get_openrouter_key()

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing or invalid. "
            "It must begin with sk-or-v1-."
        )

    model = get_config("OPENROUTER_MODEL", DEFAULT_MODEL)

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Tradesphere",
        },
    )

    # The current user question has already been added by the UI.
    # Exclude it here so it is not sent twice.
    previous_messages = st.session_state.ai_messages[:-1]

    history = [
        convert_message_for_api(message)
        for message in previous_messages[-8:]
        if message.get("role") in {"user", "assistant"}
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are Tradesphere AI, an educational market-analysis "
                "assistant. Explain indicators clearly, never guarantee "
                "profits, and do not provide personalized financial advice."
            ),
        },
        *history,
        {
            "role": "user",
            "content": (
                f"Market context:\n{build_market_context(ticker, data)}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"reasoning": {"enabled": True}},
            temperature=0.2,
            max_tokens=700,
        )
    except Exception as error:
        raise RuntimeError(describe_ai_error(error)) from error

    if not response.choices:
        raise RuntimeError("OpenRouter returned no response choices.")

    assistant_message = response.choices[0].message
    answer = assistant_message.content or ""

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    answer = str(answer).strip()

    if not answer:
        raise RuntimeError("OpenRouter returned an empty response.")

    reasoning_details = extract_reasoning_details(assistant_message)

    st.session_state.last_reasoning_details = reasoning_details
    st.session_state.ai_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "reasoning_details": reasoning_details,
        }
    )

    return answer


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
        st.session_state.last_reasoning_details = None
        st.rerun()

    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(f"Ask Tradesphere AI about {ticker}...")

    if not question:
        return

    st.session_state.ai_messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing market data..."):
            try:
                answer = ask_ai_assistant(question, ticker, data)
                st.markdown(answer)
            except Exception as error:
                st.error(f"AI Assistant error: {error}")


def initialize_session() -> None:
    defaults = {
        "authenticated": False,
        "login_error": "",
        "ai_messages": [],
        "last_reasoning_details": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()

if not st.session_state.authenticated:
    show_login_page()
    st.stop()


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

    if interval == "1h" and period in {"2y", "5y"}:
        st.warning(
            "Hourly data will automatically use a maximum period of 1 year."
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

    st.caption(
        f"AI model: `{get_config('OPENROUTER_MODEL', DEFAULT_MODEL)}`"
    )

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.ai_messages = []
        st.session_state.pop("market_data", None)
        st.session_state.pop("username", None)
        st.rerun()


saved_request = st.session_state.get("market_request")
current_request = (ticker, period, interval)

if (
    run_analysis
    or "market_data" not in st.session_state
    or saved_request != current_request
):
    with st.spinner(f"Loading {ticker or 'market'} data..."):
        try:
            raw_data = load_market_data(ticker, period, interval)
            st.session_state.market_data = calculate_indicators(raw_data)
            st.session_state.analysis_ticker = ticker
            st.session_state.market_request = current_request
        except Exception as error:
            st.error(f"Market data could not be loaded: {error}")
            st.info(
                "Try a valid ticker such as AAPL, MSFT, TSLA, or NVDA."
            )
            st.stop()


data = st.session_state.market_data
active_ticker = st.session_state.analysis_ticker
latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest

price_change = latest["Close"] - previous["Close"]
percent_change = (
    price_change / previous["Close"] * 100
    if previous["Close"]
    else 0
)

left, right = st.columns([4, 1])

with left:
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

with right:
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

    support, resistance = find_support_resistance(data)
    column_1, column_2 = st.columns(2)

    with column_1:
        st.markdown(
            '<div class="section-title">Support and Resistance</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "🟢 Support levels:",
            [round(level, 2) for level in support] or "Not enough data",
        )
        st.write(
            "🔴 Resistance levels:",
            [round(level, 2) for level in resistance]
            or "Not enough data",
        )

    with column_2:
        st.markdown(
            '<div class="section-title">Priority Alerts</div>',
            unsafe_allow_html=True,
        )

        for priority, alert in build_alert_queue(data):
            if priority == 1:
                st.error(f"Priority {priority}: {alert}")
            elif priority == 2:
                st.warning(f"Priority {priority}: {alert}")
            else:
                st.info(f"Priority {priority}: {alert}")

with technical_tab:
    st.markdown(
        '<div class="section-title">Relative Strength Index</div>',
        unsafe_allow_html=True,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line={"color": "#22c55e", "width": 2},
            fill="tozeroy",
        )
    )
    figure.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    figure.add_hline(y=30, line_dash="dash", line_color="#38bdf8")
    figure.update_layout(
        height=280,
        template="plotly_dark",
        yaxis_range=[0, 100],
        xaxis_title="Date",
        yaxis_title="RSI",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    columns = [
        "Close",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI",
        "MACD",
        "MACD_Signal",
    ]

    st.dataframe(
        data[columns].tail(20).sort_index(ascending=False).round(2),
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
