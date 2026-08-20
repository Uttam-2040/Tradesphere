# 📈 Tradesphere

Tradesphere is a modern Streamlit stock analytics dashboard for exploring market data, technical indicators, price levels, alerts, news sentiment, optional AI analysis, and Stripe subscriptions.

> **Disclaimer:** This application is for educational purposes only and does not provide financial advice.

## Features

- Interactive candlestick price chart
- SMA, EMA, RSI, MACD, and Bollinger Bands
- Support and resistance detection
- Priority-based technical alerts
- News sentiment analysis
- Optional OpenAI-powered summaries
- Optional Stripe subscription checkout
- Responsive dark dashboard interface
- Streamlit Cloud deployment support

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run main.py
