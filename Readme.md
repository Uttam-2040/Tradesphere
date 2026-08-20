# 📈 Tradesphere

Tradesphere is a Streamlit-based stock analytics application that combines market data, technical indicators, support and resistance detection, priority alerts, news sentiment analysis, optional LLM analysis, and Stripe subscription checkout.

> **Disclaimer:** This application is for educational purposes only and does not provide financial advice.

## Features

- Yahoo Finance market data through `yfinance`
- Interactive Plotly candlestick charts
- SMA, EMA, RSI, MACD, and Bollinger Bands
- Support and resistance detection using a monotonic stack
- Priority-based technical alerts using `heapq`
- News sentiment analysis using NewsAPI
- Optional OpenAI LLM sentiment analysis
- Stripe subscription checkout
- Streamlit dashboard
- Cached and memory-limited market data processing

## Project Structure

```text
.
├── app.py
├── main.py
├── requirements.txt
└── README.md
