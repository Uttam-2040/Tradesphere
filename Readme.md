# 📈 Tradesphere

A Streamlit-based stock analytics web application that combines technical indicators, market data visualization, algorithmic support/resistance detection, priority-based trading alerts, news sentiment analysis, LLM integration, and Stripe subscription payments.

> **Disclaimer:** This application is for educational purposes only and does not provide financial advice.

## Features

- Stock market data powered by Yahoo Finance through `yfinance`
- Interactive candlestick charts using Plotly
- Technical indicators:
  - SMA 20
  - SMA 50
  - EMA 20
  - RSI
  - MACD
  - Bollinger Bands
- Support and resistance detection using a monotonic stack
- Technical alert prioritization using a priority queue
- News collection through NewsAPI
- Rule-based sentiment analysis
- Optional LLM-powered sentiment analysis using OpenAI
- Stripe subscription checkout interface
- Streamlit web interface
- Cached market data to reduce repeated API requests

## Project Structure

```text
.
├── main.py
├── requirements.txt
└── README.md
