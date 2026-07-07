import json
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "Morning Stock Intelligence"
WATCHLIST_FILE = "watchlist.json"
DEFAULT_WATCHLIST = ["QBTS", "GFS", "AMKR", "PLTR", "QUBT", "BLSH", "AMD", "INTC", "SPCX", "TSLA"]

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px;}
    .main-title {font-size: 2.1rem; font-weight: 800; margin-bottom: .15rem;}
    .subtle {color: #6b7280; font-size: .95rem;}
    .stock-card {border: 1px solid #e5e7eb; border-radius: 18px; padding: 18px; margin-bottom: 14px; box-shadow: 0 2px 10px rgba(0,0,0,.04); background: white;}
    .metric-label {color: #6b7280; font-size: .8rem; margin-bottom: .1rem;}
    .metric-value {font-size: 1.25rem; font-weight: 750;}
    .bull {color: #059669; font-weight: 800;}
    .bear {color: #dc2626; font-weight: 800;}
    .neutral {color: #d97706; font-weight: 800;}
    .pill {display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 800; font-size: .85rem;}
    .pill-bull {background: #dcfce7; color: #047857;}
    .pill-bear {background: #fee2e2; color: #b91c1c;}
    .pill-neutral {background: #fef3c7; color: #92400e;}
    @media (max-width: 700px) {
      .main-title {font-size: 1.55rem;}
      .stock-card {padding: 14px; border-radius: 14px;}
      .metric-value {font-size: 1.05rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [str(x).upper().strip() for x in data if str(x).strip()]
        except Exception:
            return DEFAULT_WATCHLIST.copy()
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(items):
    clean = []
    for item in items:
        ticker = str(item).upper().strip().replace(" ", "")
        if ticker and ticker not in clean:
            clean.append(ticker)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    return clean


def pct(a, b):
    try:
        if b == 0 or b is None or math.isnan(b):
            return 0.0
        return ((a - b) / b) * 100
    except Exception:
        return 0.0


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="6mo", interval="1d", auto_adjust=False)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    return hist, info


def analyze_stock(ticker):
    hist, info = fetch_history(ticker)
    if hist is None or hist.empty or len(hist) < 20:
        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "price": None,
            "change_pct": 0,
            "score": 50,
            "outlook": "Neutral",
            "risk": "Unknown",
            "reasons": ["Not enough recent price history was available to calculate a reliable signal."],
            "risks": ["Data may be missing or delayed."],
            "hist": hist,
        }

    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)
    price = safe_float(close.iloc[-1])
    prev = safe_float(close.iloc[-2]) if len(close) >= 2 else price
    change_pct = pct(price, prev)

    sma5 = safe_float(close.rolling(5).mean().iloc[-1])
    sma20 = safe_float(close.rolling(20).mean().iloc[-1])
    sma50 = safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = safe_float(100 - (100 / (1 + rs.iloc[-1])), 50)

    avg_vol20 = safe_float(volume.rolling(20).mean().iloc[-1], 0) if len(volume) >= 20 else 0
    last_vol = safe_float(volume.iloc[-1], 0) if len(volume) else 0
    vol_ratio = last_vol / avg_vol20 if avg_vol20 else 1

    ret5 = pct(price, safe_float(close.iloc[-6])) if len(close) > 6 else 0
    ret20 = pct(price, safe_float(close.iloc[-21])) if len(close) > 21 else 0

    score = 50
    reasons = []
    risks = []

    if price > sma20:
        score += 10
        reasons.append("Price is above the 20-day average, showing short-term strength.")
    else:
        score -= 10
        risks.append("Price is below the 20-day average, which can signal weakness.")

    if sma20 > sma50:
        score += 8
        reasons.append("The 20-day average is above the 50-day average, which supports an upward trend.")
    else:
        score -= 8
        risks.append("The 20-day average is below the 50-day average, which weakens the trend.")

    if change_pct > 1:
        score += 7
        reasons.append("The stock finished the last session with positive momentum.")
    elif change_pct < -1:
        score -= 7
        risks.append("The stock finished the last session with negative momentum.")

    if ret5 > 3:
        score += 7
        reasons.append("Five-day momentum is strong.")
    elif ret5 < -3:
        score -= 7
        risks.append("Five-day momentum is weak.")

    if ret20 > 5:
        score += 6
        reasons.append("One-month trend is positive.")
    elif ret20 < -5:
        score -= 6
        risks.append("One-month trend is negative.")

    if 45 <= rsi <= 68:
        score += 6
        reasons.append("RSI is in a healthy momentum range, not extremely overbought or oversold.")
    elif rsi > 75:
        score -= 8
        risks.append("RSI is high, so the stock may be overbought and vulnerable to profit-taking.")
    elif rsi < 35:
        score -= 3
        risks.append("RSI is weak, showing selling pressure. It may rebound, but risk is elevated.")

    if vol_ratio > 1.3 and change_pct > 0:
        score += 6
        reasons.append("Volume was above normal while price moved up, which confirms buying interest.")
    elif vol_ratio > 1.3 and change_pct < 0:
        score -= 6
        risks.append("Volume was above normal while price moved down, which confirms selling pressure.")

    score = int(max(0, min(100, score)))
    if score >= 70:
        outlook = "Likely Up"
        risk = "Medium" if score < 82 else "Lower"
    elif score <= 40:
        outlook = "Likely Down"
        risk = "High"
    else:
        outlook = "Neutral"
        risk = "Medium"

    if not reasons:
        reasons.append("The signal is mixed, so the app is not seeing a strong bullish setup yet.")
    if not risks:
        risks.append("No major technical warning appeared, but market news can still change direction quickly.")

    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "price": price,
        "change_pct": change_pct,
        "score": score,
        "outlook": outlook,
        "risk": risk,
        "rsi": rsi,
        "ret5": ret5,
        "ret20": ret20,
        "vol_ratio": vol_ratio,
        "sma20": sma20,
        "sma50": sma50,
        "reasons": reasons,
        "risks": risks,
        "hist": hist.tail(60),
    }


def outlook_class(outlook):
    if "Up" in outlook:
        return "pill pill-bull", "bull"
    if "Down" in outlook:
        return "pill pill-bear", "bear"
    return "pill pill-neutral", "neutral"


if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

st.markdown(f'<div class="main-title">📈 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Mobile-friendly morning dashboard for your watchlist. Signals are research tools, not guarantees.</div>', unsafe_allow_html=True)
st.write("")

with st.expander("➕ Add / Delete Stocks", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        new_ticker = st.text_input("Add ticker", placeholder="Example: NVDA")
    with c2:
        st.write("")
        st.write("")
        if st.button("Add Stock", use_container_width=True):
            t = new_ticker.upper().strip().replace(" ", "")
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
                st.session_state.watchlist = save_watchlist(st.session_state.watchlist)
                st.success(f"Added {t}")
            elif t in st.session_state.watchlist:
                st.info(f"{t} is already on your list.")

    remove = st.multiselect("Choose stocks to delete", st.session_state.watchlist)
    if st.button("Delete Selected Stocks", use_container_width=True):
        st.session_state.watchlist = [x for x in st.session_state.watchlist if x not in remove]
        st.session_state.watchlist = save_watchlist(st.session_state.watchlist)
        st.success("Watchlist updated.")

    if st.button("Reset to Simon's Original 10", use_container_width=True):
        st.session_state.watchlist = save_watchlist(DEFAULT_WATCHLIST)
        st.success("Reset complete.")

st.write("")
run_report = st.button("☀️ Run Morning Briefing", type="primary", use_container_width=True)

if run_report or True:
    with st.spinner("Checking prices and calculating signals..."):
        results = [analyze_stock(t) for t in st.session_state.watchlist]

    bullish = [r for r in results if r["score"] >= 70]
    bearish = [r for r in results if r["score"] <= 40]
    neutral = [r for r in results if 40 < r["score"] < 70]
    avg_score = int(np.mean([r["score"] for r in results])) if results else 50
    overall = "Bullish" if avg_score >= 65 else "Bearish" if avg_score <= 40 else "Mixed / Neutral"

    st.markdown("## ☀️ Morning Briefing")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio Signal", overall, f"{avg_score}/100")
    m2.metric("Likely Up", len(bullish))
    m3.metric("Neutral", len(neutral))
    m4.metric("Likely Down", len(bearish))

    top = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    risk = sorted(results, key=lambda x: x["score"])[:3]

    left, right = st.columns(2)
    with left:
        st.markdown("### 🟢 Top Opportunities")
        for r in top:
            st.write(f"**{r['ticker']}** — {r['outlook']} ({r['score']}/100)")
    with right:
        st.markdown("### 🔴 Watch Closely")
        for r in risk:
            st.write(f"**{r['ticker']}** — {r['outlook']} ({r['score']}/100)")

    st.info("This app uses price trend, moving averages, RSI, volume, and recent momentum. News/analyst feeds can be added in the next version with a market-data API key.")

    st.markdown("## Watchlist")
    for r in results:
        pill_class, number_class = outlook_class(r["outlook"])
        price_text = "No data" if r["price"] is None else f"${r['price']:,.2f}"
        change_class = "bull" if r["change_pct"] >= 0 else "bear"
        change_text = f"{r['change_pct']:+.2f}%"

        st.markdown('<div class="stock-card">', unsafe_allow_html=True)
        h1, h2, h3 = st.columns([1.2, 1, 1])
        with h1:
            st.markdown(f"### {r['ticker']}")
            st.caption(r["name"])
        with h2:
            st.markdown('<div class="metric-label">Last Price</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{price_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="{change_class}">{change_text}</span>', unsafe_allow_html=True)
        with h3:
            st.markdown('<div class="metric-label">Today\'s Outlook</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="{pill_class}">{r["outlook"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value {number_class}">{r["score"]}/100</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk", r["risk"])
        c2.metric("5-Day", f"{r.get('ret5', 0):+.1f}%")
        c3.metric("1-Month", f"{r.get('ret20', 0):+.1f}%")
        c4.metric("RSI", f"{r.get('rsi', 50):.0f}")

        with st.expander(f"Why? Explanation for {r['ticker']}"):
            st.markdown("**Why this signal:**")
            for reason in r["reasons"][:5]:
                st.write(f"✅ {reason}")
            st.markdown("**Risks to watch:**")
            for item in r["risks"][:5]:
                st.write(f"⚠️ {item}")
            st.markdown("**Bottom line:**")
            st.write(f"{r['ticker']} currently scores **{r['score']}/100**. The outlook is **{r['outlook']}** based on the technical setup available right now.")
            if r["hist"] is not None and not r["hist"].empty:
                st.line_chart(r["hist"]["Close"])

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Educational research only. Not financial advice. Market data may be delayed or unavailable depending on the source.")
