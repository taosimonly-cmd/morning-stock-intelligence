import json, os, math, random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

APP_TITLE='Morning Stock Intelligence Pro v4'
WATCHLIST_FILE='watchlist.json'
PORTFOLIO_FILE='portfolio.json'
DEFAULT_WATCHLIST=['QBTS','GFS','AMKR','PLTR','QUBT','BLSH','AMD','INTC','SPCX','TSLA']

st.set_page_config(page_title=APP_TITLE,page_icon='📈',layout='wide',initial_sidebar_state='collapsed')

st.markdown('''<style>
.block-container{padding-top:1.2rem;max-width:1120px}.hero{padding:22px;border-radius:24px;background:linear-gradient(135deg,#111827,#1f2937);color:white;margin-bottom:18px}.hero h1{margin:0;font-size:2.1rem}.sub{color:#cbd5e1}.card{border:1px solid #e5e7eb;border-radius:20px;padding:18px;margin:12px 0;background:white;box-shadow:0 4px 18px rgba(0,0,0,.05)}.darkcard{border:1px solid #374151;border-radius:20px;padding:18px;background:#111827;color:white}.pill{display:inline-block;padding:7px 12px;border-radius:999px;font-weight:800}.bull{background:#dcfce7;color:#047857}.bear{background:#fee2e2;color:#b91c1c}.neutral{background:#fef3c7;color:#92400e}.small{font-size:.88rem;color:#6b7280}.big{font-size:1.55rem;font-weight:850}.radar{border-left:5px solid #6366f1;padding:10px 14px;background:#eef2ff;border-radius:10px;margin:8px 0}@media(max-width:700px){.hero h1{font-size:1.55rem}.card{padding:14px;border-radius:16px}.big{font-size:1.2rem}}</style>''',unsafe_allow_html=True)

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path,'r',encoding='utf-8') as f: return json.load(f)
        except Exception: return default
    return default

def save_json(path, data):
    with open(path,'w',encoding='utf-8') as f: json.dump(data,f,indent=2)

def clean_ticker(t): return str(t).upper().strip().replace(' ','')

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stooq(ticker):
    # Free endpoint with graceful fallback. No Yahoo Finance dependency.
    symbol=ticker.lower()+'.us'
    url=f'https://stooq.com/q/d/l/?s={symbol}&i=d'
    try:
        df=pd.read_csv(url)
        if 'Close' not in df.columns or len(df)<30: raise ValueError('not enough data')
        df['Date']=pd.to_datetime(df['Date'])
        return df.tail(180), None
    except Exception as e:
        return pd.DataFrame(), str(e)

def demo_history(ticker):
    seed=sum(ord(c) for c in ticker)
    rng=np.random.default_rng(seed)
    days=120
    base=25+(seed%160)
    trend=((seed%9)-4)/500
    rets=rng.normal(trend,0.025,days)
    prices=[base]
    for r in rets: prices.append(max(1,prices[-1]*(1+r)))
    dates=pd.date_range(end=datetime.today(),periods=len(prices),freq='B')
    vol=rng.integers(500000,5000000,len(prices))
    return pd.DataFrame({'Date':dates,'Open':prices,'High':np.array(prices)*1.02,'Low':np.array(prices)*.98,'Close':prices,'Volume':vol})

def pct(a,b):
    try: return 0 if not b else (float(a)-float(b))/float(b)*100
    except Exception: return 0

def analyze(ticker):
    df,err=fetch_stooq(ticker)
    source='Live market data'
    if df.empty:
        df=demo_history(ticker); source='Demo fallback data while live feed is unavailable'
    close=df['Close'].astype(float)
    volume=df['Volume'].astype(float) if 'Volume' in df else pd.Series([1]*len(df))
    price=float(close.iloc[-1]); prev=float(close.iloc[-2]); ch=pct(price,prev)
    sma20=float(close.rolling(20).mean().iloc[-1]); sma50=float(close.rolling(50).mean().iloc[-1])
    delta=close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); rsi=float((100-(100/(1+rs))).fillna(50).iloc[-1])
    ret5=pct(price,float(close.iloc[-6])); ret20=pct(price,float(close.iloc[-21]))
    vol_ratio=float(volume.iloc[-1]/max(1,volume.rolling(20).mean().iloc[-1]))
    score=50; reasons=[]; risks=[]
    if price>sma20: score+=12; reasons.append('Price is above the 20-day moving average, showing short-term strength.')
    else: score-=12; risks.append('Price is below the 20-day moving average, showing short-term weakness.')
    if sma20>sma50: score+=10; reasons.append('The 20-day average is above the 50-day average, supporting the trend.')
    else: score-=8; risks.append('The 20-day average is below the 50-day average, so trend strength is weaker.')
    if ch>1: score+=8; reasons.append('The most recent trading session closed with positive momentum.')
    elif ch<-1: score-=8; risks.append('The most recent trading session closed with negative momentum.')
    if ret5>3: score+=8; reasons.append('Five-day momentum is strong.')
    elif ret5<-3: score-=8; risks.append('Five-day momentum is weak.')
    if ret20>6: score+=7; reasons.append('One-month trend is positive.')
    elif ret20<-6: score-=7; risks.append('One-month trend is negative.')
    if 45<=rsi<=68: score+=5; reasons.append('RSI is in a healthy momentum range.')
    elif rsi>75: score-=8; risks.append('RSI is high, so profit-taking risk is elevated.')
    elif rsi<35: score-=4; risks.append('RSI is weak, showing selling pressure.')
    if vol_ratio>1.25 and ch>0: score+=5; reasons.append('Higher-than-normal volume supported the move upward.')
    elif vol_ratio>1.25 and ch<0: score-=5; risks.append('Higher-than-normal volume supported the move downward.')
    score=int(max(0,min(100,score)))
    outlook='Buy / Strong Watch' if score>=75 else 'Hold / Bullish Watch' if score>=62 else 'Neutral / Wait' if score>=42 else 'Caution / Weak'
    klass='bull' if score>=62 else 'bear' if score<42 else 'neutral'
    if not reasons: reasons=['The signal is mixed; no single bullish factor is dominating right now.']
    if not risks: risks=['No major technical warning is visible, but market news can change quickly.']
    news=[f'{ticker}: Check overnight headlines before market open.', f'{ticker}: Watch sector movement and pre-market volume.', f'{ticker}: Monitor analyst notes, earnings calendar, and macro news.']
    return dict(ticker=ticker,price=price,change=ch,score=score,outlook=outlook,klass=klass,rsi=rsi,ret5=ret5,ret20=ret20,vol_ratio=vol_ratio,reasons=reasons,risks=risks,news=news,hist=df[['Date','Close']].tail(90),source=source)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist=load_json(WATCHLIST_FILE, DEFAULT_WATCHLIST.copy())
if 'portfolio' not in st.session_state:
    st.session_state.portfolio=load_json(PORTFOLIO_FILE, {})

st.markdown(f'<div class="hero"><h1>📈 {APP_TITLE}</h1><div class="sub">Cloud-ready personal morning dashboard. Research tool only — not financial advice.</div></div>',unsafe_allow_html=True)

with st.expander('➕ Add / Delete Stocks', expanded=False):
    c1,c2=st.columns([2,1])
    with c1: new=st.text_input('Add ticker',placeholder='Example: NVDA')
    with c2:
        st.write('');
        if st.button('Add Stock',use_container_width=True):
            t=clean_ticker(new)
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t); save_json(WATCHLIST_FILE,st.session_state.watchlist); st.success(f'Added {t}')
    remove=st.multiselect('Delete tickers',st.session_state.watchlist)
    if st.button('Delete Selected',use_container_width=True):
        st.session_state.watchlist=[x for x in st.session_state.watchlist if x not in remove]; save_json(WATCHLIST_FILE,st.session_state.watchlist); st.success('Watchlist updated')

with st.expander('💼 Portfolio Tracker', expanded=False):
    st.caption('Optional: enter your shares and average cost. This stays saved in the app storage.')
    for t in st.session_state.watchlist:
        current=st.session_state.portfolio.get(t,{})
        a,b=st.columns(2)
        shares=a.number_input(f'{t} shares',min_value=0.0,value=float(current.get('shares',0.0)),step=1.0,key=f'sh_{t}')
        cost=b.number_input(f'{t} average cost',min_value=0.0,value=float(current.get('cost',0.0)),step=0.01,key=f'co_{t}')
        st.session_state.portfolio[t]={'shares':shares,'cost':cost}
    if st.button('Save Portfolio',use_container_width=True): save_json(PORTFOLIO_FILE,st.session_state.portfolio); st.success('Portfolio saved')

run=st.button('☀️ Run Morning Briefing',type='primary',use_container_width=True)
if not run:
    st.info('Tap **Run Morning Briefing** each morning. This version avoids Yahoo Finance crashes and uses safe fallback data if a live feed is unavailable.')

if run:
    with st.spinner('Building your morning intelligence report...'):
        results=[analyze(t) for t in st.session_state.watchlist]
    avg=int(np.mean([r['score'] for r in results])); mood='Bullish' if avg>=62 else 'Cautious' if avg<42 else 'Mixed'
    bullish=[r for r in results if r['score']>=62]; weak=[r for r in results if r['score']<42]
    st.markdown('## ☀️ Simon\'s Morning Briefing')
    a,b,c,d=st.columns(4)
    a.metric('Portfolio Mood',mood,f'{avg}/100'); b.metric('Bullish Watch',len(bullish)); c.metric('Caution',len(weak)); d.metric('Stocks Checked',len(results))
    st.markdown('### 📡 Simon\'s Market Radar')
    for r in sorted(results,key=lambda x:x['score'],reverse=True)[:4]:
        st.markdown(f'<div class="radar"><b>{r["ticker"]}</b>: {r["outlook"]} — confidence {r["score"]}/100. {r["reasons"][0]}</div>',unsafe_allow_html=True)
    st.markdown('### Watchlist Intelligence')
    total_value=0; total_cost=0
    for r in results:
        p=st.session_state.portfolio.get(r['ticker'],{}); shares=float(p.get('shares',0) or 0); cost=float(p.get('cost',0) or 0)
        value=shares*r['price']; basis=shares*cost; pnl=value-basis; total_value+=value; total_cost+=basis
        st.markdown('<div class="card">',unsafe_allow_html=True)
        c1,c2,c3=st.columns([1.3,1,1])
        c1.markdown(f'### {r["ticker"]}<div class="small">{r["source"]}</div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="small">Last Price</div><div class="big">${r["price"]:,.2f}</div><span class="{ "bull" if r["change"]>=0 else "bear" }">{r["change"]:+.2f}%</span>',unsafe_allow_html=True)
        c3.markdown(f'<div class="small">AI Signal</div><span class="pill {r["klass"]}">{r["outlook"]}</span><div class="big">{r["score"]}/100</div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric('5-Day',f'{r["ret5"]:+.1f}%'); m2.metric('1-Month',f'{r["ret20"]:+.1f}%'); m3.metric('RSI',f'{r["rsi"]:.0f}'); m4.metric('Position P/L',f'${pnl:,.0f}' if shares else 'Not entered')
        with st.expander(f'Why? Explanation for {r["ticker"]}'):
            st.markdown('**Why this signal:**')
            for x in r['reasons'][:5]: st.write('✅ '+x)
            st.markdown('**Risks to watch:**')
            for x in r['risks'][:5]: st.write('⚠️ '+x)
            st.markdown('**News checklist:**')
            for x in r['news']: st.write('📰 '+x)
            chart=r['hist'].set_index('Date')
            st.line_chart(chart['Close'])
        st.markdown('</div>',unsafe_allow_html=True)
    if total_value>0:
        st.markdown('## 💼 Portfolio Summary')
        pnl=total_value-total_cost
        c1,c2,c3=st.columns(3)
        c1.metric('Current Value',f'${total_value:,.2f}'); c2.metric('Cost Basis',f'${total_cost:,.2f}'); c3.metric('Total P/L',f'${pnl:,.2f}',f'{pct(total_value,total_cost):+.2f}%')

st.markdown('---')
st.caption('Educational research only. Not financial advice. Live data source may be delayed or unavailable; fallback data keeps the app usable during outages.')
