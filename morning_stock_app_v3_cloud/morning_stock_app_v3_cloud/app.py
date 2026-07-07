import json, os, math, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE='Morning Stock Intelligence Pro v5'
WATCHLIST_FILE='watchlist.json'
PORTFOLIO_FILE='portfolio.json'
DEFAULT_WATCHLIST=['QBTS','GFS','AMKR','PLTR','QUBT','BLSH','INTC','SPCX','AMD','TSLA','NVDA']

st.set_page_config(page_title=APP_TITLE,page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container{padding-top:.8rem;max-width:1080px}.hero{background:linear-gradient(135deg,#111827,#1f2937);border-radius:26px;padding:28px;margin-bottom:18px}.hero h1{font-size:2.1rem;margin:0;color:white}.hero p{color:#cbd5e1;font-size:1rem}.card{border:1px solid #263244;background:#0b1220;border-radius:18px;padding:16px;margin:11px 0}.small{color:#94a3b8;font-size:.85rem}.green{color:#22c55e;font-weight:800}.red{color:#ef4444;font-weight:800}.yellow{color:#f59e0b;font-weight:800}.pill{display:inline-block;padding:5px 10px;border-radius:999px;font-size:.82rem;font-weight:800}.p-green{background:#064e3b;color:#bbf7d0}.p-red{background:#7f1d1d;color:#fecaca}.p-yellow{background:#78350f;color:#fde68a}.source{font-size:.8rem;color:#93c5fd}.warn{background:#1e293b;border-left:5px solid #38bdf8;padding:14px;border-radius:14px;color:#bfdbfe}.stButton>button{border-radius:14px;min-height:46px;font-weight:700}@media(max-width:700px){.hero{padding:18px;border-radius:18px}.hero h1{font-size:1.45rem}.card{padding:13px;border-radius:15px}.block-container{padding-left:.7rem;padding-right:.7rem}}
</style>''',unsafe_allow_html=True)

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path,'r',encoding='utf-8') as f: return json.load(f)
        except Exception: return default
    return default

def save_json(path, data):
    with open(path,'w',encoding='utf-8') as f: json.dump(data,f,indent=2)

def clean_ticker(t): return str(t).upper().strip().replace(' ','')
def pct(a,b):
    try:
        if b in (None,0) or pd.isna(b): return 0.0
        return (float(a)-float(b))/float(b)*100
    except Exception: return 0.0

def fnum(x, d=2):
    try:
        if x is None or pd.isna(x): return None
        return float(x)
    except Exception: return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live(ticker):
    """Try several yfinance methods. Never crash; return data status and timestamp."""
    now=datetime.now().strftime('%Y-%m-%d %I:%M %p')
    out={'ticker':ticker,'price':None,'prev_close':None,'open':None,'volume':None,'name':ticker,'status':'Unavailable','source':'No live source','updated':now,'error':None,'hist':pd.DataFrame()}
    try:
        tk=yf.Ticker(ticker)
        # fast info first
        try:
            fi=tk.fast_info
            out['price']=fnum(getattr(fi,'last_price',None) or fi.get('last_price',None))
            out['prev_close']=fnum(getattr(fi,'previous_close',None) or fi.get('previous_close',None))
            out['open']=fnum(getattr(fi,'open',None) or fi.get('open',None))
            out['source']='Yahoo fast_info'
        except Exception:
            pass
        hist=tk.history(period='6mo', interval='1d', auto_adjust=False)
        if hist is not None and not hist.empty:
            out['hist']=hist
            close=hist['Close'].dropna()
            if out['price'] is None and len(close): out['price']=fnum(close.iloc[-1])
            if out['prev_close'] is None and len(close)>1: out['prev_close']=fnum(close.iloc[-2])
            if 'Volume' in hist and len(hist['Volume'].dropna()): out['volume']=fnum(hist['Volume'].dropna().iloc[-1])
            out['source']='Yahoo Finance history'
        try:
            info=tk.get_info()
            if isinstance(info,dict):
                out['name']=info.get('shortName') or info.get('longName') or ticker
                out['price']=fnum(info.get('currentPrice') or info.get('regularMarketPrice') or out['price'])
                out['prev_close']=fnum(info.get('previousClose') or out['prev_close'])
                out['source']='Yahoo quote/info'
        except Exception:
            pass
        if out['price'] is not None:
            out['status']='Live/Delayed'
        elif out['hist'] is not None and not out['hist'].empty:
            out['status']='Delayed History'
        return out
    except Exception as e:
        out['error']=str(e)
        return out

# Conservative sample fallback so app stays usable, clearly labeled
FALLBACK={'QBTS':23.09,'GFS':69.99,'AMKR':71.41,'PLTR':132.94,'QUBT':9.38,'BLSH':26.49,'INTC':124.05,'SPCX':162.07,'AMD':558.11,'TSLA':418.43,'NVDA':196.42}

def analyze(q):
    ticker=q['ticker']; price=q['price']; prev=q['prev_close']; hist=q['hist']
    used_fallback=False
    if price is None:
        price=FALLBACK.get(ticker, 100.0); prev=price; used_fallback=True
    chg=pct(price, prev if prev else price)
    score=50; reasons=[]; risks=[]; rsi=50; ret5=0; ret20=0; vol_ratio=1
    if hist is not None and not hist.empty and 'Close' in hist:
        close=hist['Close'].dropna()
        if len(close)>21:
            sma20=close.rolling(20).mean().iloc[-1]
            sma50=close.rolling(50).mean().iloc[-1] if len(close)>50 else sma20
            delta=close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
            rs=gain/loss.replace(0,np.nan); rsi=fnum(100-(100/(1+rs.iloc[-1])),50) or 50
            ret5=pct(close.iloc[-1], close.iloc[-6]) if len(close)>6 else 0
            ret20=pct(close.iloc[-1], close.iloc[-21])
            vol=hist['Volume'].dropna() if 'Volume' in hist else pd.Series(dtype=float)
            if len(vol)>20:
                avg=vol.rolling(20).mean().iloc[-1]; vol_ratio=float(vol.iloc[-1]/avg) if avg else 1
            if price>sma20: score+=10; reasons.append('Price is above the 20-day average.')
            else: score-=10; risks.append('Price is below the 20-day average.')
            if sma20>sma50: score+=8; reasons.append('Short-term trend is stronger than the 50-day trend.')
            else: score-=8; risks.append('Short-term trend is weaker than the 50-day trend.')
            if ret5>3: score+=7; reasons.append('Five-day momentum is positive.')
            elif ret5<-3: score-=7; risks.append('Five-day momentum is negative.')
            if 42<=rsi<=68: score+=6; reasons.append('RSI is in a healthy range.')
            elif rsi>75: score-=8; risks.append('RSI is high; profit-taking risk is elevated.')
            elif rsi<35: score-=4; risks.append('RSI is weak; selling pressure is present.')
            if vol_ratio>1.25 and chg>0: score+=5; reasons.append('Above-normal volume supports buying interest.')
            elif vol_ratio>1.25 and chg<0: score-=5; risks.append('Above-normal volume confirms selling pressure.')
    else:
        risks.append('Live technical history is not available right now.')
    if used_fallback:
        risks.append('Price is fallback/sample data. Do not use it as a live quote.')
        score=50
    score=int(max(0,min(100,score)))
    outlook='Likely Up' if score>=70 else 'Likely Down' if score<=40 else 'Neutral'
    if not reasons: reasons.append('The available signal is mixed or incomplete.')
    return {**q,'price':price,'change_pct':chg,'score':score,'outlook':outlook,'rsi':rsi,'ret5':ret5,'ret20':ret20,'vol_ratio':vol_ratio,'reasons':reasons,'risks':risks,'used_fallback':used_fallback}

if 'watchlist' not in st.session_state:
    st.session_state.watchlist=load_json(WATCHLIST_FILE, DEFAULT_WATCHLIST.copy())
if 'portfolio' not in st.session_state:
    st.session_state.portfolio=load_json(PORTFOLIO_FILE,{})

st.markdown(f'<div class="hero"><h1>📈 {APP_TITLE}</h1><p>Version 5 adds clearer price accuracy labels, last-updated time, fallback warnings, and portfolio tracking.</p></div>',unsafe_allow_html=True)

with st.expander('➕ Add / Delete Stocks'):
    c1,c2=st.columns([2,1])
    with c1: nt=st.text_input('Add ticker', placeholder='Example: NVDA')
    with c2:
        st.write('');
        if st.button('Add Stock', use_container_width=True):
            t=clean_ticker(nt)
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t); save_json(WATCHLIST_FILE, st.session_state.watchlist); st.success(f'Added {t}')
            elif t: st.info(f'{t} is already on your watchlist.')
    rem=st.multiselect('Delete tickers', st.session_state.watchlist)
    if st.button('Delete Selected', use_container_width=True):
        st.session_state.watchlist=[x for x in st.session_state.watchlist if x not in rem]; save_json(WATCHLIST_FILE, st.session_state.watchlist); st.success('Updated')

with st.expander('💼 Portfolio Tracker'):
    st.caption('Optional: enter shares and average cost. Data is saved for this Streamlit app session/storage.')
    for t in st.session_state.watchlist:
        cols=st.columns([1,1,1])
        p=st.session_state.portfolio.get(t,{})
        with cols[0]: st.write(f'**{t}**')
        with cols[1]: shares=st.number_input(f'{t} shares', min_value=0.0, value=float(p.get('shares',0)), step=1.0, key=f'sh_{t}')
        with cols[2]: avg=st.number_input(f'{t} avg cost', min_value=0.0, value=float(p.get('avg',0)), step=1.0, key=f'av_{t}')
        st.session_state.portfolio[t]={'shares':shares,'avg':avg}
    if st.button('Save Portfolio', use_container_width=True):
        save_json(PORTFOLIO_FILE, st.session_state.portfolio); st.success('Portfolio saved')

run=st.button('☀️ Run Morning Briefing', type='primary', use_container_width=True)
if not run:
    st.markdown('<div class="warn">Tap <b>Run Morning Briefing</b>. V5 clearly marks each quote as Live/Delayed or Fallback so you know whether the price is reliable.</div>',unsafe_allow_html=True)
else:
    results=[]
    prog=st.progress(0)
    for i,t in enumerate(st.session_state.watchlist):
        q=fetch_live(t); results.append(analyze(q)); prog.progress((i+1)/len(st.session_state.watchlist))
    prog.empty()
    avg=int(np.mean([r['score'] for r in results])) if results else 50
    bull=sum(1 for r in results if r['score']>=70); bear=sum(1 for r in results if r['score']<=40); neu=len(results)-bull-bear
    st.markdown("## ☀️ Simon's Morning Briefing")
    a,b,c,d=st.columns(4)
    a.metric('Portfolio Signal', 'Bullish' if avg>=65 else 'Bearish' if avg<=40 else 'Mixed', f'{avg}/100')
    b.metric('Likely Up', bull); c.metric('Neutral', neu); d.metric('Likely Down', bear)
    st.markdown('### Simon\'s Market Radar')
    for r in sorted(results,key=lambda x:x['score'], reverse=True)[:3]: st.write(f"✅ **{r['ticker']}** — {r['outlook']} ({r['score']}/100)")
    fallback_count=sum(1 for r in results if r['used_fallback'])
    if fallback_count: st.warning(f'{fallback_count} ticker(s) are using fallback/sample prices. Check the source label on each card before making decisions.')
    total_value=0; total_cost=0
    st.markdown('## Watchlist Intelligence')
    for r in results:
        cls='p-green' if r['outlook']=='Likely Up' else 'p-red' if r['outlook']=='Likely Down' else 'p-yellow'
        price=f"${r['price']:,.2f}" if r['price'] is not None else 'No price'
        source_label='FALLBACK / SAMPLE' if r['used_fallback'] else r['status']
        source_cls='red' if r['used_fallback'] else 'green'
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1,c2,c3=st.columns([1.1,1,1])
        with c1:
            st.markdown(f"### {r['ticker']}")
            st.caption(r.get('name',r['ticker']))
            st.markdown(f'<span class="source {source_cls}">Data: {source_label}</span>', unsafe_allow_html=True)
            st.caption(f"Last updated: {r['updated']}")
        with c2:
            st.metric('Price', price, f"{r['change_pct']:+.2f}%")
        with c3:
            st.markdown(f'<span class="pill {cls}">{r["outlook"]}</span>', unsafe_allow_html=True)
            st.markdown(f"### {r['score']}/100")
        p=st.session_state.portfolio.get(r['ticker'],{})
        shares=float(p.get('shares',0) or 0); avgcost=float(p.get('avg',0) or 0)
        if shares>0 and avgcost>0 and r['price']:
            val=shares*r['price']; cost=shares*avgcost; gl=val-cost; total_value+=val; total_cost+=cost
            st.metric('Your position', f'${val:,.2f}', f'${gl:,.2f} total gain/loss')
        m1,m2,m3=st.columns(3); m1.metric('5-Day',f"{r['ret5']:+.1f}%"); m2.metric('1-Month',f"{r['ret20']:+.1f}%"); m3.metric('RSI',f"{r['rsi']:.0f}")
        with st.expander(f'Why? Explanation for {r["ticker"]}'):
            st.write('**Positive factors**')
            for x in r['reasons'][:6]: st.write('✅ '+x)
            st.write('**Risks / warnings**')
            for x in r['risks'][:6]: st.write('⚠️ '+x)
            if r['hist'] is not None and not r['hist'].empty and 'Close' in r['hist']: st.line_chart(r['hist']['Close'].tail(90))
        st.markdown('</div>', unsafe_allow_html=True)
    if total_value>0:
        st.markdown('## Portfolio Summary')
        st.metric('Estimated Value', f'${total_value:,.2f}', f'${total_value-total_cost:,.2f} total gain/loss')

st.markdown('---')
st.caption('Educational research only. Not financial advice. V5 shows data source and fallback warnings so incorrect live quotes are easier to catch.')
