import os, base64, logging, math, functools, textwrap
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
try:
    from openai import OpenAI
except ImportError:
    import openai as OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s│%(levelname)s│%(message)s")

_PASSWORD = "MPFvive8955@#@"

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    if st.text_input("🔒 Enter password", type="password") == _PASSWORD and st.button("Login"):
        st.session_state.auth = True
        st.experimental_rerun()
    st.stop()

st.set_page_config("QualityROI Suite", "📊", layout="wide")

st.markdown("""
<style>
:root{--green:#10B981;--red:#EF4444;--blue:#2563EB;--gray:#6B7280}
body{font-variant-ligatures:none}
.main{font-size:2.6rem;font-weight:700;color:#1E3A8A;margin:.2rem 0 1rem}
.sub{font-size:1.8rem;font-weight:600;color:var(--blue);margin:.2rem 0 .8rem}
.card{background:#fff;border-radius:.6rem;padding:1.4rem;box-shadow:0 4px 6px rgba(0,0,0,.08);margin:.7rem 0}
.metric-l{font-size:1rem;color:var(--gray)}
.metric-v{font-size:1.6rem;font-weight:600;color:#111827}
.recommendation-high{background:#DCFCE7;color:#166534;padding:.5rem .8rem;border-radius:.3rem;font-weight:600}
.recommendation-medium{background:#FEF3C7;color:#92400E;padding:.5rem .8rem;border-radius:.3rem;font-weight:600}
.recommendation-low{background:#FEE2E2;color:#B91C1C;padding:.5rem .8rem;border-radius:.3rem;font-weight:600}
.chat-message{padding:1rem;border-radius:.5rem;margin:.4rem 0;max-width:80%}
.user-message{background:#E2E8F0;margin-left:auto}
.ai-message{background:#DBEAFE;margin-right:auto}
.export-btn{color:var(--blue);background:#EFF6FF;border:1px solid #BFDBFE;padding:.45rem .75rem;border-radius:.4rem;text-decoration:none;font-size:.87rem;display:inline-flex;gap:.4rem;align-items:center}
</style>
""", unsafe_allow_html=True)

fmt_cur = lambda v: f"${v:,.2f}"
fmt_pct = lambda v: f"{v:.1f}%"

def download_link(df:pd.DataFrame, fn:str, label:str)->str:
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="export-btn" download="{fn}" href="data:file/csv;base64,{b64}">📥 {label}</a>'

def bar(title,lbl,cur,new):
    fig=go.Figure()
    fig.add_bar(x=lbl,y=cur,name="Current",marker_color="#3B82F6")
    fig.add_bar(x=lbl,y=new,name="Scenario",marker_color="#10B981")
    fig.update_layout(title=title,barmode="group",height=320,margin=dict(l=10,r=10,t=35,b=20))
    st.plotly_chart(fig,use_container_width=True)

# cache OpenAI client
@st.cache_resource(show_spinner=False,max_entries=1)
def openai_client():
    k = os.getenv("OPENAI_API_KEY","")
    if not k:
        return None
    try:
        cli=OpenAI(api_key=k)
        cli.chat.completions.create(model="gpt-4o",messages=[{"role":"user","content":"hi"}],max_tokens=1)
        return cli
    except Exception as e:
        logging.error(e);return None

client=openai_client()

# state defaults
_DEF_SALES = pd.DataFrame({"date":[],"sales":[],"channel":[],"category":[]})
DEFAULTS=dict(quality=None,quality_hist=[],salvage=None,salvage_hist=[],sales=_DEF_SALES,chat=[])
for k,v in DEFAULTS.items():
    st.session_state.setdefault(k,v)

# -------------------------------- QUALITY ----------------------------------

def calc_quality(sku:str,ptype:str,sales:float,ret:float,uc:float,fix_up:float,fix_per:float,price:float):
    rr=ret/sales*100 if sales else 0
    est_rr=rr*.2
    est_ret=sales*est_rr/100
    new_uc=uc+fix_per
    cur_prof=price-uc
    new_prof=price-new_uc
    cur_margin=cur_prof/price*100 if price else 0
    new_margin=new_prof/price*100 if price else 0
    cur_p=cur_prof*(sales-ret)
    new_p=new_prof*(sales-est_ret)
    save=cur_p-new_p+(ret-est_ret)*uc
    pay=fix_up/save if save>0 else math.inf
    roi=((save*36)-(fix_up+fix_per*sales*36))/(fix_up+fix_per*sales*36)*100 if save>0 else -100
    rec,cls=("Highly Recommended","recommendation-high") if pay<3 else ("Recommended","recommendation-medium") if pay<6 else ("Consider","recommendation-medium") if pay<12 else ("Not Recommended","recommendation-low")
    return dict(sku=sku,ptype=ptype,sales=sales,ret=ret,rr=rr,est_rr=est_rr,uc=uc,new_uc=new_uc,cur_margin=cur_margin,new_margin=new_margin,save=save,pay=pay,roi=roi,rec=rec,cls=cls)

def quality_ui():
    st.markdown('<div class="sub">Quality Issue Analysis</div>',unsafe_allow_html=True)
    sku=st.text_input("SKU")
    c1,c2=st.columns(2)
    with c1:
        sales=st.number_input("Units sold (30d)",0.0)
        ret=st.number_input("Units returned (30d)",0.0)
        uc=st.number_input("Unit cost",0.0)
        price=st.number_input("Sales price",0.0)
    with c2:
        fix_up=st.number_input("Fix upfront",0.0)
        fix_per=st.number_input("Addl cost/unit",0.0)
        ptype=st.selectbox("Product type",["B2C","B2B","Both"])
    if st.button("Analyze Quality"):
        st.session_state.quality=calc_quality(sku,ptype,sales,ret,uc,fix_up,fix_per,price)
        st.session_state.quality_hist.append(st.session_state.quality)

    if st.session_state.quality:
        r=st.session_state.quality
        st.markdown(f'<div class="card"><span class="sub">{r["sku"]}</span>',unsafe_allow_html=True)
        col1,col2,col3=st.columns(3)
        with col1:
            st.markdown('<div class="metric-l">Return rate</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_pct(r["rr"])}</div>',unsafe_allow_html=True)
            st.markdown('<div class="metric-l">Future return</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_pct(r["est_rr"])}</div>',unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-l">Monthly savings</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_cur(r["save"])}</div>',unsafe_allow_html=True)
            st.markdown('<div class="metric-l">Payback</div>',unsafe_allow_html=True)
            pb="∞" if math.isinf(r["pay"]) else f"{r['pay']:.1f} mo"
            st.markdown(f'<div class="metric-v">{pb}</div>',unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-l">Current margin</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_pct(r["cur_margin"])}</div>',unsafe_allow_html=True)
            st.markdown('<div class="metric-l">Future margin</div>',unsafe_allow_html=True)
            col="var(--green)" if r["new_margin"]>=r["cur_margin"] else "var(--red)"
            st.markdown(f'<div class="metric-v" style="color:{col}">{fmt_pct(r["new_margin"])}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="{r["cls"]}">{r["rec"]}</div></div>',unsafe_allow_html=True)

# -------------------------------- SALVAGE ----------------------------------

def calc_salvage(sku:int,inv:int,uc:float,setup:float,per:float,recover:float,disc:float,price:float):
    rec=inv*recover/100
    disc_price=price*(1-disc/100)
    cost=setup+per*inv
    revenue=rec*disc_price
    write_off=(inv-rec)*uc
    profit=revenue-cost-write_off
    roi=profit/cost*100 if cost else -100
    rec_txt,cls=("Highly Recommended","recommendation-high") if profit>0 and roi>20 else ("Recommended","recommendation-medium") if profit>0 else ("Consider","recommendation-medium") if profit>-0.3*inv*uc else ("Not Recommended","recommendation-low")
    return dict(sku=sku,inv=inv,rec_units=rec,disc_price=disc_price,cost=cost,revenue=revenue,write_off=write_off,profit=profit,roi=roi,recover=recover,disc=disc,per=per,setup=setup,uc=uc,price=price,rec=rec_txt,cls=cls)

def salvage_ui():
    st.markdown('<div class="sub">Salvage / Re‑work</div>',unsafe_allow_html=True)
    sku=st.text_input("SKU",key="salv_sku")
    c1,c2=st.columns(2)
    with c1:
        inv=st.number_input("Affected inventory",1)
        uc=st.number_input("Unit cost",0.0)
        setup=st.number_input("Setup cost",0.0)
        per=st.number_input("Rework $/unit",0.0)
    with c2:
        price=st.number_input("Original price",0.0)
        recover=st.slider("Recovery %",0.0,100.0,80.0,5.0)
        disc=st.slider("Discount %",0.0,100.0,30.0,5.0)
    if st.button("Analyze Salvage"):
        st.session_state.salvage=calc_salvage(sku,inv,uc,setup,per,recover,disc,price)
        st.session_state.salvage_hist.append(st.session_state.salvage)

    if st.session_state.salvage:
        r=st.session_state.salvage
        st.markdown(f'<div class="card"><span class="sub">{r["sku"]}</span>',unsafe_allow_html=True)
        col1,col2,col3=st.columns(3)
        with col1:
            st.markdown('<div class="metric-l">Recovered units</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{r["rec_units"]:.0f}/{r["inv"]}</div>',unsafe_allow_html=True)
            st.markdown('<div class="metric-l">Discounted price</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_cur(r["disc_price"])}</div>',unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-l">Profit/Loss</div>',unsafe_allow_html=True)
            col="var(--green)" if r["profit"]>=0 else "var(--red)"
            st.markdown(f'<div class="metric-v" style="color:{col}">{fmt_cur(r["profit"])}</div>',unsafe_allow_html=True)
            st.markdown('<div class="metric-l">ROI</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_pct(r["roi"])}</div>',unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-l">Rework cost</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_cur(r["cost"])}</div>',unsafe_allow_html=True)
            st.markdown('<div class="metric-l">Write‑off cost</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-v">{fmt_cur(r["write_off"])}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="{r["cls"]}">{r["rec"]}</div>',unsafe_allow_html=True)
        
        st.markdown('<div class="sub">Scenario modeller</div>',unsafe_allow_html=True)
        s1,s2,s3=st.columns(3)
        with s1:
            rec_adj=st.slider("Recovery %",0.0,100.0,r["recover"],5.0,key="rec_adj")
        with s2:
            disc_adj=st.slider("Discount %",0.0,100.0,r["disc"],5.0,key="disc_adj")
        with s3:
            per_adj=st.number_input("Rework $/unit",0.0,value=r["per"],step=0.25,key="per_adj")
        if st.button("Run Scenario"):
            new=calc_salvage(r['sku'],r['inv'],r['uc'],r['setup'],per_adj,rec_adj,disc_adj,r['price'])
            bar("Financial impact",["Cost","Revenue","Write‑off","Profit"],[r['cost'],r['revenue'],r['write_off'],r['profit']],[new['cost'],new['revenue'],new['write_off'],new['profit']])
            st.markdown(f'<div class="{new["cls"]}">{new["rec"]}</div>',unsafe_allow_html=True)

# -------------------------------- SALES ------------------------------------

def analyze_sales(df:pd.DataFrame):
    df=df.copy();df['date']=pd.to_datetime(df['date'])
    daily=df.groupby('date')['sales'].sum().reset_index();daily['ma7']=daily['sales'].rolling(7).mean()
    monthly=daily.copy();monthly['ym']=monthly['date'].dt.to_period('M');monthly=monthly.groupby('ym')['sales'].sum().reset_index()
    chan=df.groupby('channel')['sales'].sum().reset_index();cat=df.groupby('category')['sales'].sum().reset_index()
    return dict(daily=daily,monthly=monthly,chan=chan,cat=cat)

def sales_ui():
    st.markdown('<div class="sub">Sales Analytics</div>',unsafe_allow_html=True)
    df=st.session_state.sales
    upl=st.file_uploader("CSV with date,sales,channel,category",type="csv")
    if upl:
        try:
            df=pd.read_csv(upl)
            st.session_state.sales=df
        except Exception as e:
            st.error(e)
    if st.button("Add demo data"):
        rng=pd.date_range(datetime.today()-timedelta(days=120),periods=120)
        st.session_state.sales=pd.DataFrame({
            'date':rng,'sales':np.random.randint(100,1000,len(rng)),
            'channel':np.random.choice(['Amazon','Web','Retail'],len(rng)),
            'category':np.random.choice(['Health','Fitness','Other'],len(rng))})
    df=st.session_state.sales
    if df.empty:
        st.info("Upload or create data to continue");return
    df['date']=pd.to_datetime(df['date'])
    d1,d2=st.columns(2)
    with d1:
        start=st.date_input("From",df['date'].min().date())
    with d2:
        end=st.date_input("To",df['date'].max().date())
    mask=(df['date'].dt.date>=start)&(df['date'].dt.date<=end)
    df=df[mask]
    ch=st.multiselect("Channel",sorted(df['channel'].unique()),default=list(sorted(df['channel'].unique())))
    cat=st.multiselect("Category",sorted(df['category'].unique()),default=list(sorted(df['category'].unique())))
    df=df[df['channel'].isin(ch)&df['category'].isin(cat)]
    if df.empty:
        st.warning("No data for filter");return
    res=analyze_sales(df)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<div class="metric-l">Total sales</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v">{fmt_cur(df["sales"].sum())}</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-l">Avg daily</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v">{fmt_cur(res["daily"]["sales"].mean())}</div>',unsafe_allow_html=True)
    with c3:
        growth=0
        if len(res['monthly'])>1:
            growth=(res['monthly']['sales'].iloc[-1]-res['monthly']['sales'].iloc[0])/res['monthly']['sales'].iloc[0]*100
        col="var(--green)" if growth>=0 else "var(--red)"
        st.markdown('<div class="metric-l">Growth</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v" style="color:{col}">{fmt_pct(growth)}</div>',unsafe_allow_html=True)
    st.plotly_chart(px.line(res['daily'],x='date',y=['sales','ma7'],labels={'value':'$','variable':' '},color_discrete_map={'sales':'#3B82F6','ma7':'#10B981'}),use_container_width=True)
    col1,col2=st.columns(2)
    with col1:
        st.plotly_chart(px.pie(res['chan'],values='sales',names='channel',hole=.4),use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(res['cat'],x='category',y='sales',color='category',showlegend=False),use_container_width=True)
    with st.expander("Raw"):
        st.dataframe(df,use_container_width=True)
    st.markdown(download_link(df,"sales.csv","Export CSV"),unsafe_allow_html=True)

# -------------------------------- AI CHAT ----------------------------------

def ai_chat():
    st.markdown('<div class="sub">AI Consultant</div>',unsafe_allow_html=True)
    if client is None:
        st.warning("OpenAI key not set");return
    for m in st.session_state.chat:
        cls="user-message" if m['role']=='user' else "ai-message"
        st.markdown(f'<div class="chat-message {cls}">{m['content']}</div>',unsafe_allow_html=True)
    prompt=st.text_input("Ask consultant")
    if st.button("Send") and prompt:
        st.session_state.chat.append(dict(role='user',content=prompt))
        with st.spinner():
            rsp=client.chat.completions.create(model="gpt-4o",temperature=0.7,max_tokens=512,messages=[{"role":"system","content":"You are a senior quality and ops consultant."}]+st.session_state.chat).choices[0].message.content
        st.session_state.chat.append(dict(role='assistant',content=rsp))
        st.experimental_rerun()
    if st.button("Clear chat"):
        st.session_state.chat=[];st.experimental_rerun()

# -------------------------------- DASHBOARD --------------------------------

def dashboard():
    st.markdown('<div class="main">Dashboard</div>',unsafe_allow_html=True)
    df=st.session_state.sales;qual=len(st.session_state.quality_hist);salv=len(st.session_state.salvage_hist)
    c1,c2,c3,c4=st.columns(4)
    with c1:
        rev=df[df['date']>=datetime.today()-timedelta(days=30)]['sales'].sum() if not df.empty else 0
        st.markdown('<div class="metric-l">30‑day revenue</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v">{fmt_cur(rev)}</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-l">Saved quality studies</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v">{qual}</div>',unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-l">Saved salvage studies</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v">{salv}</div>',unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-l">Chat messages</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metric-v">{len(st.session_state.chat)}</div>',unsafe_allow_html=True)
    if not df.empty:
        st.plotly_chart(px.line(df.groupby(df['date'].dt.date)['sales'].sum().reset_index(),x='date',y='sales'),use_container_width=True)

# -------------------------------- ROUTER -----------------------------------

def nav():
    st.sidebar.header("Navigation")
    pg=st.sidebar.radio("Go to",["Dashboard","Quality","Salvage","Sales","AI Chat"])
    if pg=="Quality":
        quality_ui()
    elif pg=="Salvage":
        salvage_ui()
    elif pg=="Sales":
        sales_ui()
    elif pg=="AI Chat":
        ai_chat()
    else:
        dashboard()

nav()
