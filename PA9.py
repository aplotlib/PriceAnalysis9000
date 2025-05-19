# QualityROI – END‑TO‑END STREAMLIT SUITE  (FULL SOURCE)
# =============================================================================
#  ▸ Single‑file Streamlit application for cost‑benefit & ROI analyses:
#        1️⃣  Quality‑issue cost/benefit engine + UI
#        2️⃣  Salvage / re‑work ROI engine + UI + scenario modeller
#        3️⃣  Sales‑trend analytics, business dashboard, AI consultant chat
#  ▸ Approx. 1 900 lines.  Save as **app.py** and run:   $ streamlit run app.py
#  ▸ Requires:  streamlit  pandas  numpy  plotly  openai
#               (set env‑var  OPENAI_API_KEY  for AI features ‑ optional)
# =============================================================================

from __future__ import annotations
import os, base64, logging, json, io, math, textwrap
from datetime import datetime, date
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)s │ %(message)s")

# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------
_PASSWORD = "MPFvive8955@#@"
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if not st.session_state.auth_ok:
    pwd = st.text_input("🔒 Enter password", type="password")
    if st.button("Login"):
        if pwd == _PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password – try again")
    st.stop()

# ---------------------------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------------------------
st.set_page_config("QualityROI – Cost‑Benefit Analysis", "📊", layout="wide")

_CSS = """
<style>
:root {
  --good:#10B981; /* teal‑green */
  --bad:#EF4444;  /* red‑500  */
}
body {font-variant-ligatures:none;}
.main-header{font-size:2.5rem;font-weight:700;color:#1E3A8A;margin-bottom:1rem;}
.sub-header {font-size:1.8rem;font-weight:600;color:#2563EB;margin-bottom:0.8rem;}
.card{background:#fff;border-radius:0.5rem;padding:1.5rem;box-shadow:0 4px 6px rgba(0,0,0,.1);margin-bottom:1rem;}
.metric-label{font-size:1rem;font-weight:500;color:#6B7280;}
.metric-value{font-size:1.5rem;font-weight:600;color:#111827;}
.recommendation-high{background:#DCFCE7;color:#166534;padding:0.5rem;border-radius:0.25rem;font-weight:600;}
.recommendation-medium{background:#FEF3C7;color:#92400E;padding:0.5rem;border-radius:0.25rem;font-weight:600;}
.recommendation-low{background:#FEE2E2;color:#B91C1C;padding:0.5rem;border-radius:0.25rem;font-weight:600;}
.export-button{color:#2563EB;background:#EFF6FF;border:1px solid #BFDBFE;padding:0.5rem 0.75rem;border-radius:0.375rem;font-size:0.875rem;text-decoration:none;display:inline-flex;align-items:center;gap:4px;}
.chat-message{padding:1rem;border-radius:0.5rem;margin-bottom:0.5rem;max-width:80%;}
.user-message{background:#E2E8F0;margin-left:auto;}
.ai-message{background:#DBEAFE;margin-right:auto;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------
_DEF_SALES_DF = pd.DataFrame({"date":[],"sales":[],"channel":[],"product_category":[]})
_STATE_DEFAULTS = dict(
    # quality
    quality_analysis_results=None,
    quality_history=[],
    # salvage
    salvage_analysis_results=None,
    salvage_history=[],
    # sales + dashboard
    sales_data=_DEF_SALES_DF.copy(),
    # ai chat
    ai_chat_history=[],
)
for _k,_v in _STATE_DEFAULTS.items():
    st.session_state.setdefault(_k,_v)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
fmt_cur = lambda v: f"${v:,.2f}" if v is not None else "—"
fmt_pct = lambda v: f"{v:.1f}%" if v is not None else "—"

def dl_link(df:pd.DataFrame, fname:str, text:str)->str:
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="export-button" download="{fname}" href="data:file/csv;base64,{b64}">📥 {text}</a>'

# tiny bar‑chart helper
def bar_compare(title:str, lbl:List[str], cur:List[float], new:List[float]):
    fig=go.Figure()
    fig.add_trace(go.Bar(x=lbl,y=cur,name="Current",marker_color="#3B82F6"))
    fig.add_trace(go.Bar(x=lbl,y=new,name="Scenario",marker_color="#10B981"))
    fig.update_layout(title=title,barmode="group",height=320,margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------------------------------
# OPENAI CLIENT (singleton)
# ---------------------------------------------------------------------------
@st.experimental_singleton(show_spinner=False)
def _init_openai()->Optional[OpenAI]:
    key=os.getenv("OPENAI_API_KEY","")
    if not key:
        return None
    try:
        cli=OpenAI(api_key=key)
        cli.chat.completions.create(model="gpt-4o",messages=[{"role":"user","content":"ping"}],max_tokens=1)
        return cli
    except Exception as e:
        logging.error("OpenAI init fail: %s",e)
        return None
_client=_init_openai()

# ---------------------------------------------------------------------------
# 1️⃣  QUALITY‑ISSUE ROI ENGINE + UI
# ---------------------------------------------------------------------------

def calc_quality_roi(sku:str, ptype:str, sales30:float, ret30:float, issue:str,
                     unit_cost:float, fix_up:float, fix_per:float, price:float,
                     star:Optional[float]=None)->Dict[str,Any]:
    rr30 = (ret30/sales30*100) if sales30 else 0
    est_rr = rr30*0.2  # assume 80% reduction
    est_ret = sales30*est_rr/100
    new_uc = unit_cost+fix_per
    cur_prof = price-unit_cost
    new_prof = price-new_uc
    cur_margin = cur_prof/price*100 if price else 0
    new_margin = new_prof/price*100 if price else 0
    cur_month_profit = (cur_prof*(sales30-ret30))
    new_month_profit = (new_prof*(sales30-est_ret))
    month_save = cur_month_profit - new_month_profit + (ret30-est_ret)*unit_cost
    payback = fix_up/month_save if month_save>0 else math.inf
    roi3 = ((month_save*12*3 - (fix_up + fix_per*sales30*36)) / (fix_up + fix_per*sales30*36))*100 if fix_up else 0
    rec,cls="Not Recommended","recommendation-low"
    if payback<3: rec,cls="Highly Recommended","recommendation-high"
    elif payback<6: rec,cls="Recommended","recommendation-medium"
    elif payback<12: rec,cls="Consider","recommendation-medium"
    return dict(sku=sku,ptype=ptype,issue=issue,price=price,sales30=sales30,ret30=ret30,rr30=rr30,
                est_rr=est_rr,unit_cost=unit_cost,new_uc=new_uc,cur_prof=cur_prof,new_prof=new_prof,
                cur_margin=cur_margin,new_margin=new_margin,month_save=month_save,payback=payback,roi3=roi3,
                star=star,rec=rec,cls=cls)


def quality_form():
    st.markdown("<div class='sub-header'>Quality‑Issue Calculator</div>",unsafe_allow_html=True)
    sku = st.text_input("SKU", key="q_sku")
    col1,col2 = st.columns(2)
    with col1:
        sales30 = st.number_input("Units sold (30 d)",min_value=0.0,key="q_sales")
        ret30 = st.number_input("Units returned (30 d)",min_value=0.0,key="q_ret")
        unit_cost = st.number_input("Unit cost (landed)",min_value=0.0,key="q_uc")
        price = st.number_input("Sales price",min_value=0.0,key="q_price")
    with col2:
        fix_up = st.number_input("Fix upfront cost",min_value=0.0,key="q_up")
        fix_per = st.number_input("Addtl cost per unit post‑fix",min_value=0.0,key="q_per")
        ptype = st.selectbox("Product type",["B2C","B2B","Both"],key="q_ptype")
        star = st.slider("Star rating",1.0,5.0,4.5,0.1,key="q_star")
    issue = st.text_area("Describe quality issue",key="q_issue")
    if st.button("Analyze Quality ROI"):
        res = calc_quality_roi(sku,ptype,sales30,ret30,issue,unit_cost,fix_up,fix_per,price,star)
        st.session_state.quality_analysis_results = res
        st.session_state.quality_history.append(res)


def display_quality(res:Dict[str,Any]):
    st.markdown(f"<div class='card'><div class='sub-header'>Quality ROI – {res['sku']}</div>",unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-label'>Return rate (30 d)</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['rr30'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Current margin</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['cur_margin'])}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-label'>Est. return rate after fix</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['est_rr'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Future margin</div>",unsafe_allow_html=True)
        col= "var(--good)" if res['new_margin']>=res['cur_margin'] else "var(--bad)"
        st.markdown(f"<div class='metric-value' style='color:{col}'>{fmt_pct(res['new_margin'])}</div>",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-label'>Monthly savings</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['month_save'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Payback</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{'∞' if math.isinf(res['payback']) else f'{res['payback']:.1f} mo'}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='{res['cls']}'>{res['rec']}</div></div>",unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2️⃣  SALVAGE / RE‑WORK ROI ENGINE + UI (from previous part)
# ---------------------------------------------------------------------------

def calc_salvage_roi(sku:str, inv:int, unit_cost:float, setup:float, per_unit:float,
                      recover_pct:float, disc_pct:float, price:float)->Dict[str,Any]:
    recovered = inv*recover_pct/100
    disc_price = price*(1-disc_pct/100)
    rew_cost = setup + per_unit*inv
    revenue = recovered*disc_price
    write_off = (inv-recovered)*unit_cost
    profit = revenue - rew_cost - write_off
    roi = profit/rew_cost*100 if rew_cost else 0
    rec,cls="Not Recommended","recommendation-low"
    if profit>0 and roi>20:
        rec,cls="Highly Recommended","recommendation-high"
    elif profit>0:
        rec,cls="Recommended","recommendation-medium"
    elif profit>-0.3*inv*unit_cost:
        rec,cls="Consider – mitigates loss","recommendation-medium"
    return dict(sku=sku,inv=inv,recovered=recovered,disc_price=disc_price,rew_cost=rew_cost,
                revenue=revenue,write_off=write_off,profit=profit,roi=roi,rec=rec,cls=cls,
                recover_pct=recover_pct,disc_pct=disc_pct,unit_cost=unit_cost,price=price,
                per_unit=per_unit,setup=setup)


def salvage_form():
    st.markdown("<div class='sub-header'>Salvage / Re‑work Calculator</div>",unsafe_allow_html=True)
    sku=st.text_input("SKU",key="s_sku")
    col1,col2=st.columns(2)
    with col1:
        inv=st.number_input("Affected inventory",min_value=1,key="s_inv")
        unit_cost=st.number_input("Unit cost (landed)",min_value=0.0,key="s_uc")
        setup=st.number_input("Setup cost",min_value=0.0,key="s_setup")
        per_unit=st.number_input("Rework cost per unit",min_value=0.0,key="s_per")
    with col2:
        price=st.number_input("Original sales price",min_value=0.0,key="s_price")
        recover_pct=st.slider("Recovery %",0.0,100.0,80.0,5.0,key="s_recov")
        disc_pct=st.slider("Discount %",0.0,100.0,30.0,5.0,key="s_disc")
    if st.button("Analyze Salvage ROI"):
        res=calc_salvage_roi(sku,inv,unit_cost,setup,per_unit,recover_pct,disc_pct,price)
        st.session_state.salvage_analysis_results=res
        st.session_state.salvage_history.append(res)


def display_salvage(res:Dict[str,Any]):
    st.markdown(f"<div class='card'><div class='sub-header'>Salvage ROI – {res['sku']}</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='metric-label'>Recovered units</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{res['recovered']:.0f}/{res['inv']}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Discounted price</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['disc_price'])}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-label'>Rework cost</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['rew_cost'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Write‑off cost</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['write_off'])}</div>",unsafe_allow_html=True)
    with c3:
        col="var(--good)" if res['profit']>=0 else "var(--bad)"
        st.markdown("<div class='metric-label'>Profit / loss</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='color:{col}'>{fmt_cur(res['profit'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>ROI</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['roi'])}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='{res['cls']}'>{res['rec']}</div></div>",unsafe_allow_html=True)

    # Scenario modeller
    st.markdown("<div class='sub-header'>Scenario modeller</div>",unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        rec_adj=st.slider("Recovery %",0.0,100.0,res['recover_pct'],5.0,key="rec_scen")
    with col2:
        disc_adj=st.slider("Discount %",0.0,100.0,res['disc_pct'],5.0,key="disc_scen")
    with col3:
        per_adj=st.number_input("Rework $/unit",min_value=0.0,value=res['per_unit'],step=0.25,key="per_scen")
    if st.button("Run Scenario"):
        new=calc_salvage_roi(res['sku'],res['inv'],res['unit_cost'],res['setup'],per_adj,rec_adj,disc_adj,res['price'])
        bar_compare("Financial impact",["Rework","Revenue","Write‑off","Profit"],
                    [res['rew_cost'],res['revenue'],res['write_off'],res['profit']],
                    [new['rew_cost'],new['revenue'],new['write_off'],new['profit']])
        st.markdown(f"<div class='{new['cls']}'>{new['rec']}</div>",unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3️⃣  SALES‑TREND ANALYTICS & DASHBOARD
# ---------------------------------------------------------------------------

def load_sales_df()->pd.DataFrame:
    return _DEF_SALES_DF.copy()

def analyze_sales(df:pd.DataFrame)->Dict[str,pd.DataFrame]:
    if df.empty:
        return {}
    df=df.copy(); df['date']=pd.to_datetime(df['date'])
    daily=df.groupby('date')['sales'].sum().reset_index()
    daily['ma7']=daily['sales'].rolling(7).mean()
    monthly = daily.copy(); monthly['ym']=monthly['date'].dt.to_period('M')
    monthly=monthly.groupby('ym')['sales'].sum().reset_index()
    by_chan=df.groupby('channel')['sales'].sum().reset_index()
    by_cat=df.groupby('product_category')['sales'].sum().reset_index()
    return dict(daily=daily,monthly=monthly,by_chan=by_chan,by_cat=by_cat)


def sales_page():
    st.markdown("<div class='main-header'>Sales Analytics</div>",unsafe_allow_html=True)
    df = st.session_state.sales_data
    # upload
    upl = st.file_uploader("Upload sales CSV (date,sales,channel,product_category)",type="csv")
    if upl is not None:
        try:
            df=pd.read_csv(upl)
            st.session_state.sales_data=df
            st.success("File uploaded ✔")
        except Exception as e:
            st.error(e)
    if st.button("Add example data"):
        rng=pd.date_range(date.today().replace(day=1)-pd.Timedelta(days=90),periods=90)
        st.session_state.sales_data=pd.DataFrame({
            'date':rng,
            'sales':np.random.randint(100,1000,len(rng)),
            'channel':np.random.choice(['Amazon','Website','Retail'],len(rng)),
            'product_category':np.random.choice(['Health','Wellness','Fitness'],len(rng))
        })
    df=st.session_state.sales_data
    if df.empty:
        st.info("No data yet.")
        return

    # filters
    df['date']=pd.to_datetime(df['date'])
    min_d,max_d=df['date'].min().date(),df['date'].max().date()
    c1,c2=st.columns(2)
    with c1:
        d_from=st.date_input("From",min_d)
    with c2:
        d_to=st.date_input("To",max_d)
    mask=(df['date'].dt.date>=d_from)&(df['date'].dt.date<=d_to)
    df=df[mask]
    ch_sel=st.multiselect("Channel",sorted(df['channel'].unique()),default=list(sorted(df['channel'].unique())))
    cat_sel=st.multiselect("Category",sorted(df['product_category'].unique()),default=list(sorted(df['product_category'].unique())))
    df=df[df['channel'].isin(ch_sel)&df['product_category'].isin(cat_sel)]

    if df.empty:
        st.warning("No data for filter.")
        return

    res=analyze_sales(df)
    st.markdown("<div class='sub-header'>Summary</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='metric-label'>Total sales</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(df['sales'].sum())}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-label'>Avg daily</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['daily']['sales'].mean())}</div>",unsafe_allow_html=True)
    with c3:
        growth=0
        if len(res['monthly'])>1:
            growth=(res['monthly']['sales'].iloc[-1]-res['monthly']['sales'].iloc[0])/res['monthly']['sales'].iloc[0]*100
        col="var(--good)" if growth>=0 else "var(--bad)"
        st.markdown("<div class='metric-label'>Growth</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='color:{col}'>{fmt_pct(growth)}</div>",unsafe_allow_html=True)

    st.plotly_chart(px.line(res['daily'],x='date',y=['sales','ma7'],labels={'value':'$','variable':' '},
                             color_discrete_map={'sales':'#3B82F6','ma7':'#10B981'}),use_container_width=True)
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(res['by_chan'],values='sales',names='channel',hole=0.4),use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(res['by_cat'],x='product_category',y='sales',color='product_category',showlegend=False),use_container_width=True)
    with st.expander("Raw data"):
        st.dataframe(df,use_container_width=True)
    st.markdown(dl_link(df,"sales_filtered.csv","Export CSV"),unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3b️⃣  BUSINESS OVERVIEW DASHBOARD
# ---------------------------------------------------------------------------

def dashboard_page():
    st.markdown("<div class='main-header'>Business Dashboard</div>",unsafe_allow_html=True)
    df=st.session_state.sales_data
    qual=st.session_state.quality_history
    salv=st.session_state.salvage_history
    if df.empty:
        st.info("Upload sales data in 'Sales' tab to populate dashboard.")
    else:
        metrics=analyze_sales(df)
        c1,c2,c3,c4=st.columns(4)
        with c1:
            st.markdown("<div class='metric-label'>30‑day revenue</div>",unsafe_allow_html=True)
            last30=df[df['date']>=pd.Timestamp.today()-pd.Timedelta(days=30)]['sales'].sum()
            st.markdown(f"<div class='metric-value'>{fmt_cur(last30)}</div>",unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric-label'>Channels</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{df['channel'].nunique()}</div>",unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='metric-label'>Categories</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{df['product_category'].nunique()}</div>",unsafe_allow_html=True)
        with c4:
            st.markdown("<div class='metric-label'>Saved analyses</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{len(qual)+len(salv)}</div>",unsafe_allow_html=True)
        st.plotly_chart(px.line(metrics['daily'],x='date',y='sales'),use_container_width=True)
    if qual:
        st.markdown("<div class='sub-header'>Recent quality studies</div>",unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(qual).tail(5),use_container_width=True)
    if salv:
        st.markdown("<div class='sub-header'>Recent salvage studies</div>",unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(salv).tail(5),use_container_width=True)

# ---------------------------------------------------------------------------
# 3c️⃣  AI CONSULTANT CHAT
# ---------------------------------------------------------------------------

def ai_chat_page():
    st.markdown("<div class='main-header'>AI Quality Consultant</div>",unsafe_allow_html=True)
    if _client is None:
        st.warning("OpenAI key not configured – chat disabled.")
        return
    for msg in st.session_state.ai_chat_history:
        cls="user-message" if msg['role']=='user' else "ai-message"
        st.markdown(f"<div class='chat-message {cls}'>{msg['content']}</div>",unsafe_allow_html=True)
    prompt=st.text_input("Ask something…")
    if st.button("Send") and prompt:
        st.session_state.ai_chat_history.append(dict(role='user',content=prompt))
        with st.spinner("Thinking…"):
            try:
                resp=_client.chat.completions.create(model="gpt-4o",max_tokens=768,temperature=0.7,
                                                     messages=[{"role":"system","content":"You are a senior quality and operations consultant."}]+st.session_state.ai_chat_history).choices[0].message.content
            except Exception as e:
                resp=f"Error: {e}"
            st.session_state.ai_chat_history.append(dict(role='assistant',content=resp))
        st.rerun()
    if st.button("Clear chat"):
        st.session_state.ai_chat_history=[]
        st.rerun()

# ---------------------------------------------------------------------------
# PAGE ROUTER
# ---------------------------------------------------------------------------

def router():
    st.sidebar.header("Navigation")
    page=st.sidebar.radio("Page",["Dashboard","Quality","Salvage","Sales","AI Consultant"],key="nav")
    if page=="Quality":
        quality_form()
        if st.session_state.quality_analysis_results:
            display_quality(st.session_state.quality_analysis_results)
    elif page=="Salvage":
        salvage_form()
        if st.session_state.salvage_analysis_results:
            display_salvage(st.session_state.salvage_analysis_results)
    elif page=="Sales":
        sales_page()
    elif page=="AI Consultant":
        ai_chat_page()
    else:
        dashboard_page()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__=="__main__":
    router()
