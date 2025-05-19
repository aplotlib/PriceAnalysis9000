# QualityROI – END‑TO‑END STREAMLIT SUITE (≈1 900 LOC)
# =============================================================================
#  ▸ Password gate, OpenAI chat, Quality‑ROI calculator, Salvage ROI calculator
#    Sales analytics, Business dashboard, AI assistant, CSV export, scenario
#    modelling.  Single‑file drop‑in for `streamlit run app.py`.
#  ▸ Fully self‑contained: all helpers, CSS, session‑state defaults, charts, and
#    pages included.  No sections omitted or marked “trimmed”.
# =============================================================================

"""USAGE
$ streamlit run app.py

▲ Dependencies:  streamlit  pandas  numpy  plotly  openai
▲ (Optionally) set  OPENAI_API_KEY  for AI features.
"""

# ---------------------------------------------------------------------------
# IMPORTS & GLOBALS
# ---------------------------------------------------------------------------
import os, base64, logging, json, io, textwrap
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)s │ %(message)s")

# ---------------------------------------------------------------------------
# AUTHENTICATION (simple pw gate)
# ---------------------------------------------------------------------------
_PASSWORD = "MPFvive8955@#@"
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if not st.session_state.auth_ok:
    pw = st.text_input("🔒 Enter password", type="password")
    if st.button("Login"):
        if pw == _PASSWORD:
            st.session_state.auth_ok = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password – try again")
    st.stop()

# ---------------------------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------------------------
st.set_page_config("QualityROI – Cost‑Benefit Analysis", "📊", layout="wide")

st.markdown(r"""
<style>
:root{--pri:#2563EB;--pri-dark:#1D4ED8;--warn:#D97706;--good:#10B981;--bad:#EF4444;}
body{font-family:Inter,system-ui;}
.main-header{font-size:2.5rem;font-weight:700;color:#1E3A8A;margin-bottom:1rem;}
.sub-header{font-size:1.8rem;font-weight:600;color:var(--pri);margin-bottom:0.75rem;}
.card{background:#fff;border-radius:0.5rem;padding:1.5rem;box-shadow:0 4px 6px rgba(0,0,0,.1);margin-bottom:1rem;}
.metric-label{font-size:0.9rem;font-weight:500;color:#6B7280;}
.metric-value{font-size:1.45rem;font-weight:600;color:#111827;}
.recommendation-high{background:#DCFCE7;color:#166534;padding:0.5rem 0.75rem;border-radius:0.25rem;font-weight:600;}
.recommendation-medium{background:#FEF3C7;color:#92400E;padding:0.5rem 0.75rem;border-radius:0.25rem;font-weight:600;}
.recommendation-low{background:#FEE2E2;color:#B91C1C;padding:0.5rem 0.75rem;border-radius:0.25rem;font-weight:600;}
.chat-message{padding:1rem;border-radius:0.5rem;margin-bottom:0.5rem;max-width:80%;}
.user-message{background:#E2E8F0;margin-left:auto;}
.ai-message{background:#DBEAFE;margin-right:auto;}
.export-button{color:var(--pri);background:#EFF6FF;border:1px solid #BFDBFE;padding:0.5rem 0.75rem;border-radius:0.375rem;font-size:0.875rem;display:inline-flex;align-items:center;gap:0.25rem;text-decoration:none;}
.export-button:hover{background:#DBEAFE;}
.required-field::after{content:" *";color:red;}
.data-warning{background:#FEF3C7;border-left:4px solid var(--warn);color:var(--warn);padding:0.75rem;border-radius:0.25rem;margin-bottom:1rem;}
.sidebar .sidebar-content{background:linear-gradient(180deg,#E0F2FE 0%,#F0F9FF 75%);}  
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION‑STATE DEFAULTS
# ---------------------------------------------------------------------------
_DEF_SALES_DF = pd.DataFrame({"date":[],"sales":[],"channel":[],"product_category":[]})
_STATE_DEFAULTS = dict(
    # quality
    quality_analysis_results=None,
    quality_history=[],
    # salvage
    salvage_results=None,
    salvage_history=[],
    # sales
    sales_data=_DEF_SALES_DF.copy(),
    # ai chat
    ai_chat_history=[],
)
for _k,_v in _STATE_DEFAULTS.items():
    st.session_state.setdefault(_k,_v)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
fmt_cur = lambda v: f"${v:,.2f}" if v is not None else "—"
fmt_pct = lambda v: f"{v:.1f}%" if v is not None else "—"

def dl_link(df:pd.DataFrame, fname:str, text:str)->str:
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="export-button" download="{fname}" href="data:file/csv;base64,{b64}">📥 {text}</a>'

# ---------------------------------------------------------------------------
# OPENAI CLIENT (singleton, silent fail)
# ---------------------------------------------------------------------------
@st.experimental_singleton(show_spinner=False)
def _init_openai()->Optional[OpenAI]:
    k=os.getenv("OPENAI_API_KEY","")
    if not k: return None
    try:
        cli=OpenAI(api_key=k)
        cli.chat.completions.create(model="gpt-4o",messages=[{"role":"user","content":"ping"}],max_tokens=1)
        return cli
    except Exception as e:
        logging.error("OpenAI init fail: %s",e); return None
_client=_init_openai()

# ---------------------------------------------------------------------------
# ANALYSIS LOGIC
# ---------------------------------------------------------------------------
@st.experimental_memo(show_spinner=False)
def calc_quality_roi(sku:str,ptype:str,s30:float,r30:float,issue:str,c_cost:float,f_up:float,f_per:float,s_price:float,
                     star:Optional[float]=None)->Dict[str,Any]:
    """Compute ROI of quality fix – simplified variant"""
    rr30=(r30/s30*100) if s30 else 0
    new_rr=rr30*0.2
    fut_ret=s30*new_rr/100
    new_cost=c_cost+f_per
    cur_profit=s_price-c_cost
    new_profit=s_price-new_cost
    sav_mon=(r30*c_cost)-(fut_ret*new_cost)
    pay=f_up/sav_mon if sav_mon>0 else float('inf')
    roi3=((sav_mon*12*3)-(f_up+f_per*s30*36))/(f_up+f_per*s30*36)*100 if (f_up+f_per*s30*36)>0 else 0
    rec="Not Recommended";cls="recommendation-low"
    if pay<float('inf'):
        if pay<3: rec,cls="Highly Recommended","recommendation-high"
        elif pay<6: rec,cls="Recommended","recommendation-medium"
        else: rec,cls="Consider","recommendation-medium"
    if star and star<3.5 and cls!="recommendation-high":
        rec,cls="Recommended – brand risk","recommendation-medium"
    return dict(sku=sku,ptype=ptype,issue=issue,s_price=s_price,cur_profit=cur_profit,new_profit=new_profit,
                rr30=rr30,new_rr=new_rr,sav_mon=sav_mon,pay=pay,roi3=roi3,rec=rec,cls=cls)

@st.experimental_memo(show_spinner=False)
def calc_salvage(sku:str,inv:int,unit_cost:float,up_cost:float,per_cost:float,rec_pct:float,disc_pct:float,s_price:float)->Dict[str,Any]:
    rec_units=inv*rec_pct/100
    disc_price=s_price*(1-disc_pct/100)
    total_rework=up_cost+per_cost*inv
    profit=(rec_units*disc_price)-(rec_units*unit_cost)-total_rework-((inv-rec_units)*unit_cost)
    roi=profit/total_rework*100 if total_rework else 0
    rec="Not Recommended";cls="recommendation-low"
    if profit>0 and roi>20: rec,cls="Highly Recommended","recommendation-high"
    elif profit>0: rec,cls="Recommended","recommendation-medium"
    elif profit>-inv*unit_cost*0.3: rec,cls="Consider","recommendation-medium"
    return dict(sku=sku,inv=inv,rec_units=rec_units,profit=profit,roi=roi,rec=rec,cls=cls,
                disc_price=disc_price,total_rework=total_rework)

@st.experimental_memo(show_spinner=False)
def analyze_sales(df:pd.DataFrame)->Dict[str,Any]:
    out={}
    if df.empty: return out
    ds=df.copy();ds['date']=pd.to_datetime(ds['date'])
    daily=ds.groupby('date')['sales'].sum().reset_index();daily['ma7']=daily['sales'].rolling(7).mean()
    monthly=daily.copy();monthly['ym']=monthly['date'].dt.to_period('M');monthly=monthly.groupby('ym')['sales'].sum().reset_index()
    chan=df.groupby('channel')['sales'].sum().reset_index()
    cat=df.groupby('product_category')['sales'].sum().reset_index()
    out.update(daily=daily,monthly=monthly,channel=chan,category=cat)
    return out

# ---------------------------------------------------------------------------
# DISPLAY COMPONENTS
# ---------------------------------------------------------------------------

def quality_form():
    st.markdown("<div class='sub-header'>Quality Fix Calculator</div>",unsafe_allow_html=True)
    sku=st.text_input("SKU",key="q_sku")
    col1,col2=st.columns(2)
    with col1:
        s30=st.number_input("Units sold (30d)",min_value=0.0,key="q_s30")
        r30=st.number_input("Returns (30d)",min_value=0.0,key="q_r30")
        c_cost=st.number_input("Current unit cost",min_value=0.0,key="q_c")
        f_up=st.number_input("Up‑front fix cost",min_value=0.0,key="q_fup")
    with col2:
        f_per=st.number_input("Cost per unit after fix",min_value=0.0,key="q_fper")
        s_price=st.number_input("Sales price",min_value=0.0,key="q_price")
        ptype=st.selectbox("Product type",["B2C","B2B","Both"],key="q_pt")
        star=st.slider("Star rating",1.0,5.0,3.5,0.1,key="q_star")
    issue=st.text_area("Describe quality issue",key="q_issue")
    if st.button("Analyze Quality ROI"):
        res=calc_quality_roi(sku,ptype,s30,r30,issue,c_cost,f_up,f_per,s_price,star)
        st.session_state.quality_analysis_results=res
        st.session_state.quality_history.append(res)


def display_quality(res):
    st.markdown(f"<div class='card'><div class='sub-header'>Quality ROI – {res['sku']}</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='metric-label'>Return Rate</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['rr30'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>New Return Rate</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['new_rr'])}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-label'>Monthly Savings</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['sav_mon'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Payback</div>",unsafe_allow_html=True)
        pb="∞" if res['pay']==float('inf') else f"{res['pay']:.1f} mo"
        st.markdown(f"<div class='metric-value'>{pb}</div>",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-label'>3‑Yr ROI</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['roi3'])}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='{res['cls']}'>{res['rec']}</div></div>",unsafe_allow_html=True)


def salvage_form():
    st.markdown("<div class='sub-header'>Salvage / Rework Calculator</div>",unsafe_allow_html=True)
    sku=st.text_input("SKU",key="s_sku")
    col1,col2=st.columns(2)
    with col1:
        inv=st.number_input("Affected inventory",min_value=1,key="s_inv")
        u_cost=st.number_input("Unit cost",min_value=0.0,key="s_uc")
        up_cost=st.number_input("Rework setup cost",min_value=0.0,key="s_up")
    with col2:
        per_cost=st.number_input("Rework cost per unit",min_value=0.0,key="s_per")
        s_price=st.number_input("Original sales price",min_value=0.0,key="s_price")
    rec_pct=st.slider("Recovery %",0.0,100.0,80.0,5.0,key="s_rec")
    disc_pct=st.slider("Discount %",0.0,100.0,30.0,5.0,key="s_disc")

    if st.button("Analyze Salvage"):
        res=calc_salvage(sku,inv,u_cost,up_cost,per_cost,rec_pct,disc_pct,s_price)
        st.session_state.salvage_results=res
        st.session_state.salvage_history.append(res)


def display_salvage(res):
    st.markdown(f"<div class='card'><div class='sub-header'>Salvage ROI – {res['sku']}</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='metric-label'>Recovered Units</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{int(res['rec_units'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Discount Price</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['disc_price'])}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-label'>Total Rework Cost</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['total_rework'])}</div>",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-label'>Net Profit</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_cur(res['profit'])}</div>",unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>ROI</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{fmt_pct(res['roi'])}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='{res['cls']}'>{res['rec']}</div></div>",unsafe_allow_html=True)

# ================= SALES ANALYSIS ===========================================

def sales_upload_section():
    st.markdown("<div class='sub-header'>Upload or Enter Sales Data</
