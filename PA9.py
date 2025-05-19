from __future__ import annotations
import os, base64, logging, math, uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
PASSWORD = "MPFvive8955@#@"
APP_TITLE = "QualityROI Pro"
APP_ICON = "📊"
THEMES = {
    "Light": {
        "primary": "#2563EB",
        "secondary": "#1E40AF",
        "bg": "#FFFFFF",
        "text": "#111827"
    },
    "Dark": {
        "primary": "#7DD3FC",
        "secondary": "#0284C7",
        "bg": "#0F172A",
        "text": "#F8FAFC"
    }
}

st.set_page_config(APP_TITLE, APP_ICON, layout="wide", initial_sidebar_state="expanded")

# ────────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ────────────────────────────────────────────────────────────────────────────────

def init_state():
    d: Dict[str, Any] = dict(
        auth=False,
        theme="Light",
        quality_results=None,
        salvage_results=None,
        sales_data=pd.DataFrame(columns=["date", "sales", "channel", "product_category"]),
        chat_history=[],
        quality_history=[],
        salvage_history=[]
    )
    for k, v in d.items():
        st.session_state.setdefault(k, v)

init_state()

# ────────────────────────────────────────────────────────────────────────────────
# STYLE
# ────────────────────────────────────────────────────────────────────────────────

def apply_theme():
    t = THEMES[st.session_state.theme]
    st.markdown(f"""
        <style>
            :root {{
                --pr: {t['primary']};
                --pr-dark: {t['secondary']};
                --bg: {t['bg']};
                --tx: {t['text']};
            }}
            body, .stApp {{ background: var(--bg); color: var(--tx); }}
            .main-header{{font-size:2.4rem;font-weight:700;color:var(--pr-dark);margin-bottom:1.2rem;}}
            .sub-header{{font-size:1.6rem;font-weight:600;color:var(--pr);margin-bottom:0.8rem;}}
            .card{{background:var(--bg);border:1px solid var(--pr-dark);border-radius:0.5rem;padding:1.2rem;}}
            .metric-label{{font-size:0.9rem;font-weight:500;opacity:0.8;}}
            .metric-value{{font-size:1.4rem;font-weight:600;}}
            .btn{{background:var(--pr);color:#fff;font-weight:600;padding:0.4rem 1rem;border:none;border-radius:0.3rem;}}
            .btn:hover{{background:var(--pr-dark);}}
            .recommendation-high{{background:#22c55e20;border-left:6px solid #22C55E;padding:0.5rem;}}
            .recommendation-medium{{background:#facc151f;border-left:6px solid #FACC15;padding:0.5rem;}}
            .recommendation-low{{background:#ef44441f;border-left:6px solid #EF4444;padding:0.5rem;}}
            .chat-bubble-user{{background:#0ea5e945;color:#000;border-radius:0.5rem 0.5rem 0 0.5rem;padding:0.6rem;max-width:80%;margin-left:auto;}}
            .chat-bubble-ai{{background:#6ee7b73b;border-radius:0.5rem 0.5rem 0.5rem 0;padding:0.6rem;max-width:80%;margin-right:auto;}}
        </style>
    """, unsafe_allow_html=True)

apply_theme()

# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────

fmt_cur = lambda v: f"${v:,.2f}" if v is not None else "—"
fmt_pct = lambda v: f"{v:.1f}%" if v is not None else "—"

def download_link(df: pd.DataFrame, fname: str) -> str:
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="btn" href="data:file/csv;base64,{b64}" download="{fname}">Download CSV</a>'

# ────────────────────────────────────────────────────────────────────────────────
# AUTH
# ────────────────────────────────────────────────────────────────────────────────

def auth_gate():
    if st.session_state.auth:
        return True
    st.title(APP_TITLE)
    st.subheader("Login")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter", key="login_btn"):
        if pwd == PASSWORD:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False

if not auth_gate():
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# OPENAI
# ────────────────────────────────────────────────────────────────────────────────
@st.singleton(show_spinner=False)
def openai_client():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        cli = OpenAI(api_key=key)
        cli.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"ping"}], max_tokens=1)
        return cli
    except Exception as e:
        logging.error(e)
        return None

ai = openai_client()

# ────────────────────────────────────────────────────────────────────────────────
# QUALITY ROI ENGINE
# ────────────────────────────────────────────────────────────────────────────────

def quality_calc(sku:str, sales:int, returns:int, price:float, cost:float, fix_up:float, fix_add:float):
    rr = returns / sales * 100 if sales else 0
    est_rr = rr * 0.2
    est_ret = sales * est_rr/100
    cur_prof = price - cost
    new_prof = price - (cost+fix_add)
    cur_margin = cur_prof/price*100 if price else 0
    new_margin = new_prof/price*100 if price else 0
    monthly_gain = (cur_prof*(sales-returns)) - (new_prof*(sales-est_ret))
    payback = fix_up/monthly_gain if monthly_gain>0 else math.inf
    cls, rec = ("recommendation-low","Not Recommended")
    if payback<3: cls,rec=("recommendation-high","Highly Recommended")
    elif payback<6: cls,rec=("recommendation-medium","Recommended")
    roi3 = ((monthly_gain*12*3) - (fix_up + fix_add*sales*36)) / (fix_up + fix_add*sales*36) * 100 if fix_up else 0
    return dict(sku=sku, rr=rr, est_rr=est_rr, cur_margin=cur_margin, new_margin=new_margin,
                monthly_gain=monthly_gain, payback=payback, cls=cls, rec=rec, roi3=roi3)

# ────────────────────────────────────────────────────────────────────────────────
# SALVAGE ROI ENGINE
# ────────────────────────────────────────────────────────────────────────────────

def salvage_calc(sku:str, inv:int, price:float, cost:float, setup:float, per:float, rec_pct:float, disc_pct:float):
    rec_units = inv*rec_pct/100
    disc_price = price*(1-disc_pct/100)
    revenue = rec_units*disc_price
    rework_cost = setup + inv*per
    write_off = (inv-rec_units)*cost
    profit = revenue - rework_cost - write_off
    roi = profit/rework_cost*100 if rework_cost else 0
    cls,rec=("recommendation-low","Not Recommended")
    if profit>0 and roi>20: cls,rec=("recommendation-high","Highly Recommended")
    elif profit>0: cls,rec=("recommendation-medium","Recommended")
    elif profit>-0.3*inv*cost: cls,rec=("recommendation-medium","Consider")
    return dict(sku=sku, revenue=revenue, profit=profit, roi=roi, cls=cls, rec=rec,
                rec_units=rec_units, disc_price=disc_price, rework_cost=rework_cost, write_off=write_off)

# ────────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ────────────────────────────────────────────────────────────────────────────────

def metric_block(label:str,val:str):
    st.markdown(f"<div class='metric-label'>{label}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{val}</div>",unsafe_allow_html=True)

def quality_ui():
    st.markdown("<div class='sub-header'>Quality ROI</div>",unsafe_allow_html=True)
    sku = st.text_input("SKU")
    c1,c2 = st.columns(2)
    with c1:
        sales = st.number_input("Sales (30d)",min_value=1)
        returns = st.number_input("Returns (30d)",min_value=0)
        price = st.number_input("Price",min_value=0.01)
    with c2:
        cost = st.number_input("Unit cost",min_value=0.01)
        fix_up = st.number_input("Fix upfront",min_value=0.0)
        fix_add = st.number_input("Add cost/unit",min_value=0.0)
    if st.button("Calculate", key=str(uuid.uuid4())):
        res = quality_calc(sku,sales,returns,price,cost,fix_up,fix_add)
        st.session_state.quality_results = res
        st.session_state.quality_history.append(res)
    res = st.session_state.quality_results
    if res:
        st.markdown(f"<div class='card {res['cls']}'>"+res['rec']+"</div>",unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            metric_block("Return rate", fmt_pct(res['rr']))
            metric_block("Est. return", fmt_pct(res['est_rr']))
        with c2:
            metric_block("Current margin", fmt_pct(res['cur_margin']))
            metric_block("New margin", fmt_pct(res['new_margin']))
        with c3:
            metric_block("Monthly gain", fmt_cur(res['monthly_gain']))
            metric_block("Payback", "∞" if math.isinf(res['payback']) else f"{res['payback']:.1f} mo")
        st.markdown(metric_chart([res['cur_margin'],res['new_margin']],["Current","New"],"Margin %"), unsafe_allow_html=True)

def metric_chart(values:List[float], labels:List[str], title:str):
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#3B82F6","#10B981"]))
    fig.update_layout(title=title,height=260,margin=dict(l=20,r=20,t=40,b=20))
    return st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme=None)

def salvage_ui():
    st.markdown("<div class='sub-header'>Salvage ROI</div>",unsafe_allow_html=True)
    sku = st.text_input("SKU", key="salv_sku")
    c1,c2 = st.columns(2)
    with c1:
        inv = st.number_input("Affected inventory",min_value=1)
        price = st.number_input("Price",min_value=0.01)
        cost = st.number_input("Unit cost",min_value=0.01)
    with c2:
        setup = st.number_input("Setup cost",min_value=0.0)
        per = st.number_input("Rework per unit",min_value=0.0)
        rec_pct = st.slider("Recovery %",0.0,100.0,80.0,5.0)
        disc_pct = st.slider("Discount %",0.0,100.0,30.0,5.0)
    if st.button("Calculate salvage"):
        res = salvage_calc(sku,inv,price,cost,setup,per,rec_pct,disc_pct)
        st.session_state.salvage_results = res
        st.session_state.salvage_history.append(res)
    res = st.session_state.salvage_results
    if res:
        st.markdown(f"<div class='card {res['cls']}'>"+res['rec']+"</div>",unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            metric_block("Recovered", f"{res['rec_units']:.0f}/{inv}")
            metric_block("Disc price", fmt_cur(res['disc_price']))
        with c2:
            metric_block("Rework cost", fmt_cur(res['rework_cost']))
            metric_block("Write‑off", fmt_cur(res['write_off']))
        with c3:
            metric_block("Profit", fmt_cur(res['profit']))
            metric_block("ROI", fmt_pct(res['roi']))
        st.markdown(metric_chart([res['rework_cost'],res['profit']],["Cost","Profit"],"Profitability"), unsafe_allow_html=True)

def sales_ui():
    st.markdown("<div class='sub-header'>Sales Analytics</div>",unsafe_allow_html=True)
    upl = st.file_uploader("Upload CSV", type="csv")
    if upl is not None:
        st.session_state.sales_data = pd.read_csv(upl)
    if st.session_state.sales_data.empty:
        st.info("No data")
        return
    df = st.session_state.sales_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    res = df.groupby('date')['sales'].sum().reset_index()
    res['MA7'] = res['sales'].rolling(7).mean()
    st.plotly_chart(px.line(res,x='date',y=['sales','MA7'],labels={'value':'Sales','variable':''}), use_container_width=True)
    st.markdown(download_link(df,"sales.csv"), unsafe_allow_html=True)

def dashboard_ui():
    st.markdown("<div class='sub-header'>Dashboard</div>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### Quality Studies")
        if st.session_state.quality_history:
            st.dataframe(pd.DataFrame(st.session_state.quality_history))
    with col2:
        st.markdown("### Salvage Studies")
        if st.session_state.salvage_history:
            st.dataframe(pd.DataFrame(st.session_state.salvage_history))

def ai_chat_ui():
    st.markdown("<div class='sub-header'>AI Consultant</div>",unsafe_allow_html=True)
    if ai is None:
        st.warning("No OpenAI key")
        return
    for m in st.session_state.chat_history:
        cls = 'chat-bubble-user' if m['role']=='user' else 'chat-bubble-ai'
        st.markdown(f"<div class='{cls}'>{m['content']}</div>", unsafe_allow_html=True)
    q = st.text_input("Ask…")
    if st.button("Send") and q:
        st.session_state.chat_history.append(dict(role="user", content=q))
        with st.spinner("AI thinking"):
            msg = ai.chat.completions.create(model="gpt-4o", messages=[{"role":"system","content":"Professional quality & ops advisor"}]+st.session_state.chat_history, max_tokens=512, temperature=0.7).choices[0].message.content
        st.session_state.chat_history.append(dict(role="assistant", content=msg))
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# ROUTER
# ────────────────────────────────────────────────────────────────────────────────

def router():
    st.sidebar.title(APP_TITLE)
    st.sidebar.selectbox("Theme", options=list(THEMES.keys()), key="theme", on_change=apply_theme)
    page = st.sidebar.radio("Navigate", ["Dashboard","Quality ROI","Salvage ROI","Sales","AI"], key="nav")
    if page=="Quality ROI": quality_ui()
    elif page=="Salvage ROI": salvage_ui()
    elif page=="Sales": sales_ui()
    elif page=="AI": ai_chat_ui()
    else: dashboard_ui()

router()
