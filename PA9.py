import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import io
import json
import base64
import uuid
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import traceback
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Medical Device Quality Analysis", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
# --- THEME COLORS ---
PRIMARY_COLOR   = "#0096C7"
SECONDARY_COLOR = "#48CAE4"
TERTIARY_COLOR  = "#90E0EF"
BACKGROUND_COLOR= "#F8F9FA"
CARD_BACKGROUND = "#FFFFFF"
TEXT_PRIMARY    = "#212529"
TEXT_SECONDARY  = "#6C757D"
TEXT_MUTED      = "#ADB5BD"
SUCCESS_COLOR   = "#40916C"
WARNING_COLOR   = "#E9C46A"
DANGER_COLOR    = "#E76F51"
BORDER_COLOR    = "#DEE2E6"

# --- SESSION STATE INITIALIZATION ---
if 'quality_analysis_results' not in st.session_state:
    st.session_state.quality_analysis_results = None
if 'analysis_submitted' not in st.session_state:
    st.session_state.analysis_submitted = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'standalone_chat_history' not in st.session_state:
    st.session_state.standalone_chat_history = []
if 'monte_carlo_chat_history' not in st.session_state:
    st.session_state.monte_carlo_chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "analysis"
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "basic"
if 'batch_analysis_results' not in st.session_state:
    st.session_state.batch_analysis_results = {}
if 'monte_carlo_scenario' not in st.session_state:
    st.session_state.monte_carlo_scenario = None
if 'compare_list' not in st.session_state:
    st.session_state.compare_list = []
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = None
if 'selected_device_type' not in st.session_state:
    st.session_state.selected_device_type = "Generic"
if 'root_cause_analysis' not in st.session_state:
    st.session_state.root_cause_analysis = {}
if 'process_control_data' not in st.session_state:
    st.session_state.process_control_data = {}
if 'pareto_data' not in st.session_state:
    st.session_state.pareto_data = {}

# --- CSS INJECTION ---
st.markdown(f"""
<style>
    :root {{ --primary: {PRIMARY_COLOR}; --secondary: {SECONDARY_COLOR}; --tertiary: {TERTIARY_COLOR}; 
             --background: {BACKGROUND_COLOR}; --card-bg: {CARD_BACKGROUND}; --text-primary: {TEXT_PRIMARY}; 
             --text-secondary: {TEXT_SECONDARY}; --text-muted: {TEXT_MUTED}; --success: {SUCCESS_COLOR}; 
             --warning: {WARNING_COLOR}; --danger: {DANGER_COLOR}; --border: {BORDER_COLOR}; }}
    .metric-card {{ 
        background-color: var(--card-bg); 
        border-radius: 8px; 
        padding: 1rem; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        margin-bottom: 1rem; 
        position: relative;
        border-left: 3px solid var(--primary);
    }}
    .metric-card:hover::after {{
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(0,0,0,0.8);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 99;
        opacity: 0.9;
    }}
    .metric-label {{ font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.25rem; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 600; color: var(--text-primary); }}
    .metric-subvalue {{ font-size: 0.8rem; color: var(--text-muted); }}
    .badge {{ padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
    .badge-success {{ background-color: var(--success); color: white; }}
    .badge-warning {{ background-color: var(--warning); color: white; }}
    .badge-danger {{ background-color: var(--danger); color: white; }}
    .assistant-bubble {{ background:white; border-left:3px solid var(--primary); padding:0.75rem; 
                         margin-bottom: 0.5rem; border-radius: 0 4px 4px 0; }}
    .user-bubble {{ background: {TERTIARY_COLOR}; padding:0.75rem; margin-left:auto; 
                    margin-bottom: 0.5rem; border-radius: 4px 0 0 4px; max-width: 80%; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 1rem; }}
    .stTabs [data-baseweb="tab"] {{ 
        height: 3rem; 
        white-space: pre-wrap; 
        background-color: white;
        border-radius: 4px 4px 0 0; 
        gap: 0.5rem; 
        padding-top: 0.5rem; 
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {PRIMARY_COLOR} !important; 
        color: white !important; 
        font-weight: 600;
    }}
    div.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    .export-button {{ 
        background-color: var(--primary); 
        color: white; 
        padding: 0.5rem 1rem;
        border-radius: 4px; 
        text-decoration: none; 
        display: inline-block; 
        margin-right: 0.5rem;
        transition: background-color 0.3s ease;
    }}
    .export-button:hover {{ 
        background-color: var(--secondary); 
        text-decoration: none;
        color: white;
    }}
    .stButton > button {{
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: var(--secondary);
    }}
    /* Tooltip styling */
    .tooltip {{
        position: relative;
        display: inline-block;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    /* Form styling */
    .stNumberInput input, .stTextInput input, .stSelectbox, .stTextArea textarea {{
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }}
    .stNumberInput input:focus, .stTextInput input:focus, .stSelectbox:focus, .stTextArea textarea:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 0.2rem rgba(0,150,199,0.25);
    }}
    /* Header Styling */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary);
        font-weight: 600;
    }}
    h1 {{
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
    }}
    /* Loader styling */
    .stSpinner > div {{
        border-top-color: var(--primary) !important;
    }}
    /* Navigation menu styling */
    .nav-link {{
        display: block;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
        text-decoration: none;
        transition: background-color 0.3s ease;
    }}
    .nav-link:hover {{
        background-color: var(--tertiary);
        color: var(--text-primary);
        text-decoration: none;
    }}
    .nav-link.active {{
        background-color: var(--primary);
        color: white;
        font-weight: 600;
    }}
    .chat-container {{
        height: 500px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: var(--background);
    }}
</style>
""", unsafe_allow_html=True)

# --- LOGGING SETUP ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- SENTRY INTEGRATION ---
try:
    import sentry_sdk
    sentry_dsn = st.secrets.get("sentry_dsn", None)
    if sentry_dsn:
        sentry_sdk.init(
            dsn=st.secrets.get("sentry_dsn"),
            traces_sample_rate=0.2,
            release="medical-device-quality-analysis@1.0.0"
        )
        logger.info("Sentry integration initialized")
    else:
        logger.warning("Sentry DSN not found in secrets")
except ImportError:
    logger.warning("Sentry SDK not installed")
except Exception as e:
    logger.exception("Error initializing Sentry")

# --- DEVICE TYPE INFORMATION ---
DEVICE_TYPES = {
    "Mobility Aids": {
        "description": "Devices that assist with movement and mobility",
        "examples": ["Canes", "Walkers", "Wheelchairs", "Knee Scooters", "Crutches", "Rollators"],
        "common_issues": ["Unstable frames", "Locking mechanism failures", "Cushion compression", "Wheel alignment issues", "Brake failures"],
        "manufacturing_processes": ["Aluminum extrusion", "Steel tube bending", "Injection molded plastics", "Foam molding", "Welding"],
        "inspection_methods": ["Load testing", "Stability testing", "Dimensional inspection", "Functional testing"]
    },
    "Cushions & Support": {
        "description": "Devices that provide comfort, positioning, and pressure relief",
        "examples": ["Seat cushions", "Back supports", "Wedge cushions", "Positioning aids", "Pressure relief cushions"],
        "common_issues": ["Foam compression", "Cover tears", "Stitching failures", "Pressure distribution issues", "Gel leakage"],
        "manufacturing_processes": ["Foam molding", "Gel encapsulation", "Fabric cutting and sewing", "Heat sealing", "Die cutting"],
        "inspection_methods": ["Pressure mapping", "Compression testing", "Fabric tensile testing", "Seam inspection", "Leak testing"]
    },
    "Monitoring Devices": {
        "description": "Devices that measure physiological parameters",
        "examples": ["Blood pressure monitors", "Glucose meters", "Pulse oximeters", "Temperature monitors", "Heart rate monitors"],
        "common_issues": ["Calibration drift", "Sensor failures", "Battery issues", "Display failures", "Software bugs", "Connectivity problems"],
        "manufacturing_processes": ["PCB assembly", "Plastic molding", "Sensor integration", "Software programming", "Battery installation"],
        "inspection_methods": ["Functional testing", "Calibration verification", "Software validation", "Battery life testing", "Connectivity testing"]
    },
    "Orthopedic Supports": {
        "description": "Devices that provide support, stability, and protection for musculoskeletal conditions",
        "examples": ["Knee braces", "Ankle braces", "Wrist supports", "Back braces", "Neck collars", "Shoulder supports"],
        "common_issues": ["Strap failures", "Hinge malfunctions", "Padding compression", "Material wear", "Sizing inconsistencies"],
        "manufacturing_processes": ["Fabric cutting", "Plastic injection molding", "Metal forming", "Foam molding", "Stitching and assembly"],
        "inspection_methods": ["Tension testing", "Range of motion testing", "Durability testing", "Strap pull testing", "Material stress testing"]
    },
    "Ostomy & Continence": {
        "description": "Devices related to ostomy care and continence management",
        "examples": ["Ostomy bags", "Skin barriers", "Catheters", "Incontinence pads", "Collection systems"],
        "common_issues": ["Leakage", "Adhesive failures", "Material degradation", "Valve malfunctions", "Odor control issues"],
        "manufacturing_processes": ["Film extrusion", "Injection molding", "Adhesive application", "Ultrasonic welding", "Sterilization"],
        "inspection_methods": ["Leak testing", "Adhesive strength testing", "Biocompatibility testing", "Sterilization validation", "Shelf-life testing"]
    },
    "Medical Software": {
        "description": "Software applications for medical purposes",
        "examples": ["Health monitoring apps", "Telemedicine platforms", "Patient management systems", "Medical calculators", "Connected device apps"],
        "common_issues": ["Crashes", "Data synchronization problems", "Security vulnerabilities", "UI/UX issues", "Feature malfunctions"],
        "manufacturing_processes": ["Software development", "Code versioning", "Unit testing", "Integration testing", "Cloud deployment"],
        "inspection_methods": ["Code reviews", "Automated testing", "User acceptance testing", "Performance testing", "Security testing"]
    },
    "Respiratory Aids": {
        "description": "Devices that assist with breathing and respiratory function",
        "examples": ["Nebulizers", "Oxygen concentrators", "CPAP machines", "Inhalers", "Suction devices"],
        "common_issues": ["Motor failures", "Tubing leaks", "Filter clogging", "Pressure control issues", "Battery degradation"],
        "manufacturing_processes": ["Motor assembly", "Plastic molding", "Silicone molding", "Electronics assembly", "Sterilization"],
        "inspection_methods": ["Flow rate testing", "Particle size testing", "Pressure testing", "Noise level testing", "Battery life testing"]
    },
    "Home Care Equipment": {
        "description": "Medical equipment designed for home use",
        "examples": ["Hospital beds", "Patient lifts", "Shower chairs", "Toilet risers", "Transfer boards"],
        "common_issues": ["Motor failures", "Control system issues", "Structural weaknesses", "Weight capacity failures", "Surface degradation"],
        "manufacturing_processes": ["Metal fabrication", "Welding", "Wood working", "Upholstery", "Electronics assembly", "Plastic molding"],
        "inspection_methods": ["Load testing", "Cycle testing", "Water resistance testing", "Electrical safety testing", "Stability testing"]
    },
    "Therapeutic Devices": {
        "description": "Devices used for therapy and rehabilitation",
        "examples": ["TENS units", "Heating pads", "Massage devices", "Light therapy", "Ultrasound therapy"],
        "common_issues": ["Controller failures", "Heating element issues", "Timer malfunctions", "Battery problems", "Electrode degradation"],
        "manufacturing_processes": ["Electronics assembly", "Textile integration", "Heat sealing", "Plastic molding", "Battery installation"],
        "inspection_methods": ["Electrical safety testing", "Temperature control testing", "Timer accuracy testing", "Battery life testing", "Output verification"]
    },
    "Generic": {
        "description": "General medical devices not falling into specific categories",
        "examples": ["Various medical devices", "Multi-category products", "New product types"],
        "common_issues": ["Material failures", "Design flaws", "Manufacturing defects", "User interface problems", "Functional issues"],
        "manufacturing_processes": ["Injection molding", "Metal fabrication", "PCB assembly", "3D printing", "CNC machining", "Die casting"],
        "inspection_methods": ["Visual inspection", "Functional testing", "Dimensional inspection", "Performance testing", "Safety testing"]
    }
}

# --- MANUFACTURING PROCESS TYPES ---
MANUFACTURING_PROCESSES = [
    "Injection molding",
    "Metal fabrication",
    "PCB assembly",
    "Fabric cutting & sewing",
    "Foam molding",
    "Plastic extrusion",
    "Metal machining",
    "Die casting",
    "3D printing",
    "Welding & assembly",
    "Software development",
    "Electronics assembly",
    "Silicone molding",
    "Ultrasonic welding",
    "Blow molding",
    "CNC machining",
    "Sterilization process",
    "Adhesive application",
    "Chemical processing",
    "Other/Multiple"
]

# --- ROOT CAUSE CATEGORIES ---
ROOT_CAUSE_CATEGORIES = [
    "Design flaw",
    "Material failure",
    "Manufacturing defect",
    "Assembly error",
    "Component failure",
    "Software bug",
    "User error/misuse",
    "Environmental factors",
    "Packaging issue",
    "Shipping damage",
    "Calibration drift",
    "Contamination",
    "Improper maintenance",
    "Normal wear and tear",
    "Quality control failure",
    "Unknown/Not determined"
]

# --- QUALITY CONTROL METHODS ---
QUALITY_CONTROL_METHODS = [
    "100% inspection",
    "AQL sampling (0.65)",
    "AQL sampling (1.0)",
    "AQL sampling (2.5)",
    "AQL sampling (4.0)",
    "Statistical process control",
    "Automated vision system",
    "Manual inspection",
    "Functional testing",
    "Batch testing",
    "Destructive testing",
    "Non-destructive testing",
    "First article inspection",
    "In-process checks",
    "Final QC verification",
    "Other/Custom method"
]

# --- UTILITY FUNCTIONS ---
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def format_currency(value: float) -> str:
    """Format a value as a currency string."""
    if value >= 1000000:
        return f"${value/1000000:.2f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format a value as a percentage string."""
    return f"{value:.2f}%"

def generate_download_link(df: pd.DataFrame, filename: str, text: str) -> str:
    """Generate a download link for a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="export-button">{text}</a>'
    return href

def generate_color_scale(value: float, good_threshold: float, bad_threshold: float) -> str:
    """Generate a color from green to red based on a value and thresholds."""
    if value >= good_threshold:
        return SUCCESS_COLOR
    elif value <= bad_threshold:
        return DANGER_COLOR
    else:
        # Linear interpolation between warning and success colors
        ratio = (value - bad_threshold) / (good_threshold - bad_threshold)
        return WARNING_COLOR

def get_recommendation_badge(recommendation: str) -> str:
    """Generate an HTML badge for a recommendation."""
    if "Fix Immediately" in recommendation:
        return f'<span class="badge badge-danger">{recommendation}</span>'
    elif "High Priority" in recommendation:
        return f'<span class="badge badge-warning">{recommendation}</span>'
    else:
        return f'<span class="badge badge-success">{recommendation}</span>'

def generate_pareto_data(failure_modes: Dict[str, int]) -> Dict[str, Any]:
    """Generate Pareto analysis data from failure modes."""
    # Sort failure modes by frequency in descending order
    sorted_modes = sorted(failure_modes.items(), key=lambda x: x[1], reverse=True)
    
    modes = [item[0] for item in sorted_modes]
    counts = [item[1] for item in sorted_modes]
    total = sum(counts)
    
    # Calculate cumulative percentages
    cumulative = []
    cumulative_percent = []
    cumulative_sum = 0
    
    for count in counts:
        cumulative_sum += count
        cumulative.append(cumulative_sum)
        cumulative_percent.append((cumulative_sum / total) * 100)
    
    return {
        "modes": modes,
        "counts": counts,
        "total": total,
        "cumulative": cumulative,
        "cumulative_percent": cumulative_percent
    }

# --- AI ASSISTANT FUNCTIONS ---
def check_openai_api_key():
    """Check if OpenAI API key is available and valid."""
    api_key = st.secrets.get("openai_api_key", None)
    
    if not api_key:
        st.session_state.api_key_status = "missing"
        logger.warning("OpenAI API key not found in secrets")
        return False
    
    # Verify the API key is valid with a simple test request
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4o", 
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=5
        )
        
        if response.status_code == 200:
            st.session_state.api_key_status = "valid"
            logger.info("OpenAI API key is valid")
            return True
        else:
            st.session_state.api_key_status = "invalid"
            logger.error(f"OpenAI API key is invalid: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.session_state.api_key_status = "error"
        logger.exception("Error checking OpenAI API key")
        return False

def call_openai_api(messages, model="gpt-4o", temperature=0.7, max_tokens=1024):
    """Call the OpenAI API with the given messages."""
    api_key = st.secrets.get("openai_api_key", None)
    
    # If API key is missing, return an error message
    if not api_key:
        logger.warning("OpenAI API key not found in secrets")
        return "Error: AI assistant is not available. Please contact alexander.popoff@vivehealth.com for support."
    
    # If API key is available, make the actual API call
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return f"Error: The AI assistant encountered a problem (HTTP {response.status_code}). Please try again later or contact alexander.popoff@vivehealth.com if the issue persists."
    except requests.exceptions.Timeout:
        logger.error("OpenAI API timeout")
        return "Error: The AI assistant timed out. Please try again later when the service is less busy."
    except requests.exceptions.ConnectionError:
        logger.error("OpenAI API connection error")
        return "Error: Could not connect to the AI service. Please check your internet connection and try again."
    except Exception as e:
        logger.exception("Error calling OpenAI API")
        return f"Error: The AI assistant encountered an unexpected problem. Please try again later or contact alexander.popoff@vivehealth.com if the issue persists."

def get_system_prompt(results: Dict[str, Any] = None, module_type: str = "quality") -> str:
    """Generate the system prompt for the AI assistant based on analysis results and module type."""
    device_type = st.session_state.get('selected_device_type', 'Generic')
    device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
    
    if module_type == "quality" and results:
        manufacturing_process = results.get('manufacturing_process', 'Not specified')
        
        # Check if root cause analysis exists
        root_cause_info = ""
        if 'root_cause' in results:
            root_cause_info = f"""
            Root Cause Analysis:
            - Primary Root Cause: {results['root_cause']}
            - Specific Failure Mode: {results.get('failure_mode', 'Not specified')}
            """
        
        # Check if quality control method exists
        qc_info = ""
        if 'quality_control_method' in results:
            qc_info = f"- Quality Control Method: {results['quality_control_method']}"
        
        return f"""
        You are a Medical Device Quality and Manufacturing Expert specializing in troubleshooting quality issues across a wide range of medical devices including durable medical equipment, mobility aids, orthopedic supports, monitoring devices, and more.
        
        Product details:
        - SKU: {results['sku']}
        - Device Type: {device_type}
        - Manufacturing Process: {manufacturing_process}
        - Issue Description: {results['issue_description']}
        - Return Rate (30d): {results['current_metrics']['return_rate_30d']:.2f}%
        - Current Unit Cost: ${results['current_metrics']['unit_cost']:.2f}
        - Sales Price: ${results['current_metrics']['sales_price']:.2f}
        {qc_info}
        
        {root_cause_info}
        
        Common Issues for {device_type}:
        - {', '.join(device_info['common_issues'])}
        
        Manufacturing Processes for {device_type}:
        - {', '.join(device_info['manufacturing_processes'])}
        
        Financial Impact:
        - Annual Loss Due to Returns: ${results['financial_impact']['annual_loss']:.2f}
        - ROI (3yr): {results['financial_impact']['roi_3yr']:.2f}%
        - Payback Period: {results['financial_impact']['payback_period']:.2f} months
        
        Recommendation: {results['recommendation']}
        
        Your task is to provide expert advice on:
        1. Likely specific causes of the issue based on the device type and manufacturing process
        2. Practical troubleshooting steps and corrective actions
        3. Manufacturing process improvements to prevent recurrence
        4. Quality control recommendations appropriate for this device type
        5. Best practices for implementing solutions in medical device manufacturing
        
        Be practical and specific in your responses. Focus on concrete troubleshooting steps, manufacturing process controls, and practical quality control methods. Only mention regulations when directly relevant to the quality issue being discussed.
        """
    elif module_type == "monte_carlo" and results:
        device_type = st.session_state.get('selected_device_type', 'Generic')
        device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
        
        return f"""
        You are a Risk Analysis and Manufacturing Process Expert for medical devices, specializing in {device_type}.
        
        Monte Carlo simulation results:
        - Probability of Positive ROI: {results['probability_metrics']['prob_positive_roi']:.2f}%
        - Probability of Payback < 1 Year: {results['probability_metrics']['prob_payback_1yr']:.2f}%
        
        ROI Statistics:
        - Mean ROI: {results['roi_stats']['mean']:.2f}%
        - Median ROI: {results['roi_stats']['median']:.2f}%
        - ROI Range: {results['roi_stats']['min']:.2f}% to {results['roi_stats']['max']:.2f}%
        
        Payback Statistics:
        - Mean Payback: {results['payback_stats']['mean']:.2f} months
        - Median Payback: {results['payback_stats']['median']:.2f} months
        - Payback Range: {results['payback_stats']['min']:.2f} to {results['payback_stats']['max']:.2f} months
        
        Device Type Information:
        - Device Type: {device_type}
        - Common Issues: {', '.join(device_info['common_issues'])}
        - Manufacturing Processes: {', '.join(device_info['manufacturing_processes'])}
        
        Your task is to provide expert advice on:
        1. Interpretation of simulation results for this specific type of medical device
        2. Manufacturing process risk assessment and control strategies
        3. Practical implementation approaches considering manufacturing variables
        4. Recommended quality control methods based on the device type and risk profile
        5. Key metrics to track during implementation
        
        Be practical and specific in your responses with a focus on manufacturing processes, quality control methods, and physical troubleshooting techniques appropriate for {device_type} devices.
        """
    else:
        # Default prompt for standalone assistant
        device_type = st.session_state.get('selected_device_type', 'Generic')
        device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
        
        return f"""
        You are a Medical Device Quality and Manufacturing Expert specializing in {device_type} devices.
        
        Your expertise includes:
        - Manufacturing processes: {', '.join(device_info['manufacturing_processes'])}
        - Common quality issues: {', '.join(device_info['common_issues'])}
        - Inspection methods: {', '.join(device_info['inspection_methods'])}
        - Examples of devices: {', '.join(device_info['examples'])}
        - Process validation and control techniques
        - Root cause analysis methods
        - Corrective and preventive actions (CAPA)
        - AQL sampling and inspection techniques
        - Manufacturing defect troubleshooting
        - Component failure analysis
        - Material selection and properties
        - Quality control methods and tools
        
        Additional areas of expertise:
        - Plastic injection molding troubleshooting
        - Metal fabrication quality issues
        - Electronic component and PCB failure analysis
        - Software validation for medical devices
        - 3D printing quality control
        - Textile and fabric product quality
        - Assembly process optimization
        - Statistical process control (SPC)
        - Measurement system analysis (MSA)
        - Design of experiments (DOE)
        
        When providing guidance, be specific to {device_type} and practical manufacturing issues that could cause quality problems. Focus on troubleshooting steps, material issues, process control methods, and practical quality control techniques rather than regulatory requirements.
        """

# --- CORE ANALYSIS FUNCTIONS ---
def analyze_quality_issue(
    sku: str,
    product_type: str,
    device_type: str,
    manufacturing_process: str,
    sales_30d: float,
    returns_30d: float,
    issue_description: str,
    root_cause: str = None,
    failure_mode: str = None,
    quality_control_method: str = None,
    current_unit_cost: float = 0,
    fix_cost_upfront: float = 0,
    fix_cost_per_unit: float = 0,
    sales_price: float = 0,
    expected_reduction: float = 0,
    solution_confidence: float = 0,
    annualized_growth: float = 0.0,
    regulatory_risk: int = 1,
    brand_risk: int = 1,
    medical_risk: int = 1
) -> Dict[str, Any]:
    """
    Analyze a quality issue and calculate financial impact, ROI, and recommendations.
    
    Args:
        sku: Product SKU identifier
        product_type: Type of medical device (B2C, B2B, etc.)
        device_type: Specific device category (cushion, brace, etc.)
        manufacturing_process: Primary manufacturing process
        sales_30d: Number of units sold in the last 30 days
        returns_30d: Number of units returned in the last 30 days
        issue_description: Description of the quality issue
        root_cause: Primary root cause category
        failure_mode: Specific failure mode
        quality_control_method: Current quality control method
        current_unit_cost: Current manufacturing cost per unit
        fix_cost_upfront: One-time cost to implement the fix
        fix_cost_per_unit: Additional cost per unit after implementing the fix
        sales_price: Sales price per unit
        expected_reduction: Expected percentage reduction in returns after fix (0-100)
        solution_confidence: Confidence level in the solution (0-100)
        annualized_growth: Expected annual growth rate (default: 0%)
        regulatory_risk: Regulatory risk level (1-5, where 5 is highest)
        brand_risk: Brand reputation risk level (1-5, where 5 is highest)
        medical_risk: Medical/patient risk level (1-5, where 5 is highest)
    
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Starting analysis for SKU={sku}, device_type={device_type}")
    
    try:
        # Input validation
        if sales_30d < 0 or returns_30d < 0 or current_unit_cost < 0 or fix_cost_upfront < 0 or fix_cost_per_unit < 0:
            raise ValueError("Input values cannot be negative")
        
        # Calculate current metrics
        return_rate_30d = safe_divide(returns_30d, sales_30d, 0) * 100
        
        # Calculate annualized metrics
        annual_sales = sales_30d * 12
        annual_returns = returns_30d * 12
        
        # Calculate financial impact of current situation
        margin_per_unit = sales_price - current_unit_cost
        margin_percentage = safe_divide(margin_per_unit, sales_price, 0) * 100
        loss_per_return = current_unit_cost + (sales_price * 0.15)  # Cost + 15% of price for handling
        annual_loss = annual_returns * loss_per_return
        
        # Calculate impact of proposed solution
        new_unit_cost = current_unit_cost + fix_cost_per_unit
        new_margin_per_unit = sales_price - new_unit_cost
        new_margin_percentage = safe_divide(new_margin_per_unit, sales_price, 0) * 100
        
        expected_returns_reduction = expected_reduction / 100
        reduced_returns = annual_returns * (1 - expected_returns_reduction)
        returns_prevented = annual_returns - reduced_returns
        savings = returns_prevented * loss_per_return
        
        # Apply confidence factor
        confidence_factor = solution_confidence / 100
        adjusted_savings = savings * confidence_factor
        
        # Calculate ROI and payback period
        implementation_cost = fix_cost_upfront + (annual_sales * fix_cost_per_unit)
        roi_1yr = safe_divide(adjusted_savings - implementation_cost, implementation_cost, 0) * 100
        
        # 3-year ROI with growth
        future_sales = annual_sales
        future_returns = reduced_returns
        cumulative_savings = 0
        cumulative_extra_cost = fix_cost_upfront
        
        for year in range(1, 4):
            future_sales *= (1 + annualized_growth/100)  # Convert percentage to decimal
            future_returns *= (1 + annualized_growth/100)
            returns_without_fix = (future_sales / annual_sales) * annual_returns
            returns_prevented_future = returns_without_fix - future_returns
            yearly_savings = returns_prevented_future * loss_per_return * confidence_factor
            yearly_extra_cost = future_sales * fix_cost_per_unit
            
            cumulative_savings += yearly_savings
            cumulative_extra_cost += yearly_extra_cost
        
        total_cost_3yr = fix_cost_upfront + cumulative_extra_cost
        roi_3yr = safe_divide(cumulative_savings - total_cost_3yr, total_cost_3yr, 0) * 100
        
        # Calculate payback period in months
        monthly_net_benefit = safe_divide(adjusted_savings - (annual_sales * fix_cost_per_unit), 12, 0)
        payback_period = safe_divide(fix_cost_upfront, monthly_net_benefit, float('inf'))
        
        # Calculate risk-adjusted metrics
        risk_factor = (regulatory_risk + brand_risk + medical_risk) / 3
        priority_score = (roi_3yr * 0.4) + ((5 - payback_period) * 10 * 0.3) + (risk_factor * 20 * 0.3)
        
        # Generate recommendation
        if medical_risk >= 4 or regulatory_risk >= 4:
            recommendation = "Fix Immediately - Safety/Compliance Risk"
        elif roi_3yr >= 200 and payback_period <= 6:
            recommendation = "High Priority - Strong ROI"
        elif roi_3yr >= 100 or risk_factor >= 3:
            recommendation = "Medium Priority - Good ROI"
        elif roi_3yr > 0:
            recommendation = "Consider Fix - Positive ROI"
        else:
            recommendation = "Monitor - Negative ROI"
        
        # Device-specific adjustment based on device type
        device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
        if medical_risk >= 3 and device_type in ["Monitoring Devices", "Respiratory Aids"]:
            recommendation = "Fix Immediately - Patient Risk"
        
        # Compile results
        results = {
            "sku": sku,
            "product_type": product_type,
            "device_type": device_type,
            "manufacturing_process": manufacturing_process,
            "issue_description": issue_description,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "current_metrics": {
                "sales_30d": sales_30d,
                "returns_30d": returns_30d,
                "return_rate_30d": return_rate_30d,
                "annual_sales": annual_sales,
                "annual_returns": annual_returns,
                "unit_cost": current_unit_cost,
                "sales_price": sales_price,
                "margin_per_unit": margin_per_unit,
                "margin_percentage": margin_percentage,
                "loss_per_return": loss_per_return,
                "annual_return_rate": return_rate_30d
            },
            "solution_metrics": {
                "fix_cost_upfront": fix_cost_upfront,
                "fix_cost_per_unit": fix_cost_per_unit,
                "new_unit_cost": new_unit_cost,
                "new_margin_per_unit": new_margin_per_unit,
                "new_margin_percentage": new_margin_percentage,
                "expected_reduction": expected_reduction,
                "solution_confidence": solution_confidence
            },
            "financial_impact": {
                "annual_loss": annual_loss,
                "returns_prevented": returns_prevented,
                "savings": savings,
                "adjusted_savings": adjusted_savings,
                "implementation_cost": implementation_cost,
                "roi_1yr": roi_1yr,
                "roi_3yr": roi_3yr,
                "payback_period": payback_period
            },
            "risk_assessment": {
                "regulatory_risk": regulatory_risk,
                "brand_risk": brand_risk,
                "medical_risk": medical_risk,
                "risk_factor": risk_factor,
                "priority_score": priority_score
            },
            "recommendation": recommendation,
            "device_info": {
                "common_issues": device_info['common_issues'],
                "manufacturing_processes": device_info['manufacturing_processes'],
                "inspection_methods": device_info['inspection_methods']
            }
        }
        
        # Add root cause and failure mode if provided
        if root_cause:
            results["root_cause"] = root_cause
        
        if failure_mode:
            results["failure_mode"] = failure_mode
            
        if quality_control_method:
            results["quality_control_method"] = quality_control_method
        
        logger.debug(f"Computed ROI: {roi_3yr:.2f}%")
        return results
    
    except Exception as e:
        logger.exception("Error in analyze_quality_issue")
        raise

def run_monte_carlo_simulation(
    base_unit_cost: float,
    base_sales_price: float,
    base_monthly_sales: float,
    base_return_rate: float,
    fix_cost_upfront: float,
    fix_cost_per_unit: float,
    expected_reduction: float,
    iterations: int = 1000,
    cost_std_dev: float = 0.05,
    price_std_dev: float = 0.03,
    sales_std_dev: float = 0.10,
    return_std_dev: float = 0.15,
    reduction_std_dev: float = 0.20
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation to analyze risk and probability distributions.
    
    Args:
        base_unit_cost: Base manufacturing cost per unit
        base_sales_price: Base sales price per unit
        base_monthly_sales: Base monthly sales units
        base_return_rate: Base return rate (0-100)
        fix_cost_upfront: One-time cost to implement the fix
        fix_cost_per_unit: Additional cost per unit after fix
        expected_reduction: Expected percentage reduction in returns (0-100)
        iterations: Number of simulation iterations
        cost_std_dev: Standard deviation for unit cost variations
        price_std_dev: Standard deviation for price variations
        sales_std_dev: Standard deviation for sales volume variations
        return_std_dev: Standard deviation for return rate variations
        reduction_std_dev: Standard deviation for expected reduction variations
    
    Returns:
        Dictionary containing simulation results
    """
    np.random.seed(42)  # For reproducibility
    
    try:
        # Initialize arrays to store simulation results
        roi_results = np.zeros(iterations)
        payback_results = np.zeros(iterations)
        savings_results = np.zeros(iterations)
        
        for i in range(iterations):
            # Generate random variations of input parameters
            unit_cost = np.random.normal(base_unit_cost, base_unit_cost * cost_std_dev)
            unit_cost = max(unit_cost, 0.1)  # Ensure positive
            
            sales_price = np.random.normal(base_sales_price, base_sales_price * price_std_dev)
            sales_price = max(sales_price, unit_cost * 1.1)  # Ensure price > cost
            
            monthly_sales = np.random.normal(base_monthly_sales, base_monthly_sales * sales_std_dev)
            monthly_sales = max(monthly_sales, 1)  # Ensure positive
            
            return_rate = np.random.normal(base_return_rate, base_return_rate * return_std_dev)
            return_rate = np.clip(return_rate, 0.1, 100)  # Ensure between 0.1% and 100%
            
            reduction = np.random.normal(expected_reduction, expected_reduction * reduction_std_dev)
            reduction = np.clip(reduction, 0, 100)  # Ensure between 0% and 100%
            
            # Calculate financial metrics for this iteration
            annual_sales = monthly_sales * 12
            annual_returns = (annual_sales * return_rate) / 100
            
            loss_per_return = unit_cost + (sales_price * 0.15)
            annual_loss = annual_returns * loss_per_return
            
            new_unit_cost = unit_cost + fix_cost_per_unit
            
            reduced_returns = annual_returns * (1 - (reduction / 100))
            returns_prevented = annual_returns - reduced_returns
            savings = returns_prevented * loss_per_return
            
            implementation_cost = fix_cost_upfront + (annual_sales * fix_cost_per_unit)
            roi = safe_divide(savings - implementation_cost, implementation_cost, -100) * 100
            
            monthly_net_benefit = safe_divide(savings - (annual_sales * fix_cost_per_unit), 12, 0.01)
            payback = safe_divide(fix_cost_upfront, monthly_net_benefit, 120)  # Cap at 10 years
            
            # Store results
            roi_results[i] = roi
            payback_results[i] = payback
            savings_results[i] = savings
        
        # Analyze results
        roi_mean = np.mean(roi_results)
        roi_median = np.median(roi_results)
        roi_std = np.std(roi_results)
        roi_min = np.min(roi_results)
        roi_max = np.max(roi_results)
        
        payback_mean = np.mean(payback_results)
        payback_median = np.median(payback_results)
        payback_std = np.std(payback_results)
        payback_min = np.min(payback_results)
        payback_max = np.max(payback_results)
        
        savings_mean = np.mean(savings_results)
        savings_median = np.median(savings_results)
        savings_std = np.std(savings_results)
        savings_min = np.min(savings_results)
        savings_max = np.max(savings_results)
        
        # Calculate probability of positive ROI
        prob_positive_roi = np.sum(roi_results > 0) / iterations * 100
        
        # Calculate probability of payback within 12 months
        prob_payback_1yr = np.sum(payback_results <= 12) / iterations * 100
        
        # Calculate percentiles for ROI
        roi_percentiles = {
            "p10": np.percentile(roi_results, 10),
            "p25": np.percentile(roi_results, 25),
            "p50": np.percentile(roi_results, 50),
            "p75": np.percentile(roi_results, 75),
            "p90": np.percentile(roi_results, 90)
        }
        
        # Calculate percentiles for payback
        payback_percentiles = {
            "p10": np.percentile(payback_results, 10),
            "p25": np.percentile(payback_results, 25),
            "p50": np.percentile(payback_results, 50),
            "p75": np.percentile(payback_results, 75),
            "p90": np.percentile(payback_results, 90)
        }
        
        # Prepare histogram data for ROI and payback
        roi_hist, roi_bins = np.histogram(roi_results, bins=20)
        payback_hist, payback_bins = np.histogram(payback_results, bins=20)
        
        return {
            "iterations": iterations,
            "input_parameters": {
                "base_unit_cost": base_unit_cost,
                "base_sales_price": base_sales_price,
                "base_monthly_sales": base_monthly_sales,
                "base_return_rate": base_return_rate,
                "fix_cost_upfront": fix_cost_upfront,
                "fix_cost_per_unit": fix_cost_per_unit,
                "expected_reduction": expected_reduction
            },
            "roi_stats": {
                "mean": roi_mean,
                "median": roi_median,
                "std_dev": roi_std,
                "min": roi_min,
                "max": roi_max,
                "percentiles": roi_percentiles,
                "histogram": {
                    "counts": roi_hist.tolist(),
                    "bins": roi_bins.tolist()
                }
            },
            "payback_stats": {
                "mean": payback_mean,
                "median": payback_median,
                "std_dev": payback_std,
                "min": payback_min,
                "max": payback_max,
                "percentiles": payback_percentiles,
                "histogram": {
                    "counts": payback_hist.tolist(),
                    "bins": payback_bins.tolist()
                }
            },
            "savings_stats": {
                "mean": savings_mean,
                "median": savings_median,
                "std_dev": savings_std,
                "min": savings_min,
                "max": savings_max
            },
            "probability_metrics": {
                "prob_positive_roi": prob_positive_roi,
                "prob_payback_1yr": prob_payback_1yr
            }
        }
    except Exception as e:
        logger.exception("Error in run_monte_carlo_simulation")
        raise

# --- EXPORT FUNCTIONS ---
def export_as_csv(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert analysis results to a DataFrame for CSV export."""
    if not results:
        return pd.DataFrame()
    
    data = {
        "Metric": [
            "SKU",
            "Product Type",
            "Device Type",
            "Manufacturing Process",
            "Analysis Date",
            "Return Rate (30d)",
            "Annual Return Rate",
            "Annual Loss Due to Returns",
            "Fix Cost (Upfront)",
            "Fix Cost (Per Unit)",
            "Expected Reduction in Returns",
            "ROI (1 Year)",
            "ROI (3 Year)",
            "Payback Period (Months)",
            "Recommendation"
        ],
        "Value": [
            results["sku"],
            results["product_type"],
            results["device_type"],
            results["manufacturing_process"],
            results["analysis_date"],
            f"{results['current_metrics']['return_rate_30d']:.2f}%",
            f"{results['current_metrics']['annual_return_rate']:.2f}%",
            f"${results['financial_impact']['annual_loss']:.2f}",
            f"${results['solution_metrics']['fix_cost_upfront']:.2f}",
            f"${results['solution_metrics']['fix_cost_per_unit']:.2f}",
            f"{results['solution_metrics']['expected_reduction']:.2f}%",
            f"{results['financial_impact']['roi_1yr']:.2f}%",
            f"{results['financial_impact']['roi_3yr']:.2f}%",
            f"{results['financial_impact']['payback_period']:.2f}",
            results["recommendation"]
        ]
    }
    
    # Add root cause if available
    if "root_cause" in results:
        data["Metric"].append("Root Cause")
        data["Value"].append(results["root_cause"])
    
    # Add failure mode if available
    if "failure_mode" in results:
        data["Metric"].append("Failure Mode")
        data["Value"].append(results["failure_mode"])
    
    # Add quality control method if available
    if "quality_control_method" in results:
        data["Metric"].append("Quality Control Method")
        data["Value"].append(results["quality_control_method"])
    
    return pd.DataFrame(data)

def export_monte_carlo_as_csv(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert Monte Carlo simulation results to a DataFrame for CSV export."""
    if not results:
        return pd.DataFrame()
    
    data = {
        "Metric": [
            "Iterations",
            "Probability of Positive ROI",
            "Probability of Payback < 1 Year",
            "Mean ROI",
            "Median ROI",
            "ROI Standard Deviation",
            "ROI Range (Min)",
            "ROI Range (Max)",
            "10th Percentile ROI",
            "90th Percentile ROI",
            "Mean Payback Period",
            "Median Payback Period",
            "Payback Period Standard Deviation",
            "Payback Range (Min)",
            "Payback Range (Max)",
            "10th Percentile Payback",
            "90th Percentile Payback"
        ],
        "Value": [
            f"{results['iterations']}",
            f"{results['probability_metrics']['prob_positive_roi']:.2f}%",
            f"{results['probability_metrics']['prob_payback_1yr']:.2f}%",
            f"{results['roi_stats']['mean']:.2f}%",
            f"{results['roi_stats']['median']:.2f}%",
            f"{results['roi_stats']['std_dev']:.2f}%",
            f"{results['roi_stats']['min']:.2f}%",
            f"{results['roi_stats']['max']:.2f}%",
            f"{results['roi_stats']['percentiles']['p10']:.2f}%",
            f"{results['roi_stats']['percentiles']['p90']:.2f}%",
            f"{results['payback_stats']['mean']:.2f} months",
            f"{results['payback_stats']['median']:.2f} months",
            f"{results['payback_stats']['std_dev']:.2f} months",
            f"{results['payback_stats']['min']:.2f} months",
            f"{results['payback_stats']['max']:.2f} months",
            f"{results['payback_stats']['percentiles']['p10']:.2f} months",
            f"{results['payback_stats']['percentiles']['p90']:.2f} months"
        ]
    }
    
    return pd.DataFrame(data)

def export_as_pdf(results: Dict[str, Any]) -> BytesIO:
    """Generate a PDF report of the analysis results."""
    if not results:
        return BytesIO()
    
    buffer = BytesIO()
    
    try:
        with PdfPages(buffer) as pdf:
            # Set up the figure for the first page
            plt.figure(figsize=(8.5, 11))
            plt.suptitle(f"Quality Issue Analysis: {results['sku']}", fontsize=16)
            plt.text(0.1, 0.95, f"Product: {results['product_type']}", fontsize=12)
            plt.text(0.1, 0.92, f"Device Type: {results['device_type']}", fontsize=12)
            plt.text(0.1, 0.89, f"Manufacturing Process: {results['manufacturing_process']}", fontsize=12)
            plt.text(0.1, 0.86, f"Analysis Date: {results['analysis_date']}", fontsize=12)
            plt.text(0.1, 0.83, f"Issue Description: {results['issue_description']}", fontsize=12)
            
            # Add root cause if available
            y_pos = 0.80
            if "root_cause" in results:
                plt.text(0.1, y_pos, f"Root Cause: {results['root_cause']}", fontsize=12)
                y_pos -= 0.03
            
            if "failure_mode" in results:
                plt.text(0.1, y_pos, f"Failure Mode: {results['failure_mode']}", fontsize=12)
                y_pos -= 0.03
            
            if "quality_control_method" in results:
                plt.text(0.1, y_pos, f"Quality Control Method: {results['quality_control_method']}", fontsize=12)
                y_pos -= 0.03
            
            # Key metrics (start from adjusted y position)
            plt.text(0.1, y_pos - 0.03, "Key Metrics:", fontsize=14, weight='bold')
            plt.text(0.1, y_pos - 0.07, f"Return Rate (30d): {results['current_metrics']['return_rate_30d']:.2f}%", fontsize=12)
            plt.text(0.1, y_pos - 0.11, f"Annual Loss: ${results['financial_impact']['annual_loss']:.2f}", fontsize=12)
            plt.text(0.1, y_pos - 0.15, f"ROI (3 Year): {results['financial_impact']['roi_3yr']:.2f}%", fontsize=12)
            plt.text(0.1, y_pos - 0.19, f"Payback Period: {results['financial_impact']['payback_period']:.2f} months", fontsize=12)
            plt.text(0.1, y_pos - 0.23, f"Recommendation: {results['recommendation']}", fontsize=12, weight='bold')
            
            # Add charts
            ax1 = plt.axes([0.1, 0.35, 0.35, 0.25])
            ax1.bar(['Current', 'After Fix'], 
                   [results['current_metrics']['return_rate_30d'], 
                    results['current_metrics']['return_rate_30d'] * (1 - results['solution_metrics']['expected_reduction']/100)])
            ax1.set_title('Return Rate')
            ax1.set_ylabel('Percentage')
            
            ax2 = plt.axes([0.55, 0.35, 0.35, 0.25])
            ax2.bar(['Current', 'After Fix'], 
                   [results['current_metrics']['margin_percentage'], 
                    results['solution_metrics']['new_margin_percentage']])
            ax2.set_title('Margin Percentage')
            ax2.set_ylabel('Percentage')
            
            # Device-specific information
            plt.text(0.1, 0.30, f"Common Issues for {results['device_type']}:", fontsize=12, weight='bold')
            for i, issue in enumerate(results['device_info']['common_issues'][:3]):  # Show up to 3 issues
                plt.text(0.1, 0.27 - (i * 0.03), f"• {issue}", fontsize=10)
            
            plt.text(0.5, 0.30, "Recommended Inspection Methods:", fontsize=12, weight='bold')
            for i, method in enumerate(results['device_info']['inspection_methods'][:3]):  # Show up to 3 methods
                plt.text(0.5, 0.27 - (i * 0.03), f"• {method}", fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig()
            plt.close()
            
            # Second page with financial details
            plt.figure(figsize=(8.5, 11))
            plt.suptitle("Financial Impact Details", fontsize=16)
            
            # Current state
            plt.text(0.1, 0.9, "Current State:", fontsize=14, weight='bold')
            plt.text(0.1, 0.85, f"Sales (30d): {results['current_metrics']['sales_30d']}", fontsize=12)
            plt.text(0.1, 0.8, f"Returns (30d): {results['current_metrics']['returns_30d']}", fontsize=12)
            plt.text(0.1, 0.75, f"Unit Cost: ${results['current_metrics']['unit_cost']:.2f}", fontsize=12)
            plt.text(0.1, 0.7, f"Sales Price: ${results['current_metrics']['sales_price']:.2f}", fontsize=12)
            plt.text(0.1, 0.65, f"Margin: ${results['current_metrics']['margin_per_unit']:.2f} ({results['current_metrics']['margin_percentage']:.2f}%)", fontsize=12)
            
            # Proposed solution
            plt.text(0.1, 0.55, "Proposed Solution:", fontsize=14, weight='bold')
            plt.text(0.1, 0.5, f"Upfront Cost: ${results['solution_metrics']['fix_cost_upfront']:.2f}", fontsize=12)
            plt.text(0.1, 0.45, f"Cost Per Unit: ${results['solution_metrics']['fix_cost_per_unit']:.2f}", fontsize=12)
            plt.text(0.1, 0.4, f"New Unit Cost: ${results['solution_metrics']['new_unit_cost']:.2f}", fontsize=12)
            plt.text(0.1, 0.35, f"New Margin: ${results['solution_metrics']['new_margin_per_unit']:.2f} ({results['solution_metrics']['new_margin_percentage']:.2f}%)", fontsize=12)
            plt.text(0.1, 0.3, f"Expected Reduction: {results['solution_metrics']['expected_reduction']:.2f}%", fontsize=12)
            plt.text(0.1, 0.25, f"Confidence: {results['solution_metrics']['solution_confidence']:.2f}%", fontsize=12)
            
            # Financial impact
            plt.text(0.5, 0.55, "Financial Impact:", fontsize=14, weight='bold')
            plt.text(0.5, 0.5, f"Annual Loss: ${results['financial_impact']['annual_loss']:.2f}", fontsize=12)
            plt.text(0.5, 0.45, f"Returns Prevented: {results['financial_impact']['returns_prevented']:.2f}", fontsize=12)
            plt.text(0.5, 0.4, f"Savings: ${results['financial_impact']['savings']:.2f}", fontsize=12)
            plt.text(0.5, 0.35, f"Adjusted Savings: ${results['financial_impact']['adjusted_savings']:.2f}", fontsize=12)
            plt.text(0.5, 0.3, f"Implementation Cost: ${results['financial_impact']['implementation_cost']:.2f}", fontsize=12)
            plt.text(0.5, 0.25, f"ROI (1yr): {results['financial_impact']['roi_1yr']:.2f}%", fontsize=12)
            plt.text(0.5, 0.2, f"ROI (3yr): {results['financial_impact']['roi_3yr']:.2f}%", fontsize=12)
            plt.text(0.5, 0.15, f"Payback Period: {results['financial_impact']['payback_period']:.2f} months", fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.close()
            
            # Third page with charts
            plt.figure(figsize=(8.5, 11))
            plt.suptitle("Visual Analysis", fontsize=16)
            
            # ROI chart
            ax1 = plt.axes([0.1, 0.7, 0.8, 0.2])
            roi_values = [results['financial_impact']['roi_1yr'], results['financial_impact']['roi_3yr']]
            ax1.bar(['1 Year ROI', '3 Year ROI'], roi_values, color=['#48CAE4', '#0096C7'])
            for i, v in enumerate(roi_values):
                ax1.text(i, v + 5, f"{v:.1f}%", ha='center')
            ax1.set_title('Return on Investment')
            ax1.set_ylabel('Percentage')
            
            # Return rate reduction chart
            ax2 = plt.axes([0.1, 0.4, 0.8, 0.2])
            labels = ['Before', 'After']
            values = [
                results['current_metrics']['return_rate_30d'],
                results['current_metrics']['return_rate_30d'] * (1 - results['solution_metrics']['expected_reduction'] / 100)
            ]
            ax2.bar(labels, values, color=['#E76F51', '#40916C'])
            for i, v in enumerate(values):
                ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            ax2.set_title('Return Rate Reduction')
            ax2.set_ylabel('Return Rate (%)')
            
            # Margin impact chart
            ax3 = plt.axes([0.1, 0.1, 0.8, 0.2])
            margin_labels = ['Original Margin', 'New Margin']
            margin_values = [
                results['current_metrics']['margin_percentage'],
                results['solution_metrics']['new_margin_percentage']
            ]
            ax3.bar(margin_labels, margin_values, color=['#E9C46A', '#48CAE4'])
            for i, v in enumerate(margin_values):
                ax3.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            ax3.set_title('Margin Impact')
            ax3.set_ylabel('Margin (%)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.close()
            
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.exception("Error generating PDF report")
        plt.close('all')  # Close any open figures
        raise

# --- UI COMPONENTS ---
def display_header():
    """Display the application header."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🔍 Medical Device Quality Analysis")
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            Analyze quality issues, calculate ROI for improvement projects, and troubleshoot manufacturing problems for medical devices.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        with st.popover("📋 Help", use_container_width=True):
            st.markdown("""
            ### How to use this tool
            
            1. **Quality ROI Analysis**: Enter product details and quality issue information to calculate ROI of potential fixes
            2. **Process Control Analysis**: Identify manufacturing process issues and control measures
            3. **Root Cause Analysis**: Analyze potential causes of quality problems using Pareto charts
            4. **Monte Carlo Simulation**: Understand risks and probabilities with statistical modeling
            5. **Quality AI Assistant**: Get expert advice from our AI assistant for medical device manufacturing
            
            Each tab provides specialized analysis tools with interactive visualizations and export options.
            
            #### Tips
            - Select the specific device type to get tailored analysis
            - Analyze manufacturing process controls to identify improvement areas
            - Use the AI assistant to get troubleshooting recommendations
            - Export results as CSV or PDF for reporting
            
            For additional help, contact the Quality Management team.
            """)
    
    # Add app navigation and user info in a status bar
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color: #6c757d; margin-right: 1rem;">User: Quality Manager</span>
            <span style="color: #6c757d;">Department: Product Development</span>
        </div>
        <div>
            <span style="color: #6c757d;">Last updated: {datetime.now().strftime("%Y-%m-%d")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_navigation():
    """Display the navigation menu."""
    st.sidebar.markdown("## Navigation")
    
    if st.sidebar.button("📊 Analysis Tools", key="nav_analysis", use_container_width=True, 
                         help="Access the main analysis tools"):
        st.session_state.current_page = "analysis"
        st.rerun()
    
    if st.sidebar.button("🤖 AI Assistant", key="nav_assistant", use_container_width=True,
                          help="Chat with the Quality Management AI Assistant"):
        st.session_state.current_page = "assistant"
        st.rerun()
    
    # Device type selector
    st.sidebar.markdown("## Device Type")
    device_type = st.sidebar.selectbox(
        "Select Device Type",
        options=list(DEVICE_TYPES.keys()),
        index=list(DEVICE_TYPES.keys()).index(st.session_state.get('selected_device_type', 'Generic')),
        help="Select specific device type for tailored analysis"
    )
    
    if device_type != st.session_state.get('selected_device_type', 'Generic'):
        st.session_state.selected_device_type = device_type
        st.rerun()
    
    # Display device type info
    if device_type in DEVICE_TYPES:
        device_info = DEVICE_TYPES[device_type]
        with st.sidebar.expander(f"{device_type} Information", expanded=False):
            st.markdown(f"**{device_info['description']}**")
            st.markdown("**Examples:**")
            st.markdown("\n".join([f"- {example}" for example in device_info['examples']]))
            st.markdown("**Common Issues:**")
            st.markdown("\n".join([f"- {issue}" for issue in device_info['common_issues']]))
    
    # Add OpenAI API status indicator
    if st.session_state.api_key_status is None:
        # Check API key on first load
        check_openai_api_key()
    
    status_color = {
        "valid": "#40916C",  # Green
        "missing": "#E76F51",  # Red
        "invalid": "#E76F51",  # Red
        "error": "#E9C46A",  # Yellow
        None: "#ADB5BD"  # Gray
    }
    
    status_text = {
        "valid": "AI Assistant Connected",
        "missing": "AI Assistant: API Key Missing",
        "invalid": "AI Assistant: Invalid API Key",
        "error": "AI Assistant: Connection Error",
        None: "AI Assistant: Status Unknown"
    }
    
    # Use a markdown-based status indicator 
    st.sidebar.markdown(f"""
    <div style="margin-top: 2rem; padding: 0.5rem; border-radius: 4px; background-color: #f8f9fa;">
        <div style="display: flex; align-items: center;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {status_color[st.session_state.api_key_status]}; margin-right: 8px;"></div>
            <span style="font-size: 0.8rem; color: #6c757d;">{status_text[st.session_state.api_key_status]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add helpful resources
    st.sidebar.markdown("## Resources")
    with st.sidebar.expander("Manufacturing Process Controls", expanded=False):
        st.markdown("""
        - Statistical Process Control (SPC) charts
        - Process Capability Analysis (Cp, Cpk)
        - Gauge R&R studies
        - Process Failure Mode Effects Analysis (PFMEA)
        - Design of Experiments (DOE)
        """)
    
    with st.sidebar.expander("Quality Control Methods", expanded=False):
        st.markdown("""
        - **AQL Sampling**: 0.65 (Medical), 1.0 (Critical), 2.5 (Major), 4.0 (Minor)
        - **Inspection Methods**: Visual, dimensional, functional, performance
        - **Testing Types**: Destructive, non-destructive, accelerated life
        - **Process Validation**: IQ, OQ, PQ methodologies
        - **Root Cause Analysis**: 5-Why, Fishbone, Pareto analysis
        """)
    
    with st.sidebar.expander("Defect Troubleshooting Guide", expanded=False):
        st.markdown("""
        **Injection Molding Issues:**
        - **Sink marks**: Increase hold pressure, cooling time
        - **Flash**: Decrease injection pressure, increase clamping force
        - **Short shots**: Increase injection pressure, material temperature
        - **Warping**: Balance cooling, reduce melt temperature
        
        **Metal Fabrication Issues:**
        - **Weld porosity**: Clean materials, proper gas shielding
        - **Dimensional variance**: Fixture improvements, tool wear check
        - **Surface finish issues**: Adjust feed rate, tool selection
        - **Cracking**: Stress relief, material selection review
        
        **Textile/Soft Goods Issues:**
        - **Seam failures**: Thread tension, needle size, stitch density
        - **Fabric tears**: Material handling, cutting technique review
        - **Uneven stitching**: Machine timing, operator training
        """)

def display_quality_issue_results(results: Dict[str, Any], expanded: bool = True):
    """Display the results of a quality issue analysis."""
    if not results:
        return
    
    st.markdown("### 📊 Analysis Results")
    
    # Summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Use tooltip for extra information
        with st.container():
            st.markdown(f"""
            <div class="metric-card" data-tooltip="30-day return rate based on recent sales data">
                <div class="metric-label">Return Rate (30d)</div>
                <div class="metric-value">{results['current_metrics']['return_rate_30d']:.2f}%</div>
                <div class="metric-subvalue">Monthly average</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Estimated annual financial impact from returns">
            <div class="metric-label">Annual Loss</div>
            <div class="metric-value">{format_currency(results['financial_impact']['annual_loss'])}</div>
            <div class="metric-subvalue">Due to returns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        roi_color = generate_color_scale(
            results['financial_impact']['roi_3yr'], 200, 0
        )
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Return on investment over a 3-year period">
            <div class="metric-label">3-Year ROI</div>
            <div class="metric-value" style="color: {roi_color};">{results['financial_impact']['roi_3yr']:.2f}%</div>
            <div class="metric-subvalue">1-Year: {results['financial_impact']['roi_1yr']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        payback_color = generate_color_scale(
            24 - min(results['financial_impact']['payback_period'], 24), 18, 6
        )
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Time required to recover the investment">
            <div class="metric-label">Payback Period</div>
            <div class="metric-value" style="color: {payback_color};">{results['financial_impact']['payback_period']:.1f} months</div>
            <div class="metric-subvalue">To break even</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add device type and manufacturing process info
    st.markdown(f"""
    <div style="background-color: {TERTIARY_COLOR}; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; 
              display: flex; flex-wrap: wrap; gap: 1rem;">
        <div><strong>Device Type:</strong> {results['device_type']}</div>
        <div><strong>Manufacturing Process:</strong> {results['manufacturing_process']}</div>
        
        {"<div><strong>Root Cause:</strong> " + results.get('root_cause', '') + "</div>" if 'root_cause' in results else ""}
        {"<div><strong>Failure Mode:</strong> " + results.get('failure_mode', '') + "</div>" if 'failure_mode' in results else ""}
        {"<div><strong>QC Method:</strong> " + results.get('quality_control_method', '') + "</div>" if 'quality_control_method' in results else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation banner
    st.markdown(f"""
    <div style="background-color: {PRIMARY_COLOR}; padding: 1rem; border-radius: 4px; margin: 1rem 0; 
                color: white; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 1.1rem; font-weight: 600;">Recommendation:</span> 
            <span style="font-size: 1.1rem;">{results['recommendation']}</span>
        </div>
        <div>
            {get_recommendation_badge(results['recommendation'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed results in expandable section
    with st.expander("View Detailed Analysis", expanded=expanded):
        tabs = st.tabs(["Financial Impact", "Manufacturing Analysis", "Root Cause", "Risk Assessment"])
        
        with tabs[0]:
            # Financial Impact tab
            st.subheader("Financial Impact")
            
            # Calculate values needed for display
            annual_returns = results['current_metrics']['annual_returns']
            annual_sales = results['current_metrics']['annual_sales']
            loss_per_return = results['financial_impact']['annual_loss'] / annual_returns if annual_returns > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Situation")
                st.markdown(f"""
                - **Annual Returns:** {annual_returns:.0f} units
                - **Return Rate:** {results['current_metrics']['return_rate_30d']:.2f}%
                - **Loss Per Return:** {format_currency(loss_per_return)}
                - **Annual Loss:** {format_currency(results['financial_impact']['annual_loss'])}
                """)
            
            with col2:
                st.markdown("#### After Improvement")
                st.markdown(f"""
                - **Returns Prevented:** {results['financial_impact']['returns_prevented']:.0f} units
                - **Gross Savings:** {format_currency(results['financial_impact']['savings'])}
                - **Adjusted Savings:** {format_currency(results['financial_impact']['adjusted_savings'])} (with {results['solution_metrics']['solution_confidence']}% confidence)
                - **Implementation Cost:** {format_currency(results['financial_impact']['implementation_cost'])} (includes {format_currency(results['solution_metrics']['fix_cost_upfront'])} upfront)
                """)
            
            # Waterfall chart with hover tooltips
            fig = go.Figure(go.Waterfall(
                name="Financial Impact",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Current Loss", "Prevented Returns", "Implementation Cost", "Ongoing Costs", "Net Impact"],
                textposition="outside",
                text=[
                    f"${results['financial_impact']['annual_loss']:,.0f}",
                    f"+${results['financial_impact']['adjusted_savings']:,.0f}",
                    f"-${results['solution_metrics']['fix_cost_upfront']:,.0f}",
                    f"-${(annual_sales * results['solution_metrics']['fix_cost_per_unit']):,.0f}",
                    f"${(results['financial_impact']['adjusted_savings'] - results['financial_impact']['implementation_cost']):,.0f}"
                ],
                y=[
                    -results['financial_impact']['annual_loss'],
                    results['financial_impact']['adjusted_savings'],
                    -results['solution_metrics']['fix_cost_upfront'],
                    -(annual_sales * results['solution_metrics']['fix_cost_per_unit']),
                    0
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": DANGER_COLOR}},
                increasing={"marker": {"color": SUCCESS_COLOR}},
                totals={"marker": {"color": SECONDARY_COLOR}},
                hoverinfo="text",
                hovertext=[
                    f"Current annual loss due to returns: ${results['financial_impact']['annual_loss']:,.0f}",
                    f"Savings from prevented returns: +${results['financial_impact']['adjusted_savings']:,.0f}",
                    f"One-time implementation cost: -${results['solution_metrics']['fix_cost_upfront']:,.0f}",
                    f"Annual ongoing costs: -${(annual_sales * results['solution_metrics']['fix_cost_per_unit']):,.0f}",
                    f"Net annual impact: ${(results['financial_impact']['adjusted_savings'] - results['financial_impact']['implementation_cost']):,.0f}"
                ]
            ))
            
            fig.update_layout(
                title="Financial Impact Waterfall",
                showlegend=False,
                height=400,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            # Manufacturing Analysis tab
            st.subheader("Manufacturing Process Analysis")
            
            # Device type specific information
            device_info = results['device_info']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Common Issues for this Device Type")
                for issue in device_info['common_issues']:
                    st.markdown(f"- {issue}")
                
                st.markdown("#### Manufacturing Processes")
                for process in device_info['manufacturing_processes']:
                    st.markdown(f"- {process}")
            
            with col2:
                st.markdown("#### Recommended Inspection Methods")
                for method in device_info['inspection_methods']:
                    st.markdown(f"- {method}")
                
                st.markdown("#### Process Control Recommendations")
                # This would be based on the specific device type and manufacturing process
                if results['manufacturing_process'] == "Injection molding":
                    st.markdown("""
                    - Monitor and control mold temperature
                    - Verify material drying and processing parameters
                    - Implement cavity pressure sensors
                    - Regular preventive maintenance of molds
                    - Control material lot-to-lot variation
                    """)
                elif results['manufacturing_process'] == "PCB assembly":
                    st.markdown("""
                    - Automated optical inspection (AOI)
                    - X-ray inspection for BGAs and hidden joints
                    - Monitor reflow profiles
                    - Regular cleaning of stencils
                    - Component placement verification
                    """)
                elif results['manufacturing_process'] in ["Fabric cutting & sewing", "Foam molding"]:
                    st.markdown("""
                    - Regular calibration of cutting equipment
                    - Tension control for sewing operations
                    - Material property verification
                    - Environmental controls (temperature, humidity)
                    - Operator training and certification
                    """)
                elif results['manufacturing_process'] in ["Metal fabrication", "Metal machining", "Welding & assembly"]:
                    st.markdown("""
                    - Material certification verification
                    - Tool wear monitoring
                    - Regular calibration of measurement equipment
                    - Weld quality inspection
                    - Fixture validation
                    """)
                else:
                    st.markdown("""
                    - Statistical Process Control (SPC)
                    - First Article Inspection (FAI)
                    - Regular process audits
                    - Preventive maintenance program
                    - Operator training and certification
                    """)
            
            # Process capability chart (simulated for demonstration)
            st.markdown("#### Process Capability Analysis")
            
            # Create simulated process capability data
            np.random.seed(42)  # For reproducibility
            spec_lower = 95
            spec_upper = 105
            process_mean = 100
            process_std = 1.5
            
            # Generate sample data
            sample_size = 100
            measurements = np.random.normal(process_mean, process_std, sample_size)
            
            # Calculate process capability indices
            cp = (spec_upper - spec_lower) / (6 * process_std)
            cpk = min(
                (process_mean - spec_lower) / (3 * process_std),
                (spec_upper - process_mean) / (3 * process_std)
            )
            
            # Create histogram with specifications
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=measurements,
                opacity=0.75,
                marker_color=SECONDARY_COLOR,
                name="Process Distribution"
            ))
            
            # Add specification lines
            fig.add_shape(
                type="line",
                x0=spec_lower, y0=0,
                x1=spec_lower, y1=18,
                line=dict(color="red", width=2, dash="dash"),
                name="Lower Spec"
            )
            
            fig.add_shape(
                type="line",
                x0=spec_upper, y0=0,
                x1=spec_upper, y1=18,
                line=dict(color="red", width=2, dash="dash"),
                name="Upper Spec"
            )
            
            # Add mean line
            fig.add_shape(
                type="line",
                x0=np.mean(measurements), y0=0,
                x1=np.mean(measurements), y1=18,
                line=dict(color="green", width=2),
                name="Process Mean"
            )
            
            # Add annotations for Cp and Cpk
            fig.add_annotation(
                x=spec_lower + 2,
                y=15,
                text=f"Cp: {cp:.2f}<br>Cpk: {cpk:.2f}",
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            fig.update_layout(
                title=f"Process Capability for {results['manufacturing_process']}",
                xaxis_title="Measurement Value",
                yaxis_title="Frequency",
                height=400,
                shapes=[
                    # Add normal distribution curve
                    dict(
                        type="path",
                        path=f"M {spec_lower} 0 C {spec_lower} 0, {(spec_lower + spec_upper) / 2} 20, {spec_upper} 0 Z",
                        fillcolor="rgba(0, 150, 199, 0.2)",
                        line=dict(color="rgba(0, 150, 199, 0.5)"),
                        layer="below"
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Process improvement recommendations
            st.markdown("#### Process Improvement Recommendations")
            
            if cpk < 1.0:
                st.warning("⚠️ Process is not capable (Cpk < 1.0). Process improvement is required.")
                st.markdown("""
                **Recommendations:**
                1. Identify and reduce sources of variation
                2. Center process on target
                3. Implement Statistical Process Control
                4. Consider process redesign if capability cannot be improved
                """)
            elif cpk < 1.33:
                st.info("ℹ️ Process is marginally capable (1.0 < Cpk < 1.33). Improvement is recommended.")
                st.markdown("""
                **Recommendations:**
                1. Continue monitoring the process
                2. Implement targeted improvements to reduce variation
                3. Review process controls and operator training
                4. Consider preventive maintenance to maintain stability
                """)
            else:
                st.success("✅ Process is capable (Cpk > 1.33). Maintain current controls.")
                st.markdown("""
                **Recommendations:**
                1. Maintain current process controls
                2. Continue regular monitoring
                3. Document process parameters for future reference
                4. Train new operators to maintain consistency
                """)
        
        with tabs[2]:
            # Root Cause Analysis tab
            st.subheader("Root Cause Analysis")
            
            # Root cause information if available
            if 'root_cause' in results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Primary Root Cause")
                    st.info(f"**{results['root_cause']}**")
                    
                    # Provide specific recommendations based on root cause
                    st.markdown("#### Recommended Actions")
                    
                    if results['root_cause'] == "Design flaw":
                        st.markdown("""
                        1. Conduct design review with cross-functional team
                        2. Perform FMEA on the affected components
                        3. Update design specifications
                        4. Verify design changes with testing
                        5. Update documentation and notify stakeholders
                        """)
                    elif results['root_cause'] == "Material failure":
                        st.markdown("""
                        1. Review material specifications
                        2. Test alternative materials
                        3. Audit material suppliers
                        4. Implement incoming material testing
                        5. Update material requirements
                        """)
                    elif results['root_cause'] == "Manufacturing defect":
                        st.markdown("""
                        1. Analyze manufacturing process parameters
                        2. Review process control methods
                        3. Improve operator training
                        4. Implement additional in-process checks
                        5. Update manufacturing procedures
                        """)
                    elif results['root_cause'] == "Assembly error":
                        st.markdown("""
                        1. Review assembly processes and fixtures
                        2. Improve work instructions with visual aids
                        3. Implement error-proofing (poka-yoke)
                        4. Enhance operator training
                        5. Add verification steps at critical points
                        """)
                    elif results['root_cause'] == "Component failure":
                        st.markdown("""
                        1. Evaluate component specifications
                        2. Audit component suppliers
                        3. Implement incoming inspection
                        4. Consider alternative components
                        5. Review environmental conditions
                        """)
                    elif results['root_cause'] == "Software bug":
                        st.markdown("""
                        1. Review code with structured walkthrough
                        2. Implement additional unit tests
                        3. Conduct system-level testing
                        4. Review error handling
                        5. Improve version control and testing
                        """)
                    else:
                        st.markdown("""
                        1. Gather additional data on failure mode
                        2. Form cross-functional investigation team
                        3. Implement containment actions
                        4. Develop and verify corrective actions
                        5. Monitor effectiveness after implementation
                        """)
                
                with col2:
                    # Specific failure mode if available
                    if 'failure_mode' in results:
                        st.markdown("#### Specific Failure Mode")
                        st.info(f"**{results['failure_mode']}**")
                    
                    # Quality control method if available
                    if 'quality_control_method' in results:
                        st.markdown("#### Current Quality Control Method")
                        st.info(f"**{results['quality_control_method']}**")
                        
                        st.markdown("#### QC Method Effectiveness")
                        
                        # Evaluate QC method based on return rate
                        if results['current_metrics']['return_rate_30d'] > 5.0:
                            st.error("❌ Current quality control method is not effective")
                        elif results['current_metrics']['return_rate_30d'] > 2.0:
                            st.warning("⚠️ Current quality control method is partially effective")
                        else:
                            st.success("✅ Current quality control method is effective, but improvement is still possible")
                        
                        # Recommend QC improvements
                        st.markdown("#### QC Improvement Recommendations")
                        
                        if results['quality_control_method'].startswith("AQL"):
                            st.markdown("""
                            - Consider tightening AQL level
                            - Implement additional inspection points
                            - Review inspector training and tools
                            - Add functional testing
                            - Implement mistake-proofing
                            """)
                        elif "inspection" in results['quality_control_method'].lower():
                            st.markdown("""
                            - Increase inspection frequency
                            - Add automated inspection where possible
                            - Improve inspection criteria and checklists
                            - Enhance inspector training
                            - Implement process controls upstream
                            """)
                        elif "testing" in results['quality_control_method'].lower():
                            st.markdown("""
                            - Review test protocols and coverage
                            - Improve test equipment calibration
                            - Consider environmental stress testing
                            - Add accelerated life testing
                            - Implement statistical sampling plan
                            """)
                        else:
                            st.markdown("""
                            - Implement Statistical Process Control
                            - Add in-process quality checks
                            - Consider automated inspection/testing
                            - Enhance operator training
                            - Improve documentation and traceability
                            """)
            
            # Create Pareto chart with simulated data if no root cause data
            st.markdown("#### Pareto Analysis of Failure Modes")
            
            # Create sample data for Pareto chart
            if 'failure_mode' in results and 'root_cause' in results:
                # Create simulated data based on the selected failure mode and root cause
                failure_modes = {
                    results['failure_mode']: 35,
                    f"Similar issue to {results['failure_mode']}": 22,
                    f"Related to {results['root_cause']}": 18,
                    "Other related issue 1": 12,
                    "Other related issue 2": 8,
                    "Miscellaneous": 5
                }
            else:
                # Default simulated data
                device_info = DEVICE_TYPES.get(results['device_type'], DEVICE_TYPES['Generic'])
                common_issues = device_info['common_issues']
                
                failure_modes = {}
                for i, issue in enumerate(common_issues):
                    failure_modes[issue] = 40 - (i * 5) if i < len(common_issues) else 5
                
                # Add some more generic ones if needed
                if len(failure_modes) < 5:
                    failure_modes["Assembly defect"] = 10
                    failure_modes["Component failure"] = 8
                    failure_modes["Packaging damage"] = 6
                    failure_modes["Miscellaneous"] = 5
            
            # Store in session state for reuse
            st.session_state.pareto_data = generate_pareto_data(failure_modes)
            pareto_data = st.session_state.pareto_data
            
            # Create Pareto chart
            fig = go.Figure()
            
            # Add bar chart for failure modes
            fig.add_trace(go.Bar(
                x=pareto_data['modes'],
                y=pareto_data['counts'],
                name="Count",
                marker_color=PRIMARY_COLOR
            ))
            
            # Add line chart for cumulative percentage
            fig.add_trace(go.Scatter(
                x=pareto_data['modes'],
                y=pareto_data['cumulative_percent'],
                name="Cumulative %",
                marker=dict(color=DANGER_COLOR),
                line=dict(width=3),
                yaxis="y2"
            ))
            
            # Update layout
            fig.update_layout(
                title="Pareto Analysis of Failure Modes",
                xaxis_title="Failure Mode",
                yaxis=dict(
                    title="Count",
                    titlefont=dict(color=PRIMARY_COLOR),
                    tickfont=dict(color=PRIMARY_COLOR)
                ),
                yaxis2=dict(
                    title="Cumulative %",
                    titlefont=dict(color=DANGER_COLOR),
                    tickfont=dict(color=DANGER_COLOR),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    range=[0, 100]
                ),
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add 80% line for Pareto principle
            fig.add_shape(
                type="line",
                x0=-0.5, y0=80,
                x1=len(pareto_data['modes'])-0.5, y1=80,
                line=dict(color="red", width=2, dash="dash"),
                yref="y2"
            )
            
            # Add annotation for 80% line
            fig.add_annotation(
                x=len(pareto_data['modes'])-1,
                y=80,
                xref="x",
                yref="y2",
                text="80% of Issues",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=-30
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Root cause analysis recommendations
            st.markdown("#### Root Cause Analysis Methodology")
            
            # Fishbone diagram explanation
            st.markdown("""
            **Recommended Approach: Fishbone (Ishikawa) Diagram Analysis**
            
            Focus on these key categories for medical device manufacturing:
            1. **Materials**: Quality issues, specifications, handling, storage
            2. **Methods**: Process parameters, procedures, work instructions
            3. **Machinery**: Equipment maintenance, calibration, tooling
            4. **Measurement**: Testing, gauges, inspection methods
            5. **People**: Training, skills, experience, supervision
            6. **Environment**: Temperature, humidity, cleanliness, layout
            
            Conduct structured investigation with a cross-functional team including engineering, manufacturing, and quality personnel.
            """)
            
        with tabs[3]:
            # Risk Assessment tab
            st.subheader("Risk Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['risk_assessment']['regulatory_risk'],
                    title={"text": "Regulatory Risk"},
                    gauge={
                        "axis": {"range": [None, 5], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 2], "color": SUCCESS_COLOR},
                            {"range": [2, 4], "color": WARNING_COLOR},
                            {"range": [4, 5], "color": DANGER_COLOR}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 4
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['risk_assessment']['brand_risk'],
                    title={"text": "Brand Risk"},
                    gauge={
                        "axis": {"range": [None, 5], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 2], "color": SUCCESS_COLOR},
                            {"range": [2, 4], "color": WARNING_COLOR},
                            {"range": [4, 5], "color": DANGER_COLOR}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 4
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['risk_assessment']['medical_risk'],
                    title={"text": "Medical Risk"},
                    gauge={
                        "axis": {"range": [None, 5], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 2], "color": SUCCESS_COLOR},
                            {"range": [2, 4], "color": WARNING_COLOR},
                            {"range": [4, 5], "color": DANGER_COLOR}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 4
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Risk Factor Analysis")
                st.markdown(f"""
                - **Overall Risk Factor:** {results['risk_assessment']['risk_factor']:.2f}/5.0
                - **Priority Score:** {results['risk_assessment']['priority_score']:.2f}/100
                
                *The priority score combines financial metrics and risk factors to determine the overall priority of this improvement project.*
                """)
            
            with col2:
                st.markdown("#### Risk Mitigation")
                if results['risk_assessment']['medical_risk'] >= 4:
                    st.error("⚠️ High medical risk detected. Immediate action recommended.")
                elif results['risk_assessment']['regulatory_risk'] >= 4:
                    st.warning("⚠️ High regulatory risk detected. Prioritize compliance actions.")
                elif results['risk_assessment']['brand_risk'] >= 4:
                    st.info("⚠️ High brand risk detected. Consider customer communication plan.")
                else:
                    st.success("✅ Risk levels are manageable with standard protocols.")
                    
                # Add device-specific risk information
                st.markdown("#### Device-Specific Risk Considerations")
                
                device_type = results['device_type']
                if device_type == "Mobility Aids":
                    st.markdown("""
                    - Fall risk from device failure
                    - User injury from structural collapse
                    - Stability issues during normal use
                    - Weight capacity limitations
                    """)
                elif device_type == "Cushions & Support":
                    st.markdown("""
                    - Pressure ulcer development
                    - Improper positioning leading to injury
                    - Skin irritation from materials
                    - Support degradation over time
                    """)
                elif device_type == "Monitoring Devices":
                    st.markdown("""
                    - False readings leading to improper treatment
                    - Missed critical notifications
                    - Battery failure during use
                    - Software reliability issues
                    """)
                elif device_type == "Orthopedic Supports":
                    st.markdown("""
                    - Improper support leading to injury
                    - Restriction of circulation
                    - Skin irritation or pressure points
                    - Component failure during use
                    """)
                elif device_type == "Ostomy & Continence":
                    st.markdown("""
                    - Leakage leading to skin damage
                    - Adhesive allergic reactions
                    - Incorrect fit causing discomfort
                    - Material integrity issues
                    """)
                else:
                    st.markdown("""
                    - User injury from device failure
                    - Improper function affecting treatment
                    - Component failure during use
                    - Material integrity issues
                    """)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if results:
            df = export_as_csv(results)
            csv_download = generate_download_link(
                df, 
                f"quality_analysis_{results['sku']}_{datetime.now().strftime('%Y%m%d')}.csv", 
                "📥 Export as CSV"
            )
            st.markdown(csv_download, unsafe_allow_html=True)
    
    with col2:
        if results:
            try:
                pdf_buffer = export_as_pdf(results)
                pdf_data = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                pdf_download = f'<a href="data:application/pdf;base64,{pdf_data}" download="quality_analysis_{results["sku"]}_{datetime.now().strftime("%Y%m%d")}.pdf" class="export-button">📄 Export as PDF</a>'
                st.markdown(pdf_download, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

def display_quality_analysis_form():
    """Display the form for quality issue analysis."""
    with st.form(key="quality_analysis_form"):
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem;">
            <i class="fas fa-clipboard-list"></i> Product Information
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sku = st.text_input("SKU", placeholder="Enter product SKU", help="Unique identifier for the product")
        
        with col2:
            product_type_options = ["B2C - Consumer", "B2B - Professional", "B2B - Healthcare", "OEM"]
            product_type = st.selectbox(
                "Product Type",
                options=product_type_options,
                index=0,
                help="Select the market segment for this product"
            )
        
        with col3:
            # Use the current device type from session state
            device_type = st.selectbox(
                "Device Type",
                options=list(DEVICE_TYPES.keys()),
                index=list(DEVICE_TYPES.keys()).index(st.session_state.get('selected_device_type', 'Generic')),
                help="Select the specific device category"
            )
            
            # Update session state if changed
            if device_type != st.session_state.get('selected_device_type', 'Generic'):
                st.session_state.selected_device_type = device_type
        
        col1, col2 = st.columns(2)
        
        with col1:
            manufacturing_process = st.selectbox(
                "Manufacturing Process",
                options=MANUFACTURING_PROCESSES,
                index=0,
                help="Primary manufacturing process for this product"
            )
        
        with col2:
            issue_description = st.text_input(
                "Issue Description", 
                placeholder="Brief description of the quality issue",
                help="Describe the quality problem being addressed"
            )
        
        # Root cause analysis section
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-search"></i> Root Cause Analysis
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            root_cause = st.selectbox(
                "Primary Root Cause",
                options=["Not determined"] + ROOT_CAUSE_CATEGORIES,
                index=0,
                help="Select the primary root cause category"
            )
            
            # Only use the root cause if it's not "Not determined"
            root_cause = None if root_cause == "Not determined" else root_cause
        
        with col2:
            failure_mode = st.text_input(
                "Specific Failure Mode",
                placeholder="E.g., cracked housing, loose connection",
                help="Specific mode of failure observed"
            )
        
        with col3:
            quality_control_method = st.selectbox(
                "Quality Control Method",
                options=["Not specified"] + QUALITY_CONTROL_METHODS,
                index=0,
                help="Current quality control method being used"
            )
            
            # Only use the QC method if it's not "Not specified"
            quality_control_method = None if quality_control_method == "Not specified" else quality_control_method
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-chart-line"></i> Sales & Returns Data
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sales_30d = st.number_input(
                "Sales (Last 30 Days)",
                min_value=0,
                value=1000,
                step=100,
                help="Number of units sold in the last 30 days"
            )
        
        with col2:
            returns_30d = st.number_input(
                "Returns (Last 30 Days)",
                min_value=0,
                value=50,
                step=10,
                help="Number of units returned in the last 30 days"
            )
        
        with col3:
            annualized_growth = st.slider(
                "Expected Annual Growth (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                format="%.1f%%",
                help="Projected annual growth rate for sales volume"
            )
            
            # Calculate and display current return rate for user reference
            if sales_30d > 0:
                return_rate = (returns_30d / sales_30d) * 100
                st.caption(f"Current return rate: {return_rate:.2f}%")
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-dollar-sign"></i> Cost & Pricing Information
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_unit_cost = st.number_input(
                "Current Unit Cost ($)",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="Manufacturing cost per unit before improvements"
            )
        
        with col2:
            sales_price = st.number_input(
                "Sales Price ($)",
                min_value=0.0,
                value=25.0,
                step=1.0,
                help="Retail or wholesale price per unit"
            )
        
        with col3:
            margin = sales_price - current_unit_cost
            margin_percentage = safe_divide(margin, sales_price, 0) * 100
            
            st.markdown(
                f"""
                <div style="background-color: {CARD_BACKGROUND}; padding: 0.75rem; border-radius: 4px; 
                        border-left: 3px solid {PRIMARY_COLOR}; margin-top: 1.55rem;">
                    <div style="font-size: 0.85rem; color: {TEXT_SECONDARY}; margin-bottom: 0.25rem;">Current Margin</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {TEXT_PRIMARY};">${margin:.2f} ({margin_percentage:.2f}%)</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-tools"></i> Proposed Solution
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fix_cost_upfront = st.number_input(
                "Fix Cost - Upfront ($)",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                help="One-time cost to implement the solution"
            )
        
        with col2:
            fix_cost_per_unit = st.number_input(
                "Fix Cost - Per Unit ($)",
                min_value=0.0,
                value=0.5,
                step=0.1,
                help="Additional cost per unit after implementing the solution"
            )
        
        with col3:
            expected_reduction = st.slider(
                "Expected Return Reduction (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                format="%.1f%%",
                help="Projected percentage reduction in return rate"
            )
        
        with col4:
            solution_confidence = st.slider(
                "Solution Confidence (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=5.0,
                format="%.1f%%",
                help="Confidence level in the effectiveness of the solution"
            )
        
        st.markdown("""
        <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1.5rem;">
            <i class="fas fa-exclamation-triangle"></i> Risk Assessment
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regulatory_risk = st.slider(
                "Regulatory Risk",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="1 = Low, 5 = High. Considers potential regulatory impact."
            )
        
        with col2:
            brand_risk = st.slider(
                "Brand Risk",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Low, 5 = High. Considers impact on brand reputation."
            )
        
        with col3:
            medical_risk = st.slider(
                "Medical Risk",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="1 = Low, 5 = High. Considers potential patient impact."
            )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button("Run Analysis", use_container_width=True)
        
        if submitted:
            if not sku:
                st.error("⚠️ SKU is required")
                return None
            
            if sales_30d <= 0:
                st.error("⚠️ Sales must be greater than zero")
                return None
            
            if current_unit_cost <= 0 or sales_price <= 0:
                st.error("⚠️ Cost and price must be greater than zero")
                return None
            
            with st.spinner("Analyzing quality issue... Please wait"):
                try:
                    results = analyze_quality_issue(
                        sku=sku,
                        product_type=product_type,
                        device_type=device_type,
                        manufacturing_process=manufacturing_process,
                        sales_30d=sales_30d,
                        returns_30d=returns_30d,
                        issue_description=issue_description,
                        root_cause=root_cause,
                        failure_mode=failure_mode if failure_mode else None,
                        quality_control_method=quality_control_method,
                        current_unit_cost=current_unit_cost,
                        fix_cost_upfront=fix_cost_upfront,
                        fix_cost_per_unit=fix_cost_per_unit,
                        sales_price=sales_price,
                        expected_reduction=expected_reduction,
                        solution_confidence=solution_confidence,
                        annualized_growth=annualized_growth,
                        regulatory_risk=regulatory_risk,
                        brand_risk=brand_risk,
                        medical_risk=medical_risk
                    )
                    
                    st.session_state.quality_analysis_results = results
                    st.session_state.analysis_submitted = True
                    
                    # Store variables needed for charts in session state
                    st.session_state.annualized_growth = annualized_growth
                    st.session_state.monthly_net_benefit = safe_divide(
                        results['financial_impact']['adjusted_savings'] - 
                        (results['current_metrics']['annual_sales'] * results['solution_metrics']['fix_cost_per_unit']), 
                        12, 0
                    )
                    st.session_state.cumulative_savings = results['financial_impact']['adjusted_savings'] * 3  # Simplified calculation
                    
                    # Prepare initial message for AI assistant
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    return results
                except Exception as e:
                    st.error(f"Error analyzing quality issue: {str(e)}")
                    logger.exception("Error in quality analysis")
                    return None
        
        return None

def display_ai_assistant(results: Dict[str, Any] = None, module_type: str = "quality"):
    """Display the AI assistant chat interface."""
    
    # Get the appropriate system prompt and chat history based on module type
    system_prompt = get_system_prompt(results, module_type)
    
    # Determine which chat history to use based on module type
    if module_type == "quality" and results:
        chat_history_key = "chat_history"
    elif module_type == "monte_carlo":
        chat_history_key = "monte_carlo_chat_history"
    else:
        chat_history_key = "standalone_chat_history"
    
    # Set the title based on module type
    if module_type == "quality":
        title = "Medical Device Manufacturing & Quality AI Assistant"
    elif module_type == "monte_carlo":
        title = "Risk Analysis AI Assistant"
    else:
        title = "Medical Device Manufacturing AI Assistant"
    
    st.markdown(f"""
    <h3 style="color: #0096C7; border-bottom: 2px solid #0096C7; padding-bottom: 0.5rem; margin-top: 1rem;">
        <i class="fas fa-robot"></i> {title}
    </h3>
    """, unsafe_allow_html=True)
    
    # Add explanation based on the selected device type
    device_type = st.session_state.get('selected_device_type', 'Generic')
    device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
    
    # Add a brief explanation of the assistant's capabilities based on module type
    if module_type == "quality":
        explanation = f"Ask questions about quality issues, manufacturing processes, and troubleshooting for {device_type} devices."
    elif module_type == "monte_carlo":
        explanation = f"Ask questions about risk analysis, manufacturing process variability, and implementation strategies for {device_type} devices."
    else:
        explanation = f"Ask questions about manufacturing processes, quality control, and troubleshooting for medical devices, particularly {device_type}."
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; border-left: 3px solid #0096C7;">
        <strong>AI features:</strong> {explanation}
        <br><br>
        <strong>Device expertise:</strong> {device_info['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # Get the appropriate chat history
    chat_history = getattr(st.session_state, chat_history_key)
    
    # Display chat history in a scrollable container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # If no chat history yet, add a welcome message
    if not chat_history:
        if module_type == "quality" and results:
            st.markdown(f"""
            <div class="assistant-bubble">
                <strong>AI Assistant:</strong> I've analyzed the quality data for {results['sku']}. The return rate is {results['current_metrics']['return_rate_30d']:.2f}% with an estimated annual loss of {format_currency(results['financial_impact']['annual_loss'])}. 
                <br><br>
                Based on the analysis, my recommendation is: <strong>{results['recommendation']}</strong>
                <br><br>
                For this {results['device_type']} using {results['manufacturing_process']}, I can help you identify specific causes and troubleshooting steps. What specific aspects would you like help with?
            </div>
            """, unsafe_allow_html=True)
        elif module_type == "monte_carlo" and results:
            st.markdown(f"""
            <div class="assistant-bubble">
                <strong>AI Assistant:</strong> I've analyzed your Monte Carlo simulation results for your {st.session_state.get('selected_device_type', 'Generic')} device quality improvement project.
                <br><br>
                There's a {results['probability_metrics']['prob_positive_roi']:.1f}% probability of achieving positive ROI, with a median ROI of {results['roi_stats']['median']:.2f}%.
                <br><br>
                Given the manufacturing variability in {st.session_state.get('selected_device_type', 'Generic')} products, what specific risk factors would you like to discuss?
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-bubble">
                <strong>AI Assistant:</strong> Hello! I'm your Medical Device Manufacturing and Quality Expert specializing in {device_type} devices.
                <br><br>
                I can help with:
                <ul>
                    <li>Manufacturing process troubleshooting for {device_type}</li>
                    <li>Quality issues with {', '.join(device_info['examples'][:3])}</li>
                    <li>Root cause analysis for common problems like {', '.join(device_info['common_issues'][:3])}</li>
                    <li>Process control recommendations</li>
                    <li>Quality control and inspection methods</li>
                </ul>
                
                How can I assist you with your {device_type} manufacturing or quality needs today?
            </div>
            """, unsafe_allow_html=True)
    else:
        # Display actual chat history
        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble"><strong>AI Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area with suggested prompts
    user_input = st.text_input(
        f"Ask about {device_type} manufacturing and quality:",
        key=f"chat_input_{module_type}", 
        placeholder="Type your question here or use the suggested questions below"
    )
    
    # Create device-specific suggested prompts
    device_specific_prompts = {
        "Mobility Aids": [
            "What are common causes of frame instability?",
            "How to troubleshoot wheel alignment issues?",
            "Best QC methods for weight capacity testing?"
        ],
        "Cushions & Support": [
            "What causes premature foam compression?",
            "How to prevent cover material tears?",
            "Best testing methods for pressure distribution?"
        ],
        "Monitoring Devices": [
            "Common causes of sensor drift in BPMs?",
            "How to improve PCB assembly quality?",
            "Best calibration verification methods?"
        ],
        "Orthopedic Supports": [
            "What causes hinge mechanism failures?",
            "How to prevent strap attachment issues?",
            "Quality tests for material durability?"
        ],
        "Ostomy & Continence": [
            "What causes adhesive bond failures?",
            "How to improve seal integrity?",
            "Best leak testing methods?"
        ],
        "Medical Software": [
            "Common causes of app crashes?",
            "How to improve testing for medical apps?",
            "Best validation methods for software updates?"
        ],
        "Respiratory Aids": [
            "What causes flow rate inconsistencies?",
            "How to troubleshoot motor noise issues?",
            "Quality checks for tubing connections?"
        ],
        "Home Care Equipment": [
            "Common causes of motor failures?",
            "How to improve structural stability?",
            "Best load testing methods?"
        ],
        "Therapeutic Devices": [
            "What causes controller malfunctions?",
            "How to prevent electrode degradation?",
            "Best output verification methods?"
        ],
        "Generic": [
            "Common injection molding defects?",
            "How to improve assembly quality?",
            "Best statistical process control methods?"
        ]
    }
    
    # Get the appropriate suggested prompts based on device type
    suggested_prompts = device_specific_prompts.get(device_type, device_specific_prompts["Generic"])
    
    # Add manufacturing process specific prompts if results are available
    if results and 'manufacturing_process' in results:
        manufacturing_process = results['manufacturing_process']
        
        if manufacturing_process == "Injection molding":
            suggested_prompts = [
                "What causes sink marks in injection molded parts?",
                "How to troubleshoot short shots?",
                "Best process controls for plastic molding?"
            ]
        elif manufacturing_process in ["Metal fabrication", "Metal machining", "Welding & assembly"]:
            suggested_prompts = [
                "Common causes of metal part dimensional variation?",
                "How to prevent weld failures?",
                "Best quality checks for metal components?"
            ]
        elif manufacturing_process == "PCB assembly":
            suggested_prompts = [
                "What causes solder joint failures?",
                "How to prevent component misalignment?",
                "Best inspection methods for PCBs?"
            ]
        elif manufacturing_process in ["Fabric cutting & sewing", "Foam molding"]:
            suggested_prompts = [
                "Common causes of seam failures?",
                "How to improve foam molding consistency?",
                "Best quality checks for textile products?"
            ]
    
    # Suggested prompt buttons in columns for better spacing
    col1, col2, col3 = st.columns(3)
    
    # Different suggested prompts based on context
    with col1:
        if st.button(suggested_prompts[0], key=f"prompt1_{module_type}", use_container_width=True):
            user_input = suggested_prompts[0]
    
    with col2:
        if st.button(suggested_prompts[1], key=f"prompt2_{module_type}", use_container_width=True):
            user_input = suggested_prompts[1]
    
    with col3:
        if st.button(suggested_prompts[2], key=f"prompt3_{module_type}", use_container_width=True):
            user_input = suggested_prompts[2]
    
    # Add a submit button
    if st.button("Send", key=f"send_msg_btn_{module_type}", use_container_width=True):
        if user_input:
            # Add user message to chat history
            getattr(st.session_state, chat_history_key).append({"role": "user", "content": user_input})
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in getattr(st.session_state, chat_history_key)
            ]
            
            # Show spinner while waiting for API response
            with st.spinner(f"{title} is thinking..."):
                ai_resp = call_openai_api(messages)
            
            # Add AI response to chat history
            getattr(st.session_state, chat_history_key).append({"role": "assistant", "content": ai_resp})
            st.rerun()
    
    # Add a clear conversation button
    if getattr(st.session_state, chat_history_key):
        if st.button(
            "Clear Conversation", 
            key=f"clear_chat_btn_{module_type}"
        ):
            setattr(st.session_state, chat_history_key, [])
            st.rerun()
    
    # Display API connection status
    api_key = st.secrets.get("openai_api_key", None)
    if api_key:
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem; font-size: 0.8rem; color: #6c757d;">
            <i class="fas fa-circle" style="color: #40916C;"></i> AI assistant connected
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem; font-size: 0.8rem; color: #6c757d;">
            <i class="fas fa-circle" style="color: #E76F51;"></i> AI assistant not connected - API key missing
            <p>Please contact alexander.popoff@vivehealth.com for support.</p>
        </div>
        """, unsafe_allow_html=True)

def display_process_control_analysis():
    """Display the process control analysis UI."""
    st.markdown("### 📈 Process Control Analysis")
    st.markdown("""
    Analyze manufacturing process controls and variability for quality improvement.
    """)
    
    # Get device type info
    device_type = st.session_state.get('selected_device_type', 'Generic')
    device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
    
    # Get common manufacturing processes for selected device type
    common_processes = device_info['manufacturing_processes']
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form(key="process_control_form"):
            st.markdown("#### Manufacturing Process Information")
            
            manufacturing_process = st.selectbox(
                "Primary Manufacturing Process",
                options=MANUFACTURING_PROCESSES,
                index=MANUFACTURING_PROCESSES.index(common_processes[0]) if common_processes[0] in MANUFACTURING_PROCESSES else 0,
                help="Primary manufacturing process being analyzed"
            )
            
            process_step = st.text_input(
                "Process Step",
                placeholder="E.g., Injection molding, Assembly, Welding",
                help="Specific manufacturing step being analyzed"
            )
            
            # Different parameters based on manufacturing process
            if manufacturing_process == "Injection molding":
                param1_name = "Mold Temperature (°C)"
                param1_default = 80
                param2_name = "Injection Pressure (MPa)"
                param2_default = 100
                param3_name = "Cooling Time (sec)"
                param3_default = 15
            elif manufacturing_process in ["Metal fabrication", "Metal machining"]:
                param1_name = "Cutting Speed (m/min)"
                param1_default = 150
                param2_name = "Feed Rate (mm/rev)"
                param2_default = 0.2
                param3_name = "Tool Life (%)"
                param3_default = 80
            elif manufacturing_process == "PCB assembly":
                param1_name = "Soldering Temperature (°C)"
                param1_default = 245
                param2_name = "Conveyor Speed (cm/min)"
                param2_default = 60
                param3_name = "Preheat Time (sec)"
                param3_default = 30
            elif manufacturing_process in ["Fabric cutting & sewing", "Foam molding"]:
                param1_name = "Material Thickness (mm)"
                param1_default = 5
                param2_name = "Cutting Pressure (kPa)"
                param2_default = 200
                param3_name = "Feed Rate (mm/sec)"
                param3_default = 10
            else:
                param1_name = "Parameter 1"
                param1_default = 100
                param2_name = "Parameter 2"
                param2_default = 50
                param3_name = "Parameter 3"
                param3_default = 25
            
            # Process parameters
            st.markdown("#### Process Parameters")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                param1 = st.number_input(
                    param1_name,
                    min_value=0.0,
                    value=float(param1_default),
                    step=1.0
                )
                
                param1_lower = st.number_input(
                    f"{param1_name} Lower Limit",
                    min_value=0.0,
                    value=float(param1_default * 0.9),
                    step=1.0
                )
                
                param1_upper = st.number_input(
                    f"{param1_name} Upper Limit",
                    min_value=0.0,
                    value=float(param1_default * 1.1),
                    step=1.0
                )
            
            with col_b:
                param2 = st.number_input(
                    param2_name,
                    min_value=0.0,
                    value=float(param2_default),
                    step=1.0
                )
                
                param2_lower = st.number_input(
                    f"{param2_name} Lower Limit",
                    min_value=0.0,
                    value=float(param2_default * 0.9),
                    step=1.0
                )
                
                param2_upper = st.number_input(
                    f"{param2_name} Upper Limit",
                    min_value=0.0,
                    value=float(param2_default * 1.1),
                    step=1.0
                )
            
            # Additional parameters and process capability
            st.markdown("#### Process Capability")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                param3 = st.number_input(
                    param3_name,
                    min_value=0.0,
                    value=float(param3_default),
                    step=1.0
                )
                
                cpk_value = st.number_input(
                    "Process Capability (Cpk)",
                    min_value=0.0,
                    max_value=3.0,
                    value=1.33,
                    step=0.1,
                    help="Current process capability index"
                )
            
            with col_b:
                defect_rate = st.number_input(
                    "Current Defect Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=2.5,
                    step=0.1,
                    help="Current defect rate percentage"
                )
                
                sample_size = st.number_input(
                    "Sample Size",
                    min_value=1,
                    value=100,
                    step=10,
                    help="Number of samples used for analysis"
                )
            
            submitted = st.form_submit_button("Analyze Process Control")
            
            if submitted:
                with st.spinner("Analyzing process control..."):
                    try:
                        # Generate simulated process control data
                        np.random.seed(42)  # For reproducibility
                        
                        # Generate sample data for parameter 1
                        param1_std = (param1_upper - param1_lower) / (6 * cpk_value)
                        param1_samples = np.random.normal(param1, param1_std, sample_size)
                        
                        # Generate sample data for parameter 2
                        param2_std = (param2_upper - param2_lower) / (6 * cpk_value)
                        param2_samples = np.random.normal(param2, param2_std, sample_size)
                        
                        # Calculate process capability indices
                        param1_cp = (param1_upper - param1_lower) / (6 * np.std(param1_samples))
                        param1_cpk = min(
                            (np.mean(param1_samples) - param1_lower) / (3 * np.std(param1_samples)),
                            (param1_upper - np.mean(param1_samples)) / (3 * np.std(param1_samples))
                        )
                        
                        param2_cp = (param2_upper - param2_lower) / (6 * np.std(param2_samples))
                        param2_cpk = min(
                            (np.mean(param2_samples) - param2_lower) / (3 * np.std(param2_samples)),
                            (param2_upper - np.mean(param2_samples)) / (3 * np.std(param2_samples))
                        )
                        
                        # Generate control chart data (mock data for X-bar chart)
                        num_subgroups = 25
                        subgroup_size = 4
                        xbar_data = []
                        
                        for i in range(num_subgroups):
                            # Add some patterns to make it interesting
                            if i > 15:
                                # Trend pattern in later points
                                shift = (i - 15) * param1_std * 0.2
                            else:
                                shift = 0
                                
                            subgroup = np.random.normal(param1 + shift, param1_std, subgroup_size)
                            xbar_data.append(np.mean(subgroup))
                        
                        # Calculate control limits
                        overall_mean = np.mean(xbar_data)
                        overall_std = np.std(param1_samples)
                        
                        ucl = overall_mean + 3 * overall_std / np.sqrt(subgroup_size)
                        lcl = overall_mean - 3 * overall_std / np.sqrt(subgroup_size)
                        
                        # Store in session state for reuse
                        st.session_state.process_control_data = {
                            "manufacturing_process": manufacturing_process,
                            "process_step": process_step,
                            "param1_name": param1_name,
                            "param1_samples": param1_samples.tolist(),
                            "param1_mean": float(np.mean(param1_samples)),
                            "param1_std": float(np.std(param1_samples)),
                            "param1_cp": float(param1_cp),
                            "param1_cpk": float(param1_cpk),
                            "param1_lower": param1_lower,
                            "param1_upper": param1_upper,
                            "param2_name": param2_name,
                            "param2_samples": param2_samples.tolist(),
                            "param2_mean": float(np.mean(param2_samples)),
                            "param2_std": float(np.std(param2_samples)),
                            "param2_cp": float(param2_cp),
                            "param2_cpk": float(param2_cpk),
                            "param2_lower": param2_lower,
                            "param2_upper": param2_upper,
                            "param3_name": param3_name,
                            "param3_value": param3,
                            "defect_rate": defect_rate,
                            "sample_size": sample_size,
                            "xbar_data": xbar_data,
                            "ucl": float(ucl),
                            "lcl": float(lcl),
                            "overall_mean": float(overall_mean)
                        }
                        
                    except Exception as e:
                        st.error(f"Error analyzing process control: {str(e)}")
                        logger.exception("Error in process control analysis")
    
    with col2:
        if 'process_control_data' in st.session_state and st.session_state.process_control_data:
            data = st.session_state.process_control_data
            
            st.markdown("#### Process Capability Analysis")
            
            # Summary metrics
            col_a, col_b = st.columns(2)
            
            with col_a:
                cp_color = generate_color_scale(data['param1_cp'], 1.33, 1.0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{data['param1_name']} Capability (Cp)</div>
                    <div class="metric-value" style="color: {cp_color};">{data['param1_cp']:.2f}</div>
                    <div class="metric-subvalue">Cpk: {data['param1_cpk']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                defect_color = generate_color_scale(5.0 - min(data['defect_rate'], 5.0), 4.0, 1.0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Defect Rate</div>
                    <div class="metric-value" style="color: {defect_color};">{data['defect_rate']:.2f}%</div>
                    <div class="metric-subvalue">Based on {data['sample_size']} samples</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Process capability chart for parameter 1
            st.markdown(f"#### {data['param1_name']} Process Capability")
            
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data['param1_samples'],
                opacity=0.75,
                marker_color=SECONDARY_COLOR,
                name="Process Distribution"
            ))
            
            # Add specification lines
            fig.add_shape(
                type="line",
                x0=data['param1_lower'], y0=0,
                x1=data['param1_lower'], y1=20,
                line=dict(color="red", width=2, dash="dash"),
                name="Lower Spec"
            )
            
            fig.add_shape(
                type="line",
                x0=data['param1_upper'], y0=0,
                x1=data['param1_upper'], y1=20,
                line=dict(color="red", width=2, dash="dash"),
                name="Upper Spec"
            )
            
            # Add mean line
            fig.add_shape(
                type="line",
                x0=data['param1_mean'], y0=0,
                x1=data['param1_mean'], y1=20,
                line=dict(color="green", width=2),
                name="Process Mean"
            )
            
            # Add annotations for Cp and Cpk
            fig.add_annotation(
                x=data['param1_lower'] + (data['param1_upper'] - data['param1_lower']) * 0.2,
                y=15,
                text=f"Cp: {data['param1_cp']:.2f}<br>Cpk: {data['param1_cpk']:.2f}",
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            fig.update_layout(
                title=f"Process Capability for {data['param1_name']}",
                xaxis_title=data['param1_name'],
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Control chart (X-bar chart)
            st.markdown("#### Statistical Process Control Chart")
            
            fig = go.Figure()
            
            # Add X-bar values
            fig.add_trace(go.Scatter(
                x=list(range(1, len(data['xbar_data']) + 1)),
                y=data['xbar_data'],
                mode='lines+markers',
                name='X-bar',
                line=dict(color=PRIMARY_COLOR, width=2),
                marker=dict(size=8)
            ))
            
            # Add center line (process mean)
            fig.add_shape(
                type="line",
                x0=1, y0=data['overall_mean'],
                x1=len(data['xbar_data']), y1=data['overall_mean'],
                line=dict(color="green", width=2),
                name="Center Line"
            )
            
            # Add upper control limit
            fig.add_shape(
                type="line",
                x0=1, y0=data['ucl'],
                x1=len(data['xbar_data']), y1=data['ucl'],
                line=dict(color="red", width=2, dash="dash"),
                name="UCL"
            )
            
            # Add lower control limit
            fig.add_shape(
                type="line",
                x0=1, y0=data['lcl'],
                x1=len(data['xbar_data']), y1=data['lcl'],
                line=dict(color="red", width=2, dash="dash"),
                name="LCL"
            )
            
            # Add annotations for control limits
            fig.add_annotation(
                x=len(data['xbar_data']),
                y=data['ucl'],
                text=f"UCL: {data['ucl']:.2f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="red")
            )
            
            fig.add_annotation(
                x=len(data['xbar_data']),
                y=data['lcl'],
                text=f"LCL: {data['lcl']:.2f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="red")
            )
            
            fig.update_layout(
                title=f"X-bar Control Chart for {data['param1_name']}",
                xaxis_title="Sample Group",
                yaxis_title=data['param1_name'],
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Process improvement recommendations
            st.markdown("#### Process Control Recommendations")
            
            if data['param1_cpk'] < 1.0:
                st.warning("⚠️ Process is not capable (Cpk < 1.0). Process improvement is required.")
                
                if data['manufacturing_process'] == "Injection molding":
                    st.markdown("""
                    **Recommended Actions:**
                    1. Verify mold temperature control system and sensors
                    2. Check material drying parameters and conditions
                    3. Implement cavity pressure sensors for better process control
                    4. Review gate and runner design for balanced filling
                    5. Consider scientific molding approach with comprehensive DOE
                    """)
                elif data['manufacturing_process'] in ["Metal fabrication", "Metal machining"]:
                    st.markdown("""
                    **Recommended Actions:**
                    1. Implement tool wear monitoring system
                    2. Review fixture design and clamping force
                    3. Check machine alignment and calibration
                    4. Optimize cutting parameters through DOE
                    5. Consider improved cooling/lubrication methods
                    """)
                elif data['manufacturing_process'] == "PCB assembly":
                    st.markdown("""
                    **Recommended Actions:**
                    1. Verify reflow oven temperature profile
                    2. Check solder paste application consistency
                    3. Implement automated optical inspection (AOI)
                    4. Review component placement accuracy
                    5. Control ambient conditions (temperature, humidity)
                    """)
                elif data['manufacturing_process'] in ["Fabric cutting & sewing", "Foam molding"]:
                    st.markdown("""
                    **Recommended Actions:**
                    1. Verify material property consistency
                    2. Check cutting tool condition and replacement schedule
                    3. Implement tension control for fabric handling
                    4. Control environmental conditions
                    5. Enhance operator training and work instructions
                    """)
                else:
                    st.markdown("""
                    **Recommended Actions:**
                    1. Identify and address special cause variation
                    2. Conduct Measurement System Analysis (MSA)
                    3. Implement Statistical Process Control (SPC)
                    4. Review process parameters and tolerances
                    5. Consider process redesign if capability cannot be improved
                    """)
            elif data['param1_cpk'] < 1.33:
                st.info("ℹ️ Process is marginally capable (1.0 < Cpk < 1.33). Improvement is recommended.")
                st.markdown("""
                **Recommended Actions:**
                1. Continue monitoring the process with SPC charts
                2. Implement targeted improvements to reduce variation
                3. Review and optimize process parameters
                4. Enhance preventive maintenance procedures
                5. Conduct operator training refreshers
                """)
            else:
                st.success("✅ Process is capable (Cpk > 1.33). Maintain current controls.")
                st.markdown("""
                **Actions to Maintain Process:**
                1. Document current process parameters and controls
                2. Continue regular monitoring with SPC
                3. Implement preventive maintenance schedule
                4. Standardize setup procedures
                5. Conduct periodic capability studies
                """)
            
            # Show potential improvement with better process control
            st.markdown("#### Potential Quality Improvement")
            
            # Calculate potential defect rate with improved Cpk
            improved_cpk = 1.67  # Target for improved process
            current_dpmo = 3.4 * (10 ** (6 - (data['param1_cpk'] * 3)))  # Approximation
            improved_dpmo = 3.4 * (10 ** (6 - (improved_cpk * 3)))  # Approximation
            
            current_defect_pct = min(data['defect_rate'], current_dpmo / 10000)  # Cap at calculated rate
            improved_defect_pct = improved_dpmo / 10000
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=["Current Process", "Improved Process"],
                y=[current_defect_pct, improved_defect_pct],
                text=[f"{current_defect_pct:.2f}%", f"{improved_defect_pct:.4f}%"],
                textposition="auto",
                marker_color=[WARNING_COLOR, SUCCESS_COLOR]
            ))
            
            fig.update_layout(
                title="Potential Defect Rate Improvement",
                xaxis_title="Process State",
                yaxis_title="Defect Rate (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Potential cost savings
            if 'quality_analysis_results' in st.session_state and st.session_state.quality_analysis_results:
                quality_results = st.session_state.quality_analysis_results
                
                # Calculate potential savings
                annual_returns = quality_results['current_metrics']['annual_returns']
                loss_per_return = quality_results['current_metrics']['loss_per_return']
                
                # Simple estimate: improvement in defect rate translates to improvement in returns
                if current_defect_pct > 0:
                    potential_reduction = (current_defect_pct - improved_defect_pct) / current_defect_pct
                    potential_returns_prevented = annual_returns * potential_reduction
                    potential_savings = potential_returns_prevented * loss_per_return
                    
                    st.markdown(f"""
                    **Potential Annual Savings: {format_currency(potential_savings)}**
                    
                    By improving process capability from Cpk {data['param1_cpk']:.2f} to Cpk {improved_cpk:.2f}, 
                    you could prevent approximately {potential_returns_prevented:.0f} returns per year.
                    """)

def run_monte_carlo_simulation_ui():
    """Display the Monte Carlo simulation UI."""
    st.markdown("### 🎲 Monte Carlo Simulation")
    st.markdown("""
    Run a Monte Carlo simulation to understand risks and probabilities in your quality improvement project.
    """)
    
    # Get device type info for context-specific parameters
    device_type = st.session_state.get('selected_device_type', 'Generic')
    device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form(key="monte_carlo_form"):
            st.markdown("#### Device Information")
            
            # Display the selected device type
            st.info(f"**Device Type:** {device_type}")
            st.caption(f"Common issues: {', '.join(device_info['common_issues'][:3])}")
            
            manufacturing_process = st.selectbox(
                "Manufacturing Process",
                options=MANUFACTURING_PROCESSES,
                index=MANUFACTURING_PROCESSES.index(device_info['manufacturing_processes'][0]) if len(device_info['manufacturing_processes']) > 0 and device_info['manufacturing_processes'][0] in MANUFACTURING_PROCESSES else 0,
                help="Primary manufacturing process for this device"
            )
            
            st.markdown("#### Base Parameters")
            
            base_unit_cost = st.number_input(
                "Base Unit Cost ($)",
                min_value=0.01,
                value=10.0,
                step=1.0
            )
            
            base_sales_price = st.number_input(
                "Base Sales Price ($)",
                min_value=0.01,
                value=25.0,
                step=1.0
            )
            
            base_monthly_sales = st.number_input(
                "Base Monthly Sales (Units)",
                min_value=1,
                value=1000,
                step=100
            )
            
            base_return_rate = st.number_input(
                "Base Return Rate (%)",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.5
            )
            
            st.markdown("#### Solution Parameters")
            
            fix_cost_upfront = st.number_input(
                "Fix Cost - Upfront ($)",
                min_value=0.0,
                value=5000.0,
                step=500.0
            )
            
            fix_cost_per_unit = st.number_input(
                "Fix Cost - Per Unit ($)",
                min_value=0.0,
                value=0.5,
                step=0.1
            )
            
            expected_reduction = st.number_input(
                "Expected Return Reduction (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0
            )
            
            st.markdown("#### Simulation Settings")
            
            iterations = st.slider(
                "Number of Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            with st.expander("Manufacturing Variability Settings"):
                # Adjust default variability based on device type and manufacturing process
                if manufacturing_process == "Injection molding":
                    default_cost_var = 0.08
                    default_return_var = 0.20
                    default_reduction_var = 0.25
                elif manufacturing_process in ["Metal fabrication", "Metal machining"]:
                    default_cost_var = 0.06
                    default_return_var = 0.15
                    default_reduction_var = 0.20
                elif manufacturing_process == "PCB assembly":
                    default_cost_var = 0.07
                    default_return_var = 0.25
                    default_reduction_var = 0.30
                elif manufacturing_process in ["Fabric cutting & sewing", "Foam molding"]:
                    default_cost_var = 0.10
                    default_return_var = 0.25
                    default_reduction_var = 0.30
                else:
                    default_cost_var = 0.05
                    default_return_var = 0.15
                    default_reduction_var = 0.20
                
                st.markdown(f"#### Variability for {device_type} using {manufacturing_process}")
                
                cost_std_dev = st.slider(
                    "Manufacturing Cost Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=default_cost_var,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base cost"
                )
                
                price_std_dev = st.slider(
                    "Price Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.03,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base price"
                )
                
                sales_std_dev = st.slider(
                    "Sales Volume Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.10,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base sales"
                )
                
                return_std_dev = st.slider(
                    "Return Rate Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=default_return_var,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of base return rate"
                )
                
                reduction_std_dev = st.slider(
                    "Solution Effectiveness Variability",
                    min_value=0.01,
                    max_value=0.50,
                    value=default_reduction_var,
                    step=0.01,
                    format="%.2f",
                    help="Standard deviation as a fraction of expected reduction"
                )
            
            submitted = st.form_submit_button("Run Simulation")
            
            if submitted:
                with st.spinner(f"Running Monte Carlo simulation with {iterations} iterations..."):
                    try:
                        results = run_monte_carlo_simulation(
                            base_unit_cost=base_unit_cost,
                            base_sales_price=base_sales_price,
                            base_monthly_sales=base_monthly_sales,
                            base_return_rate=base_return_rate,
                            fix_cost_upfront=fix_cost_upfront,
                            fix_cost_per_unit=fix_cost_per_unit,
                            expected_reduction=expected_reduction,
                            iterations=iterations,
                            cost_std_dev=cost_std_dev,
                            price_std_dev=price_std_dev,
                            sales_std_dev=sales_std_dev,
                            return_std_dev=return_std_dev,
                            reduction_std_dev=reduction_std_dev
                        )
                        
                        st.session_state.monte_carlo_scenario = results
                    except Exception as e:
                        st.error(f"Error running Monte Carlo simulation: {str(e)}")
                        logger.exception("Error in Monte Carlo simulation")
    
    with col2:
        if hasattr(st.session_state, 'monte_carlo_scenario') and st.session_state.monte_carlo_scenario:
            results = st.session_state.monte_carlo_scenario
            
            st.markdown("#### Simulation Results")
            
            # Probability metrics
            col_a, col_b = st.columns(2)
            
            with col_a:
                prob_positive_roi_color = generate_color_scale(
                    results['probability_metrics']['prob_positive_roi'], 90, 50
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability of Positive ROI</div>
                    <div class="metric-value" style="color: {prob_positive_roi_color};">{results['probability_metrics']['prob_positive_roi']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                prob_payback_1yr_color = generate_color_scale(
                    results['probability_metrics']['prob_payback_1yr'], 90, 50
                )
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability of Payback < 1 Year</div>
                    <div class="metric-value" style="color: {prob_payback_1yr_color};">{results['probability_metrics']['prob_payback_1yr']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ROI distribution
            st.markdown("#### ROI Distribution")
            
            roi_hist = results['roi_stats']['histogram']['counts']
            roi_bins = results['roi_stats']['histogram']['bins']
            roi_bin_centers = [(roi_bins[i] + roi_bins[i+1])/2 for i in range(len(roi_bins)-1)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=roi_bin_centers,
                y=roi_hist,
                marker_color=PRIMARY_COLOR,
                name="ROI Distribution"
            ))
            
            # Add a vertical line at ROI = 0
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=0, y1=max(roi_hist),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="ROI Distribution",
                xaxis_title="ROI (%)",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Payback distribution
            st.markdown("#### Payback Period Distribution")
            
            payback_hist = results['payback_stats']['histogram']['counts']
            payback_bins = results['payback_stats']['histogram']['bins']
            payback_bin_centers = [(payback_bins[i] + payback_bins[i+1])/2 for i in range(len(payback_bins)-1)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=payback_bin_centers,
                y=payback_hist,
                marker_color=SECONDARY_COLOR,
                name="Payback Distribution"
            ))
            
            # Add a vertical line at Payback = 12 months
            fig.add_shape(
                type="line",
                x0=12, y0=0,
                x1=12, y1=max(payback_hist),
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="Payback Period Distribution",
                xaxis_title="Payback Period (Months)",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Manufacturing-specific insights based on device type and process
            st.markdown(f"#### {device_type} Manufacturing Insights")
            
            device_specific_insights = {
                "Mobility Aids": [
                    "Frame stability variability is a key factor",
                    "Welding process control is critical for consistent quality",
                    "Component assembly variations affect overall quality"
                ],
                "Cushions & Support": [
                    "Foam density variations significantly impact performance",
                    "Cover material and stitching consistency are critical",
                    "Environmental factors during production affect quality"
                ],
                "Monitoring Devices": [
                    "PCB assembly process control is critical for electronics reliability",
                    "Sensor calibration variability directly impacts return rates",
                    "Software validation effectiveness varies with complexity"
                ],
                "Orthopedic Supports": [
                    "Material cutting precision affects fit and comfort",
                    "Hinge and strapping assembly variability is significant",
                    "Material batch variations impact overall quality"
                ],
                "Ostomy & Continence": [
                    "Seal integrity variations significantly impact performance",
                    "Adhesive application consistency is critical",
                    "Material welding/bonding process control is essential"
                ],
                "Medical Software": [
                    "Testing coverage variability impacts bug detection",
                    "User interface consistency affects user satisfaction",
                    "Integration testing thoroughness varies with complexity"
                ],
                "Respiratory Aids": [
                    "Air flow control component consistency is critical",
                    "Assembly precision directly impacts device performance",
                    "Motor performance variability affects reliability"
                ],
                "Therapeutic Devices": [
                    "Electronics assembly quality directly impacts functionality",
                    "Sensor calibration consistency affects therapeutic effectiveness",
                    "Housing assembly variations impact durability"
                ],
                "Generic": [
                    "Manufacturing process variability is a key factor",
                    "Component quality consistency affects overall performance",
                    "Assembly precision varies with complexity"
                ]
            }
            
            # Process-specific insights
            process_specific_insights = {
                "Injection molding": [
                    "Material drying variations impact part quality",
                    "Mold temperature consistency is critical",
                    "Gate freeze-off timing affects part dimensions"
                ],
                "Metal fabrication": [
                    "Material property variations affect consistency",
                    "Tool wear rates impact dimensional accuracy",
                    "Fixture stability variations affect precision"
                ],
                "PCB assembly": [
                    "Solder paste application consistency is critical",
                    "Component placement accuracy varies with speed",
                    "Reflow profile control affects joint quality"
                ],
                "Fabric cutting & sewing": [
                    "Material tension variations affect dimension accuracy",
                    "Cutting tool wear impacts edge quality",
                    "Sewing tension consistency affects seam strength"
                ],
                "Foam molding": [
                    "Material mixing variations impact density",
                    "Mold temperature control affects cell structure",
                    "Curing time consistency impacts resilience"
                ]
            }
            
            # Display device-specific insights
            device_insights = device_specific_insights.get(device_type, device_specific_insights["Generic"])
            for insight in device_insights:
                st.markdown(f"- {insight}")
            
            # Display process-specific insights if available
            if manufacturing_process in process_specific_insights:
                st.markdown(f"#### {manufacturing_process} Process Insights")
                process_insights = process_specific_insights[manufacturing_process]
                for insight in process_insights:
                    st.markdown(f"- {insight}")
            
            # Statistics table
            st.markdown("#### Statistical Summary")
            
            stats_df = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "10th Percentile", "25th Percentile", "75th Percentile", "90th Percentile"],
                "ROI (%)": [
                    f"{results['roi_stats']['mean']:.2f}",
                    f"{results['roi_stats']['median']:.2f}",
                    f"{results['roi_stats']['std_dev']:.2f}",
                    f"{results['roi_stats']['min']:.2f}",
                    f"{results['roi_stats']['max']:.2f}",
                    f"{results['roi_stats']['percentiles']['p10']:.2f}",
                    f"{results['roi_stats']['percentiles']['p25']:.2f}",
                    f"{results['roi_stats']['percentiles']['p75']:.2f}",
                    f"{results['roi_stats']['percentiles']['p90']:.2f}"
                ],
                "Payback (Months)": [
                    f"{results['payback_stats']['mean']:.2f}",
                    f"{results['payback_stats']['median']:.2f}",
                    f"{results['payback_stats']['std_dev']:.2f}",
                    f"{results['payback_stats']['min']:.2f}",
                    f"{results['payback_stats']['max']:.2f}",
                    f"{results['payback_stats']['percentiles']['p10']:.2f}",
                    f"{results['payback_stats']['percentiles']['p25']:.2f}",
                    f"{results['payback_stats']['percentiles']['p75']:.2f}",
                    f"{results['payback_stats']['percentiles']['p90']:.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Insights section
            st.markdown("#### Simulation Insights")
            
            if results['probability_metrics']['prob_positive_roi'] >= 90:
                st.success("✅ This project has a **high probability of success** with over 90% chance of positive ROI.")
            elif results['probability_metrics']['prob_positive_roi'] >= 70:
                st.info("ℹ️ This project has a **good probability of success** with over 70% chance of positive ROI.")
            elif results['probability_metrics']['prob_positive_roi'] >= 50:
                st.warning("⚠️ This project has a **moderate risk** with just over 50% chance of positive ROI.")
            else:
                st.error("❌ This project has a **high risk** with less than 50% chance of positive ROI.")
            
            st.markdown(f"""
            **Key Insights:**
            
            - There is a **{results['probability_metrics']['prob_positive_roi']:.1f}%** probability of achieving a positive ROI
            - There is a **{results['probability_metrics']['prob_payback_1yr']:.1f}%** probability of achieving payback within 1 year
            - The median ROI is **{results['roi_stats']['median']:.2f}%**
            - The median payback period is **{results['payback_stats']['median']:.2f}** months
            
            The simulation accounts for manufacturing variability in {device_type} production using {manufacturing_process}.
            """)
            
            # Export options
            col_a, col_b = st.columns(2)
            
            with col_a:
                df = export_monte_carlo_as_csv(results)
                csv_download = generate_download_link(
                    df, 
                    f"monte_carlo_analysis_{device_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv", 
                    "📥 Export as CSV"
                )
                st.markdown(csv_download, unsafe_allow_html=True)
            
            # Add AI assistant for risk analysis
            st.markdown("### 🤖 Risk Analysis AI Assistant")
            display_ai_assistant(results, "monte_carlo")

def display_analysis_page():
    """Display the main analysis page with tabs."""
    display_header()
    
    tabs = st.tabs(["Quality ROI Analysis", "Process Control Analysis", "Root Cause Analysis", "Monte Carlo Simulation"])
    
    with tabs[0]:
        if not st.session_state.analysis_submitted:
            results = display_quality_analysis_form()
            if results:
                # Show AI assistant directly with the results for better visibility
                col1, col2 = st.columns([3, 2])
                with col1:
                    display_quality_issue_results(results)
                with col2:
                    display_ai_assistant(results)
        else:
            col1, col2 = st.columns([1, 6])
            with col1:
                if st.button("Start New Analysis", key="new_analysis_btn"):
                    st.session_state.analysis_submitted = False
                    st.session_state.quality_analysis_results = None
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Display a badge with the SKU being analyzed
            with col2:
                st.markdown(f"""
                <div style="background-color: {TERTIARY_COLOR}; padding: 0.5rem 1rem; border-radius: 20px; 
                      display: inline-block; margin-bottom: 1rem;">
                    <span style="font-weight: 600;">Analyzing SKU:</span> {st.session_state.quality_analysis_results['sku']} | 
                    <span style="font-weight: 600;">Type:</span> {st.session_state.quality_analysis_results['device_type']}
                </div>
                """, unsafe_allow_html=True)
            
            # Show results and AI assistant side by side for better integration
            col1, col2 = st.columns([3, 2])
            
            with col1:
                display_quality_issue_results(st.session_state.quality_analysis_results)
                
                # Add export options below the results
                st.markdown("### Export Options")
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    if st.session_state.quality_analysis_results:
                        results = st.session_state.quality_analysis_results
                        df = export_as_csv(results)
                        csv_download = generate_download_link(
                            df, 
                            f"quality_analysis_{results['sku']}_{datetime.now().strftime('%Y%m%d')}.csv", 
                            "📥 Export as CSV"
                        )
                        st.markdown(csv_download, unsafe_allow_html=True)
                
                with exp_col2:
                    if st.session_state.quality_analysis_results:
                        results = st.session_state.quality_analysis_results
                        try:
                            pdf_buffer = export_as_pdf(results)
                            pdf_data = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                            pdf_download = f'<a href="data:application/pdf;base64,{pdf_data}" download="quality_analysis_{results["sku"]}_{datetime.now().strftime("%Y%m%d")}.pdf" class="export-button">📄 Export as PDF</a>'
                            st.markdown(pdf_download, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")
            
            with col2:
                display_ai_assistant(st.session_state.quality_analysis_results)
    
    with tabs[1]:
        display_process_control_analysis()
    
    with tabs[2]:
        st.markdown("### 🧩 Root Cause Analysis")
        
        # Get device type info
        device_type = st.session_state.get('selected_device_type', 'Generic')
        device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
        
        st.markdown(f"""
        Analyze potential root causes of quality issues for {device_type} devices.
        
        Common issues for this device type include:
        - {', '.join(device_info['common_issues'])}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form(key="root_cause_form"):
                st.markdown("#### Issue Information")
                
                issue_description = st.text_area(
                    "Issue Description",
                    placeholder="Describe the quality issue in detail...",
                    help="Provide a detailed description of the problem"
                )
                
                root_cause = st.selectbox(
                    "Primary Root Cause Category",
                    options=ROOT_CAUSE_CATEGORIES,
                    index=0,
                    help="Select the primary root cause category"
                )
                
                # Generate failure modes based on root cause and device type
                if root_cause == "Design flaw":
                    failure_mode_options = [
                        f"Inadequate {device_type.lower()} design specifications",
                        f"Poor ergonomic design for {device_type.lower()}",
                        "Improper material selection",
                        "Insufficient stress analysis",
                        "Incompatible component interfaces",
                        "Other design issue"
                    ]
                elif root_cause == "Material failure":
                    failure_mode_options = [
                        "Material degradation",
                        "Inconsistent material properties",
                        "Contamination in raw materials",
                        "Wrong material grade used",
                        "Material fatigue",
                        "Other material issue"
                    ]
                elif root_cause == "Manufacturing defect":
                    failure_mode_options = [
                        "Process parameter deviation",
                        "Equipment malfunction",
                        "Operator error",
                        "Poor process control",
                        "Tooling wear or damage",
                        "Other manufacturing issue"
                    ]
                elif root_cause == "Assembly error":
                    failure_mode_options = [
                        "Missing components",
                        "Incorrect assembly sequence",
                        "Improper fastening",
                        "Misalignment during assembly",
                        "Wrong components used",
                        "Other assembly issue"
                    ]
                elif root_cause == "Component failure":
                    failure_mode_options = [
                        "Electronic component failure",
                        "Mechanical component wear",
                        "Fastener failure",
                        "Bearing/hinge failure",
                        "Sensor calibration drift",
                        "Other component issue"
                    ]
                elif root_cause == "Software bug":
                    failure_mode_options = [
                        "Logic error",
                        "Memory leak",
                        "User interface issue",
                        "Data processing error",
                        "Device communication problem",
                        "Other software issue"
                    ]
                else:
                    failure_mode_options = [
                        "Unknown failure mode",
                        "Multiple failure modes",
                        "Intermittent issue",
                        "Environmental factor",
                        "User-related issue",
                        "Other issue"
                    ]
                
                failure_mode = st.selectbox(
                    "Specific Failure Mode",
                    options=failure_mode_options,
                    index=0,
                    help="Select the specific failure mode"
                )
                
                st.markdown("#### Frequency Analysis")
                
                # Create sample data for demonstration
                failure_modes_input = st.text_area(
                    "Failure Modes and Counts",
                    value=f"{failure_mode_options[0]}: 35\n{failure_mode_options[1]}: 22\n{failure_mode_options[2]}: 18\n{failure_mode_options[3]}: 12\n{failure_mode_options[4]}: 8\n{failure_mode_options[5]}: 5",
                    help="Enter each failure mode and count on a new line in the format 'Mode: Count'"
                )
                
                submitted = st.form_submit_button("Analyze Root Causes")
                
                if submitted:
                    with st.spinner("Analyzing root causes..."):
                        try:
                            # Parse failure modes and counts
                            failure_modes = {}
                            for line in failure_modes_input.strip().split('\n'):
                                if ':' in line:
                                    mode, count = line.split(':', 1)
                                    try:
                                        failure_modes[mode.strip()] = int(count.strip())
                                    except ValueError:
                                        st.error(f"Invalid count for '{mode}': {count}")
                            
                            # Generate Pareto analysis
                            if failure_modes:
                                st.session_state.root_cause_analysis = {
                                    "issue_description": issue_description,
                                    "root_cause": root_cause,
                                    "failure_mode": failure_mode,
                                    "failure_modes": failure_modes,
                                    "pareto_data": generate_pareto_data(failure_modes)
                                }
                            else:
                                st.error("No valid failure modes found. Please check the format.")
                        except Exception as e:
                            st.error(f"Error analyzing root causes: {str(e)}")
                            logger.exception("Error in root cause analysis")
        
        with col2:
            if 'root_cause_analysis' in st.session_state and st.session_state.root_cause_analysis:
                data = st.session_state.root_cause_analysis
                
                st.markdown("#### Root Cause Analysis Results")
                
                # Display primary root cause and failure mode
                st.markdown(f"""
                **Primary Root Cause:** {data['root_cause']}  
                **Main Failure Mode:** {data['failure_mode']}
                """)
                
                # Create Pareto chart
                pareto_data = data['pareto_data']
                
                fig = go.Figure()
                
                # Add bar chart for failure modes
                fig.add_trace(go.Bar(
                    x=pareto_data['modes'],
                    y=pareto_data['counts'],
                    name="Count",
                    marker_color=PRIMARY_COLOR
                ))
                
                # Add line chart for cumulative percentage
                fig.add_trace(go.Scatter(
                    x=pareto_data['modes'],
                    y=pareto_data['cumulative_percent'],
                    name="Cumulative %",
                    marker=dict(color=DANGER_COLOR),
                    line=dict(width=3),
                    yaxis="y2"
                ))
                
                # Update layout
                fig.update_layout(
                    title="Pareto Analysis of Failure Modes",
                    xaxis_title="Failure Mode",
                    yaxis=dict(
                        title="Count",
                        titlefont=dict(color=PRIMARY_COLOR),
                        tickfont=dict(color=PRIMARY_COLOR)
                    ),
                    yaxis2=dict(
                        title="Cumulative %",
                        titlefont=dict(color=DANGER_COLOR),
                        tickfont=dict(color=DANGER_COLOR),
                        anchor="x",
                        overlaying="y",
                        side="right",
                        range=[0, 100]
                    ),
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Add 80% line for Pareto principle
                fig.add_shape(
                    type="line",
                    x0=-0.5, y0=80,
                    x1=len(pareto_data['modes'])-0.5, y1=80,
                    line=dict(color="red", width=2, dash="dash"),
                    yref="y2"
                )
                
                # Add annotation for 80% line
                fig.add_annotation(
                    x=len(pareto_data['modes'])-1,
                    y=80,
                    xref="x",
                    yref="y2",
                    text="80% of Issues",
                    showarrow=True,
                    arrowhead=1,
                    ax=50,
                    ay=-30
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add fishbone diagram explanation (simplified visualization)
                st.markdown("#### Ishikawa (Fishbone) Analysis")
                
                # Create simplified fishbone diagram
                categories = ["Materials", "Methods", "Machinery", "Measurement", "People", "Environment"]
                
                # Create a sunburst chart as a simplified fishbone visualization
                labels = ["Root Cause"]
                parents = [""]
                values = [100]
                
                # Add main categories
                for category in categories:
                    labels.append(category)
                    parents.append("Root Cause")
                    values.append(16)  # Equal distribution for main categories
                
                # Add sub-causes based on root cause and device type
                sub_causes = {
                    "Materials": ["Quality issues", "Specifications", "Handling", "Storage"],
                    "Methods": ["Process parameters", "Procedures", "Work instructions", "Testing"],
                    "Machinery": ["Maintenance", "Calibration", "Tooling", "Settings"],
                    "Measurement": ["Testing methods", "Gauges", "Inspection", "Standards"],
                    "People": ["Training", "Skills", "Experience", "Supervision"],
                    "Environment": ["Temperature", "Humidity", "Cleanliness", "Layout"]
                }
                
                for category, causes in sub_causes.items():
                    for cause in causes:
                        labels.append(cause)
                        parents.append(category)
                        values.append(4)  # Equal distribution for sub-causes
                
                fig = go.Figure(go.Sunburst(
                    labels=labels,
                    parents=parents,
                    values=values,
                    branchvalues="total",
                    marker=dict(
                        colors=[PRIMARY_COLOR] + [SECONDARY_COLOR] * len(categories) + [TERTIARY_COLOR] * sum(len(causes) for causes in sub_causes.values()),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{label}</b><br>Category: %{parent}<br>',
                    maxdepth=2
                ))
                
                fig.update_layout(
                    title="Root Cause Categories",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("#### Recommended Actions")
                
                # Generate recommendations based on primary root cause
                if data['root_cause'] == "Design flaw":
                    st.markdown("""
                    1. Conduct design review with cross-functional team
                    2. Perform FMEA on the affected components
                    3. Update design specifications and validation protocols
                    4. Verify design changes with appropriate testing
                    5. Update documentation and notify stakeholders
                    """)
                elif data['root_cause'] == "Material failure":
                    st.markdown("""
                    1. Review material specifications and requirements
                    2. Test alternative materials with improved properties
                    3. Audit material suppliers and handling processes
                    4. Implement incoming material testing protocols
                    5. Update material requirements and specifications
                    """)
                elif data['root_cause'] == "Manufacturing defect":
                    st.markdown("""
                    1. Analyze manufacturing process parameters and controls
                    2. Implement Statistical Process Control (SPC)
                    3. Review and update operator training procedures
                    4. Implement additional in-process quality checks
                    5. Consider process automation for critical steps
                    """)
                elif data['root_cause'] == "Assembly error":
                    st.markdown("""
                    1. Review assembly processes, fixtures, and tools
                    2. Improve work instructions with visual aids
                    3. Implement error-proofing (poka-yoke) mechanisms
                    4. Enhanced operator training for critical assembly steps
                    5. Add verification steps and inspection points
                    """)
                elif data['root_cause'] == "Component failure":
                    st.markdown("""
                    1. Evaluate component specifications and requirements
                    2. Audit component suppliers and validation processes
                    3. Implement incoming component testing/inspection
                    4. Consider alternative components with better reliability
                    5. Review environmental and usage conditions
                    """)
                elif data['root_cause'] == "Software bug":
                    st.markdown("""
                    1. Conduct thorough code reviews of affected modules
                    2. Implement additional unit and integration tests
                    3. Enhance error handling and logging capabilities
                    4. Improve version control and release processes
                    5. Consider automated testing frameworks
                    """)
                else:
                    st.markdown("""
                    1. Gather additional data to identify specific root causes
                    2. Form cross-functional investigation team
                    3. Implement immediate containment actions
                    4. Develop and verify corrective actions
                    5. Monitor effectiveness after implementation
                    """)
    
    with tabs[3]:
        run_monte_carlo_simulation_ui()

def display_standalone_assistant_page():
    """Display the standalone AI assistant page."""
    st.title("🤖 Medical Device Manufacturing & Quality AI Assistant")
    
    # Get device type info
    device_type = st.session_state.get('selected_device_type', 'Generic')
    device_info = DEVICE_TYPES.get(device_type, DEVICE_TYPES['Generic'])
    
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        Chat with our AI assistant to get expert advice on medical device manufacturing, 
        quality issues, and troubleshooting for {device_type} devices.
    </div>
    """, unsafe_allow_html=True)
    
    # Split the page into two columns
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Main chat interface
        display_ai_assistant()
    
    with col2:
        # Information panel for the selected device type
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
            <h4>{device_type} - Device Information</h4>
            <p>{device_info['description']}</p>
            
            <strong>Examples:</strong>
            <ul style="margin-left: 1rem; padding-left: 0.5rem;">
                {' '.join([f'<li>{example}</li>' for example in device_info['examples']])}
            </ul>
            
            <strong>Common Issues:</strong>
            <ul style="margin-left: 1rem; padding-left: 0.5rem;">
                {' '.join([f'<li>{issue}</li>' for issue in device_info['common_issues']])}
            </ul>
            
            <strong>Manufacturing Processes:</strong>
            <ul style="margin-left: 1rem; padding-left: 0.5rem;">
                {' '.join([f'<li>{process}</li>' for process in device_info['manufacturing_processes']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Manufacturing issues troubleshooting guide
        with st.expander("Manufacturing Troubleshooting Guide", expanded=False):
            if device_type == "Mobility Aids":
                st.markdown("""
                **Common Manufacturing Issues:**
                
                **Frame instability:**
                - Check welding quality and consistency
                - Verify torque specifications on fasteners
                - Inspect material thickness and quality
                
                **Wheel alignment issues:**
                - Check axle straightness and mounting
                - Verify bearing installation and quality
                - Inspect frame alignment and squareness
                
                **Brake failures:**
                - Verify spring tension and mechanism alignment
                - Check cable routing and attachment points
                - Inspect brake pad material and installation
                """)
            elif device_type == "Cushions & Support":
                st.markdown("""
                **Common Manufacturing Issues:**
                
                **Foam compression issues:**
                - Verify foam density and compression resistance
                - Check molding process parameters
                - Inspect foam curing conditions
                
                **Cover material tears:**
                - Check fabric cutting alignment and tension
                - Verify seam allowances and stitch density
                - Inspect material handling procedures
                
                **Pressure distribution problems:**
                - Verify foam layering and assembly
                - Check insert placement and alignment
                - Test pressure mapping for consistency
                """)
            elif device_type == "Monitoring Devices":
                st.markdown("""
                **Common Manufacturing Issues:**
                
                **Sensor calibration drift:**
                - Verify initial calibration procedures
                - Check component thermal stability
                - Inspect calibration reference standards
                
                **Display/electronics failures:**
                - Check PCB solder joint quality
                - Verify component placement accuracy
                - Inspect conformal coating application
                
                **Battery issues:**
                - Verify battery connection design
                - Check protection circuit function
                - Test charging and discharging cycles
                """)
            elif device_type == "Orthopedic Supports":
                st.markdown("""
                **Common Manufacturing Issues:**
                
                **Strap failures:**
                - Check stitching pattern and thread tension
                - Verify attachment point reinforcement
                - Inspect elastic material quality and stretch
                
                **Hinge malfunctions:**
                - Verify alignment during assembly
                - Check rivet/fastener installation
                - Inspect lubrication and friction surfaces
                
                **Material wear issues:**
                - Check edge finishing processes
                - Verify material thickness consistency
                - Inspect hook-and-loop fastener quality
                """)
            else:
                st.markdown("""
                **Common Manufacturing Issues:**
                
                **Injection molding defects:**
                - Check material drying and moisture content
                - Verify mold temperature and cooling
                - Inspect injection speed and pressure parameters
                
                **Assembly problems:**
                - Check fixture design and part positioning
                - Verify fastener installation procedures
                - Inspect component alignment guides
                
                **Dimensional variations:**
                - Check tool wear and replacement schedule
                - Verify machine calibration and maintenance
                - Inspect environmental conditions (temperature/humidity)
                """)
        
        # Quality control reference
        with st.expander("Quality Control Methods", expanded=False):
            st.markdown("""
            ### AQL Sampling Plans
            
            **AQL 0.65 (Critical):**
            - For critical characteristics affecting safety
            - High inspection rigor (typically used for medical devices)
            - Example: For lot size 1000, sample size 80 units
            
            **AQL 1.0 (Major):**
            - For major characteristics affecting function
            - Medium inspection rigor
            - Example: For lot size 1000, sample size 50 units
            
            **AQL 2.5 (Minor):**
            - For minor characteristics affecting appearance
            - Lower inspection rigor
            - Example: For lot size 1000, sample size 32 units
            
            ### Inspection Methods
            
            **Visual Inspection:**
            - Workmanship standards with visual aids
            - Magnification for small features
            - Lighting conditions specified
            
            **Dimensional Inspection:**
            - Gauges, calipers, CMM measurement
            - Go/no-go fixtures for critical dimensions
            - Specified measurement accuracy requirements
            
            **Functional Testing:**
            - Operation cycling tests
            - Performance parameter verification
            - Load testing to specified requirements
            """)
        
        # Example questions for the selected device type
        st.markdown("""
        <div style="background-color: #e9f7fe; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
            <h4>Example Questions</h4>
            <ul style="margin-left: 1rem; padding-left: 0.5rem;">
                <li>What are common manufacturing defects in {device_type}?</li>
                <li>How can we improve quality control for {device_info['examples'][0] if device_info['examples'] else 'medical devices'}?</li>
                <li>What process controls should we implement for {device_info['manufacturing_processes'][0] if device_info['manufacturing_processes'] else 'manufacturing'}?</li>
                <li>How to troubleshoot {device_info['common_issues'][0] if device_info['common_issues'] else 'quality issues'}?</li>
                <li>What testing methods are appropriate for {device_type}?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN APPLICATION ---
def main():
    """Main application function."""
    try:
        # Add FontAwesome to the page for icons
        st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        """, unsafe_allow_html=True)
        
        # Display sidebar navigation
        display_navigation()
        
        # Handle page navigation
        if st.session_state.current_page == "analysis":
            display_analysis_page()
        elif st.session_state.current_page == "assistant":
            display_standalone_assistant_page()
        
        # Add footer with version and additional info
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 4px; margin-top: 2rem; 
                    text-align: center; border-top: 1px solid #dee2e6;">
            <div style="color: #6c757d; font-size: 0.8rem;">
                Medical Device Quality Analysis Tool v1.1.0 | © 2025 Medical Device Quality Management
            </div>
            <div style="color: #6c757d; font-size: 0.8rem; margin-top: 0.5rem;">
                For support contact: <a href="mailto:alexander.popoff@vivehealth.com">alexander.popoff@vivehealth.com</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Unexpected application error")
        
        # Show detailed error message and recovery options
        with st.expander("Error Details", expanded=True):
            st.code(traceback.format_exc())
            
            st.markdown("""
            ### Troubleshooting Options
            
            1. **Refresh the page** to restart the application
            2. **Clear browser cache** and try again
            3. If the problem persists, please contact support with the error details above
            """)
            
            if st.button("Reset Application State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
