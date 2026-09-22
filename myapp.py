import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Default Prediction & Risk Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- MINIMAL CLEAN STYLING (THEME & HIGH CONTRAST CSS) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --bg-main: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #475569;
    --border-color: #e2e8f0;
}

/* Base resets & typography */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

/* Ensure ALL text elements in main content have sharp dark contrast */
.stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6, label {
    color: #0f172a !important;
}

/* Sidebar styling - Dark Slate Minimal */
section[data-testid="stSidebar"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155;
    padding-top: 20px;
}

/* Sidebar markdown & header text contrast */
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .sidebar-title {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

.sidebar-brand {
    padding: 10px 5px 18px 5px;
    border-bottom: 1px solid #334155;
    margin-bottom: 20px;
}

.sidebar-logo-text {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.3px;
}

.sidebar-tagline {
    font-size: 12px;
    color: #94a3b8 !important;
    margin-top: 3px;
}

/* Widget input labels readability */
label[data-testid="stWidgetLabel"] {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 4px !important;
}

/* Streamlit Input Fields Contrast Fix */
div[data-baseweb="input"] > div, 
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}

div[data-baseweb="input"] input {
    color: #0f172a !important;
    font-weight: 500 !important;
}

/* Custom Navigation Radio Buttons */
div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] > div {
    gap: 6px;
}

div[data-testid="stRadio"] > div > label {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 10px 14px;
    color: #e2e8f0 !important;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.15s ease;
    cursor: pointer;
    width: 100%;
}

div[data-testid="stRadio"] > div > label:hover {
    background: rgba(255, 255, 255, 0.12);
    color: #ffffff !important;
}

div[data-testid="stRadio"] > div > label[data-checked="true"] {
    background: #4f46e5 !important;
    color: #ffffff !important;
    font-weight: 700;
    border-color: #6366f1;
}

/* Minimal Header Header Banner */
.minimal-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 24px 30px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.minimal-title {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a !important;
    margin: 0 0 4px 0;
    letter-spacing: -0.4px;
}

.minimal-subtitle {
    font-size: 14px;
    color: #475569 !important;
    margin: 0;
}

/* Card Styling - Simple & Minimal */
.clean-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.card-title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a !important;
    margin: 0 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #f1f5f9;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* KPI Cards */
.kpi-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.kpi-title {
    font-size: 12px;
    font-weight: 600;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a !important;
    margin-top: 4px;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 12px;
    color: #475569 !important;
    margin-top: 6px;
}

/* Minimal Prediction Box */
.pred-container-low {
    background: #f0fdf4;
    border: 1.5px solid #bbf7d0;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.pred-container-med {
    background: #fffbeb;
    border: 1.5px solid #fde68a;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.pred-container-high {
    background: #fef2f2;
    border: 1.5px solid #fecaca;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.pred-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.badge-low { background: #dcfce7; color: #166534 !important; }
.badge-med { background: #fef3c7; color: #92400e !important; }
.badge-high { background: #fee2e2; color: #991b1b !important; }

.pred-score {
    font-size: 36px;
    font-weight: 800;
    margin: 4px 0 8px 0;
    color: #0f172a !important;
}

/* Primary Action Button */
.stButton > button {
    width: 100%;
    background-color: #4f46e5;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 15px;
    padding: 12px 20px;
    border-radius: 8px;
    border: none;
    transition: background 0.15s ease;
}

.stButton > button:hover {
    background-color: #4338ca;
}

/* Factor Pills */
.factor-pill-positive {
    background: #f0fdf4;
    color: #166534 !important;
    border: 1px solid #cbd5e1;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
}

.factor-pill-negative {
    background: #fef2f2;
    color: #991b1b !important;
    border: 1px solid #cbd5e1;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
}

/* Hide streamlit footer & hamburger menu */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Keep Streamlit top header bar accessible for sidebar toggle */
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 99999 !important;
}

/* Sidebar Toggle Arrow Control Button - Prominent Indigo Button */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[aria-label="Open sidebar"], 
button[aria-label="Close sidebar"],
button[aria-label="Expand sidebar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    border: none !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
button[aria-label="Open sidebar"] svg,
button[aria-label="Close sidebar"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
    stroke: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    file_name = "Loan_default.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "LoanID": [f"LN-{1000+i}" for i in range(n)],
            "Age": np.random.randint(21, 68, n),
            "Income": np.random.randint(25000, 150000, n),
            "LoanAmount": np.random.randint(5000, 120000, n),
            "CreditScore": np.random.randint(400, 850, n),
            "MonthsEmployed": np.random.randint(6, 120, n),
            "NumCreditLines": np.random.randint(1, 6, n),
            "InterestRate": np.round(np.random.uniform(4.5, 24.0, n), 2),
            "LoanTerm": np.random.choice([12, 24, 36, 48, 60], n),
            "DTIRatio": np.round(np.random.uniform(0.1, 0.65, n), 2),
            "Education": np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n),
            "EmploymentType": np.random.choice(["Full-time", "Part-time", "Self-employed", "Unemployed"], n),
            "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], n),
            "HasMortgage": np.random.choice(["Yes", "No"], n),
            "HasDependents": np.random.choice(["Yes", "No"], n),
            "LoanPurpose": np.random.choice(["Auto", "Business", "Education", "Home", "Other"], n),
            "HasCoSigner": np.random.choice(["Yes", "No"], n),
            "Default": np.random.choice([0, 1], n, p=[0.85, 0.15])
        })
    return df

df_raw = load_data()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo-text">💳 FinPulse</div>
            <div class="sidebar-tagline">Loan Analytics & Risk Portal</div>
            <div style="margin-top:6px;">
                <span style="background:rgba(99,102,241,0.2); color:#818cf8 !important; padding:3px 8px; border-radius:12px; font-size:10px; font-weight:700;">v2.4 Enterprise</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:11px; font-weight:700; color:#94a3b8 !important; margin-top:10px; margin-bottom:4px; text-transform:uppercase;'>DATA SOURCE</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Custom CSV Dataset",
        type=["csv"],
        help="Upload CSV file containing loan application data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Custom CSV dataset loaded!")
            dataset_status = "Custom CSV File"
        except Exception:
            st.error("Error reading file. Using default dataset.")
            df = df_raw
            dataset_status = "Default Dataset"
    else:
        df = df_raw
        dataset_status = "Loan_default.csv" if os.path.exists("Loan_default.csv") else "Demo Dataset"

    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; font-size: 11px; color: #cbd5e1 !important; margin-bottom: 15px;">
            Active Source: <strong style="color:#ffffff !important;">{dataset_status}</strong>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:11px; font-weight:700; color:#94a3b8 !important; margin-bottom:6px; text-transform:uppercase;'>MAIN MENU</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📁 Dataset Explorer",
            "📊 Risk Analytics",
            "🤖 Smart Prediction",
            "💡 What-If Simulator",
            "🎯 Model Evaluation (Task 5)"
        ]
    )

    st.markdown("<hr style='border-color:#334155; margin:16px 0;'>", unsafe_allow_html=True)

    # Sidebar Dataset Quick Stats Widget
    default_pct = (df['Default'].mean() * 100) if 'Default' in df.columns else 0
    avg_credit_sb = df['CreditScore'].mean() if 'CreditScore' in df.columns else 0
    avg_loan_sb = df['LoanAmount'].mean() if 'LoanAmount' in df.columns else 0

    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom:14px;">
            <div style="font-size: 11px; font-weight:700; color: #94a3b8 !important; text-transform: uppercase;">PORTFOLIO QUICK STATS</div>
            <div style="font-size: 18px; font-weight: 800; color: #ffffff !important; margin-top: 4px;">{len(df):,} <span style="font-size: 11px; color: #94a3b8 !important;">Applicants</span></div>
            <div style="font-size: 12px; color: #cbd5e1 !important; margin-top: 6px; display:flex; justify-content:space-between;">
                <span>Default Rate:</span>
                <strong style="color: {'#f87171' if default_pct > 15 else '#4ade80'} !important;">{default_pct:.1f}%</strong>
            </div>
            <div style="font-size: 12px; color: #cbd5e1 !important; margin-top: 4px; display:flex; justify-content:space-between;">
                <span>Avg FICO:</span>
                <strong style="color: #ffffff !important;">{avg_credit_sb:.0f}</strong>
            </div>
            <div style="font-size: 12px; color: #cbd5e1 !important; margin-top: 4px; display:flex; justify-content:space-between;">
                <span>Avg Loan:</span>
                <strong style="color: #ffffff !important;">${avg_loan_sb:,.0f}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Risk Tier Guide
    st.markdown("""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid #334155; border-radius: 8px; padding: 10px;">
            <div style="font-size: 10px; font-weight:700; color: #94a3b8 !important; text-transform: uppercase; margin-bottom:6px;">RISK TIER THRESHOLDS</div>
            <div style="font-size:11px; color:#4ade80 !important; margin-bottom:2px;">🟢 <strong>Low Risk:</strong> &lt; 30% Probability</div>
            <div style="font-size:11px; color:#fbbf24 !important; margin-bottom:2px;">🟡 <strong>Moderate Risk:</strong> 30% - 55%</div>
            <div style="font-size:11px; color:#f87171 !important;">🔴 <strong>High Risk:</strong> &gt; 55% Probability</div>
        </div>
    """, unsafe_allow_html=True)


# Financial EMI Helper
def calculate_emi(principal, annual_rate, term_months):
    if annual_rate == 0 or term_months == 0:
        return principal / max(1, term_months)
    monthly_rate = annual_rate / 100 / 12
    try:
        emi = (principal * monthly_rate * math.pow(1 + monthly_rate, term_months)) / (math.pow(1 + monthly_rate, term_months) - 1)
        return emi
    except Exception:
        return principal / max(1, term_months)


# Load Pretrained ML Model Pipeline
@st.cache_resource
def load_trained_model():
    model_path = "loan_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


# =====================================================
# 1. DASHBOARD
# =====================================================
if page == "🏠 Dashboard":
    st.markdown("""
        <div class="minimal-header">
            <div class="minimal-title">Portfolio Overview</div>
            <div class="minimal-subtitle">Loan portfolio performance metrics, default rates, and dataset insights.</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    total_records = len(df)
    avg_income = df['Income'].mean() if 'Income' in df.columns else 0
    avg_loan = df['LoanAmount'].mean() if 'LoanAmount' in df.columns else 0
    avg_credit = df['CreditScore'].mean() if 'CreditScore' in df.columns else 0
    default_rate = (df['Default'].mean() * 100) if 'Default' in df.columns else 0

    with col1:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Total Applications</div>
                <div class="kpi-value">{total_records:,}</div>
                <div class="kpi-subtext">Active portfolio count</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Avg Loan Amount</div>
                <div class="kpi-value">${avg_loan:,.0f}</div>
                <div class="kpi-subtext">Avg Income: ${avg_income:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Avg Credit Score</div>
                <div class="kpi-value">{avg_credit:.0f}</div>
                <div class="kpi-subtext">Target: 650+ FICO</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Default Rate</div>
                <div class="kpi-value" style="color: {'#dc2626' if default_rate > 15 else '#16a34a'};">{default_rate:.1f}%</div>
                <div class="kpi-subtext">Overall portfolio risk</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("""
            <div class="clean-card">
                <div class="card-title">Credit Score Tier Distribution</div>
        """, unsafe_allow_html=True)
        
        if 'CreditScore' in df.columns:
            bins = [300, 580, 670, 740, 850]
            labels = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Excellent (740+)']
            df['CreditTier'] = pd.cut(df['CreditScore'], bins=bins, labels=labels)
            tier_counts = df['CreditTier'].value_counts().reindex(labels).fillna(0)
            st.bar_chart(tier_counts, color="#4f46e5")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("""
            <div class="clean-card">
                <div class="card-title">Default Rate by Loan Purpose</div>
        """, unsafe_allow_html=True)
        
        if 'LoanPurpose' in df.columns and 'Default' in df.columns:
            purpose_default = df.groupby('LoanPurpose')['Default'].mean() * 100
            st.bar_chart(purpose_default, color="#f59e0b")
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 2. DATASET EXPLORER
# =====================================================
elif page == "📁 Dataset Explorer":
    st.markdown("""
        <div class="minimal-header">
            <div class="minimal-title">Dataset Explorer</div>
            <div class="minimal-subtitle">Inspect raw records, slice data with filters, and verify data hygiene.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔍 Filter Options</div>", unsafe_allow_html=True)
    
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        purposes = ["All"] + list(df['LoanPurpose'].dropna().unique()) if 'LoanPurpose' in df.columns else ["All"]
        selected_purpose = st.selectbox("Loan Purpose", purposes)
            
    with f_col2:
        emp_types = ["All"] + list(df['EmploymentType'].dropna().unique()) if 'EmploymentType' in df.columns else ["All"]
        selected_emp = st.selectbox("Employment Type", emp_types)
            
    with f_col3:
        if 'CreditScore' in df.columns:
            min_c, max_c = int(df['CreditScore'].min()), int(df['CreditScore'].max())
            credit_range = st.slider("Credit Score Range", min_c, max_c, (min_c, max_c))
        else:
            credit_range = (300, 850)

    st.markdown("</div>", unsafe_allow_html=True)

    filtered_df = df.copy()
    if selected_purpose != "All" and 'LoanPurpose' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['LoanPurpose'] == selected_purpose]
    if selected_emp != "All" and 'EmploymentType' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['EmploymentType'] == selected_emp]
    if 'CreditScore' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['CreditScore'] >= credit_range[0]) & (filtered_df['CreditScore'] <= credit_range[1])]

    st.markdown(f"**Showing {len(filtered_df):,} records out of {len(df):,} total**")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    tab_stats, tab_missing = st.tabs(["📊 Statistical Summary", "⚠️ Missing Values"])
    with tab_stats:
        num_df = filtered_df.select_dtypes(include="number")
        if not num_df.empty:
            st.dataframe(num_df.describe().T.style.format("{:.2f}"), use_container_width=True)
    with tab_missing:
        missing_count = filtered_df.isnull().sum()
        missing_df = pd.DataFrame({"Feature": missing_count.index, "Missing Values": missing_count.values})
        st.dataframe(missing_df, use_container_width=True)


# =====================================================
# 3. RISK ANALYTICS
# =====================================================
elif page == "📊 Risk Analytics":
    st.markdown("""
        <div class="minimal-header">
            <div class="minimal-title">Risk Analytics & Feature Correlations</div>
            <div class="minimal-subtitle">Analyze relationships between features and calculate correlation matrices.</div>
        </div>
    """, unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) >= 2:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📌 Feature Relationship Chart</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("X-Axis Feature", num_cols, index=min(1, len(num_cols)-1))
        with c2:
            y_var = st.selectbox("Y-Axis Feature", num_cols, index=min(2, len(num_cols)-1))
            
        st.scatter_chart(df.sample(min(1000, len(df))), x=x_var, y=y_var, color="Default" if "Default" in df.columns else None)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔥 Correlation Matrix Table</div>", unsafe_allow_html=True)
        corr = df[num_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap="Blues").format("{:.2f}"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 4. SMART PREDICTION ENGINE (CONNECTED TO MODEL.PKL)
# =====================================================
elif page == "🤖 Smart Prediction":
    st.markdown("""
        <div class="minimal-header">
            <div class="minimal-title">Smart Loan Risk Predictor</div>
            <div class="minimal-subtitle">Fill out applicant details across all dataset fields to generate an accurate default risk assessment powered by trained Machine Learning models.</div>
        </div>
    """, unsafe_allow_html=True)

    # 3-Column Clean Form Containing ALL 16 Dataset Fields
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("""
            <div class="clean-card">
                <div class="card-title">👤 Personal Profile</div>
        """, unsafe_allow_html=True)
        
        age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1, help="Applicant age in years")
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], index=1)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
        has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"], index=0)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col_f2:
        st.markdown("""
            <div class="clean-card">
                <div class="card-title">💼 Employment & Credit</div>
        """, unsafe_allow_html=True)
        
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=65000, step=5000)
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"], index=0)
        months_employed = st.number_input("Months Employed", min_value=0, max_value=480, value=36, step=6)
        credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=680)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=1, max_value=20, value=3, step=1)
        dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.01, max_value=0.99, value=0.35, step=0.05, help="Existing monthly debt over monthly gross income")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_f3:
        st.markdown("""
            <div class="clean-card">
                <div class="card-title">💳 Loan Request</div>
        """, unsafe_allow_html=True)
        
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=250000, value=25000, step=2500)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=35.0, value=10.5, step=0.5)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
        loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"], index=0)
        has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"], index=0)
        has_cosigner = st.selectbox("Has Co-Signer?", ["No", "Yes"], index=0)

        st.markdown("</div>", unsafe_allow_html=True)

    # Financial EMI & Payment Ratio Preview
    emi = calculate_emi(loan_amount, interest_rate, loan_term)
    monthly_income = income / 12 if income > 0 else 1
    payment_dti = (emi / monthly_income)

    st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px 20px; margin-bottom:20px; display:flex; justify-content:space-around; text-align:center;">
            <div>
                <span style="font-size:12px; color:#64748b; font-weight:600;">ESTIMATED MONTHLY PAYMENT (EMI)</span>
                <div style="font-size:20px; font-weight:800; color:#4f46e5;">${emi:,.2f}</div>
            </div>
            <div>
                <span style="font-size:12px; color:#64748b; font-weight:600;">NEW PAYMENT-TO-INCOME RATIO</span>
                <div style="font-size:20px; font-weight:800; color:{'#dc2626' if payment_dti > 0.4 else '#16a34a'};">{payment_dti*100:.1f}%</div>
            </div>
            <div>
                <span style="font-size:12px; color:#64748b; font-weight:600;">LOAN-TO-INCOME RATIO</span>
                <div style="font-size:20px; font-weight:800; color:#0f172a;">{(loan_amount/income):.2f}x</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🚀 Calculate Risk Assessment")

    if predict_btn:
        # Construct DataFrame for the applicant matching trained model schema
        applicant_df = pd.DataFrame([{
            'Age': int(age),
            'Income': int(income),
            'LoanAmount': int(loan_amount),
            'CreditScore': int(credit_score),
            'MonthsEmployed': int(months_employed),
            'NumCreditLines': int(num_credit_lines),
            'InterestRate': float(interest_rate),
            'LoanTerm': int(loan_term),
            'DTIRatio': float(dti_ratio),
            'Education': str(education),
            'EmploymentType': str(employment_type),
            'MaritalStatus': str(marital_status),
            'HasMortgage': str(has_mortgage),
            'HasDependents': str(has_dependents),
            'LoanPurpose': str(loan_purpose),
            'HasCoSigner': str(has_cosigner)
        }])

        model = load_trained_model()
        if model is not None:
            # Predict default probability using the saved Gradient Boosting model pipeline
            prob_default = float(model.predict_proba(applicant_df)[0, 1])
            risk_score = prob_default * 100
            model_info = "⚡ Powered by Trained ML Model (loan_model.pkl — Gradient Boosting)"
        else:
            # Fallback heuristic calculation if model file not found
            risk_points = 0.0
            if credit_score < 580: risk_points += 35
            elif credit_score < 670: risk_points += 20
            elif credit_score < 740: risk_points += 8

            if dti_ratio > 0.45: risk_points += 20
            elif dti_ratio > 0.30: risk_points += 10
            if payment_dti > 0.40: risk_points += 15

            if employment_type == "Unemployed": risk_points += 30
            elif employment_type == "Part-time": risk_points += 12
            if months_employed < 12: risk_points += 10
            elif months_employed > 36: risk_points -= 8

            if has_cosigner == "Yes": risk_points -= 15
            if has_mortgage == "Yes": risk_points += 5
            if has_dependents == "Yes": risk_points += 4

            if interest_rate > 15.0: risk_points += 10
            if loan_amount > income * 1.5: risk_points += 12
            if education in ["Master's", "PhD"]: risk_points -= 5

            risk_score = max(5.0, min(95.0, risk_points))
            model_info = "⚙️ Powered by Heuristic Scoring Engine (loan_model.pkl not loaded)"

        # Output Badge & Classification
        if risk_score < 30:
            c_box = "pred-container-low"
            c_badge = "badge-low"
            c_text = "LOW RISK (APPROVED)"
            desc = "Applicant demonstrates strong financial health and high probability of repayment."
        elif risk_score < 55:
            c_box = "pred-container-med"
            c_badge = "badge-med"
            c_text = "MODERATE RISK (CONDITIONAL)"
            desc = "Applicant requires additional documentation or a co-signer to mitigate risk."
        else:
            c_box = "pred-container-high"
            c_badge = "badge-high"
            c_text = "HIGH RISK (DEFAULT WARNING)"
            desc = "Applicant exceeds standard risk thresholds. Elevated default likelihood."

        st.markdown(f"""
            <div class="{c_box}">
                <span class="pred-badge {c_badge}">{c_text}</span>
                <div class="pred-score">{risk_score:.1f}% Default Probability</div>
                <div style="font-size:14px; color:#475569; margin-bottom:8px;">{desc}</div>
                <div style="font-size:12px; font-weight:600; color:#4f46e5;">{model_info}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Drivers Breakdown
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("""
                <div class="clean-card">
                    <div class="card-title" style="color:#16a34a !important;">✅ Favorable Factors</div>
            """, unsafe_allow_html=True)
            if credit_score >= 670:
                st.markdown("<div class='factor-pill-positive'>✓ Prime Credit Score (FICO 670+)</div>", unsafe_allow_html=True)
            if has_cosigner == "Yes":
                st.markdown("<div class='factor-pill-positive'>✓ Co-Signer Security Guarantee Present</div>", unsafe_allow_html=True)
            if dti_ratio <= 0.35:
                st.markdown("<div class='factor-pill-positive'>✓ Healthy Debt-to-Income Ratio (<35%)</div>", unsafe_allow_html=True)
            if months_employed >= 36:
                st.markdown("<div class='factor-pill-positive'>✓ Stable Employment History (3+ Years)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with r_col2:
            st.markdown("""
                <div class="clean-card">
                    <div class="card-title" style="color:#dc2626 !important;">⚠️ Risk Factors</div>
            """, unsafe_allow_html=True)
            if credit_score < 620:
                st.markdown("<div class='factor-pill-negative'>⚠ Below Average Credit Score (<620)</div>", unsafe_allow_html=True)
            if dti_ratio > 0.40:
                st.markdown("<div class='factor-pill-negative'>⚠ High Debt-To-Income Ratio (>40%)</div>", unsafe_allow_html=True)
            if interest_rate > 15.0:
                st.markdown("<div class='factor-pill-negative'>⚠ High Interest Rate (>15%)</div>", unsafe_allow_html=True)
            if employment_type == "Unemployed":
                st.markdown("<div class='factor-pill-negative'>⚠ Unemployed Employment Status</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 5. WHAT-IF SIMULATOR
# =====================================================
elif page == "💡 What-If Simulator":
    st.markdown("""
        <div class="minimal-header">
            <div class="minimal-title">What-If Risk Simulator</div>
            <div class="minimal-subtitle">Adjust simulation levers to see how improving credit score or adding a co-signer reduces default risk.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>⚙️ Simulation Levers</div>", unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        sim_credit = st.slider("Simulated Credit Score", 300, 850, 680)
        sim_loan = st.slider("Simulated Loan Amount ($)", 5000, 150000, 25000, step=5000)
    with sim_col2:
        sim_interest = st.slider("Simulated Interest Rate (%)", 3.0, 25.0, 10.5, step=0.5)
        sim_cosigner = st.radio("Include Co-Signer?", ["No", "Yes"], horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

    model = load_trained_model()
    if model is not None:
        sim_df = pd.DataFrame([{
            'Age': 35,
            'Income': 65000,
            'LoanAmount': int(sim_loan),
            'CreditScore': int(sim_credit),
            'MonthsEmployed': 36,
            'NumCreditLines': 3,
            'InterestRate': float(sim_interest),
            'LoanTerm': 36,
            'DTIRatio': 0.35,
            'Education': "Bachelor's",
            'EmploymentType': 'Full-time',
            'MaritalStatus': 'Married',
            'HasMortgage': 'No',
            'HasDependents': 'No',
            'LoanPurpose': 'Auto',
            'HasCoSigner': str(sim_cosigner)
        }])
        sim_prob = float(model.predict_proba(sim_df)[0, 1])
        sim_risk_score = sim_prob * 100
        sim_source = "⚡ Powered by Trained ML Model (loan_model.pkl)"
    else:
        base_risk = 45.0 + (650 - sim_credit) * 0.10 + (sim_interest - 10) * 1.2 - (15 if sim_cosigner == "Yes" else 0)
        sim_risk_score = max(5.0, min(95.0, base_risk))
        sim_source = "⚙️ Simulation Formula"

    st.markdown(f"""
        <div class="clean-card" style="text-align:center;">
            <div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase;">Simulated Risk Result</div>
            <div style="font-size:42px; font-weight:800; color:{'#16a34a' if sim_risk_score < 35 else '#dc2626'}; margin:6px 0;">{sim_risk_score:.1f}%</div>
            <div style="font-size:14px; font-weight:600; color:#0f172a;">Risk Level: {'🟢 LOW RISK' if sim_risk_score < 35 else ('🟡 MODERATE RISK' if sim_risk_score < 60 else '🔴 HIGH RISK')}</div>
            <div style="font-size:12px; color:#4f46e5; margin-top:6px; font-weight:600;">{sim_source}</div>
        </div>
    """, unsafe_allow_html=True)


# =====================================================
# 6. MODEL EVALUATION (TASK 5)
# =====================================================
elif page == "🎯 Model Evaluation (Task 5)":
    st.markdown("""
        <div class="minimal-header">
            <div class="minimal-title">Task 5: Model Evaluation & Tuning</div>
            <div class="minimal-subtitle">Comprehensive model performance analysis, cross-validation, and hyperparameter tuning.</div>
        </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

    @st.cache_data(show_spinner=False)
    def run_model_pipeline(data):
        # Preprocessing
        df_ml = data.copy()
        
        # DOWN-SAMPLE TO PREVENT FREEZE/BLUR
        # Limit to 1500 records max so GridSearch completes fast without crashing Streamlit
        if len(df_ml) > 1500:
            df_ml = df_ml.sample(n=1500, random_state=42)
        
        # Drop identifiers if present
        if 'LoanID' in df_ml.columns:
            df_ml = df_ml.drop('LoanID', axis=1)
            
        # Encode categorical variables
        le = LabelEncoder()
        cat_cols = df_ml.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            
        # Separate features and target
        if 'Default' not in df_ml.columns:
            return None, None, None
            
        X = df_ml.drop('Default', axis=1)
        y = df_ml['Default']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        
        results = []
        best_model_name = ""
        best_cv_score = 0
        best_model = None
        
        for name, model in models.items():
            # Train Model
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, zero_division=0)
            recall = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            # Cross Validation (5-Fold)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Check Fit
            fit_status = "Good Fit ✅"
            if train_acc - test_acc > 0.05:
                fit_status = "Overfitting ⚠️"
            elif train_acc < 0.6 and test_acc < 0.6:
                fit_status = "Underfitting ⚠️"
                
            results.append({
                "Model": name,
                "Train Acc": train_acc,
                "Test Acc (Accuracy)": test_acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "5-Fold CV Avg": cv_mean,
                "CV Spread (Std)": cv_std,
                "Fit Status": fit_status
            })
            
            if cv_mean > best_cv_score:
                best_cv_score = cv_mean
                best_model_name = name
                best_model = model
                
        # Perform GridSearchCV on best model
        grid_search_result = {}
        if best_model_name == "Random Forest":
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            tuned_acc = accuracy_score(y_test, grid.predict(X_test))
            grid_search_result = {"Best Params": str(grid.best_params_), "Tuned Test Acc": tuned_acc}
        elif best_model_name == "Gradient Boosting":
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            tuned_acc = accuracy_score(y_test, grid.predict(X_test))
            grid_search_result = {"Best Params": str(grid.best_params_), "Tuned Test Acc": tuned_acc}
        elif best_model_name == "AdaBoost":
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            grid = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            tuned_acc = accuracy_score(y_test, grid.predict(X_test))
            grid_search_result = {"Best Params": str(grid.best_params_), "Tuned Test Acc": tuned_acc}
        elif best_model_name == "Decision Tree":
            param_grid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
            grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            tuned_acc = accuracy_score(y_test, grid.predict(X_test))
            grid_search_result = {"Best Params": str(grid.best_params_), "Tuned Test Acc": tuned_acc}
        elif best_model_name == "Logistic Regression":
            param_grid = {'C': [0.1, 1.0, 10.0]}
            grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            tuned_acc = accuracy_score(y_test, grid.predict(X_test))
            grid_search_result = {"Best Params": str(grid.best_params_), "Tuned Test Acc": tuned_acc}
            
        return pd.DataFrame(results), best_model_name, grid_search_result

    with st.spinner("Running ML Pipeline (Evaluation, CV, Tuning)... This might take a minute."):
        results_df, best_model_name, grid_search_result = run_model_pipeline(df)
        
    if results_df is None:
        st.error("Dataset missing 'Default' target variable for classification.")
    else:
        st.markdown("<div class='clean-card'><div class='card-title'>1, 2, 3 & 4. Model Evaluation & Comparison</div>", unsafe_allow_html=True)
        st.markdown("Includes Base vs Test scores (Overfitting check), Accuracy, Precision, Recall, F1-Score, and 5-Fold Cross Validation.")
        
        # Display formatted table
        display_df = results_df.copy()
        for col in ["Train Acc", "Test Acc (Accuracy)", "Precision", "Recall", "F1-Score", "5-Fold CV Avg", "CV Spread (Std)"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
        st.dataframe(display_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown(f"<div class='clean-card'><div class='card-title'>🏆 Best Model Selected</div>", unsafe_allow_html=True)
            st.markdown(f"**{best_model_name}** selected based on highest Cross-Validation accuracy and stability.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='clean-card'><div class='card-title'>6. Advanced Models Implemented</div>", unsafe_allow_html=True)
            st.markdown("✅ Random Forest (Bagging)<br>✅ AdaBoost (Boosting)<br>✅ Gradient Boosting (Advanced Boosting)", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown("<div class='clean-card'><div class='card-title'>5. Hyperparameter Tuning</div>", unsafe_allow_html=True)
            st.markdown(f"Applied GridSearchCV to **{best_model_name}**:")
            if grid_search_result:
                st.success(f"Best Parameters Found: {grid_search_result.get('Best Params')}")
                st.info(f"Tuned Model Test Accuracy: {grid_search_result.get('Tuned Test Acc'):.4f}")
                
                baseline = results_df[results_df['Model'] == best_model_name]['Test Acc (Accuracy)'].values[0]
                st.markdown(f"Baseline Test Accuracy was: **{baseline:.4f}**")
            else:
                st.warning("Tuning failed or no parameters tested.")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Model Explanations for User ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class='clean-card'>
                <div class='card-title'>🧠 Understanding the Models: Which to choose and why?</div>
                <div style='font-size:14px; color:#475569; line-height:1.6;'>
                    <strong>1. Logistic Regression:</strong> A basic statistical baseline model. It calculates the probability of default using a linear mathematical equation. It is very fast and easy to interpret, but struggles if the relationships in the data are complex or non-linear.<br><br>
                    <strong>2. Decision Tree:</strong> A flowchart-like model that splits data based on questions (e.g., "Is income < $50k?"). It's highly interpretable but notoriously prone to <em>overfitting</em>—meaning it tends to memorize the training data rather than learning general rules.<br><br>
                    <strong>3. Random Forest (Bagging):</strong> An ensemble method that builds hundreds of different Decision Trees and averages their predictions (majority vote). <strong>Why choose it:</strong> It automatically fixes the overfitting problem of single Decision Trees. It is extremely robust, stable, and usually performs exceptionally well right out of the box.<br><br>
                    <strong>4. AdaBoost (Boosting):</strong> Instead of building trees independently, AdaBoost builds them sequentially. Each new tree focuses specifically on correcting the mistakes (misclassifications) made by the previous tree. <strong>Why choose it:</strong> It is excellent at boosting accuracy on borderline or difficult-to-predict applicants.<br><br>
                    <strong>5. Gradient Boosting (Advanced Boosting):</strong> Similar to AdaBoost, but uses an advanced mathematical technique (gradient descent) to minimize prediction errors. <strong>Why choose it:</strong> It frequently yields the absolute highest accuracy in machine learning competitions, though it requires careful hyperparameter tuning to avoid overfitting.<br>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='clean-card'>
                <div class='card-title'>💡 Why did we select <strong>{best_model_name}</strong> as the winner?</div>
                <div style='font-size:14px; color:#475569; line-height:1.6;'>
                    The system automatically selected <strong>{best_model_name}</strong> because it achieved the highest <strong>5-Fold Cross-Validation Average Score</strong>. 
                    <br><br>
                    <strong>Why does Cross-Validation matter?</strong> Instead of just testing the model once, Cross-Validation cuts the data into 5 equal pieces. It trains the model 5 separate times on different chunks and averages the score. This proves that the model didn't just get "lucky" on one specific test set. 
                    <br><br>
                    A high Cross-Validation score, combined with a low CV Spread (Standard Deviation), guarantees that the model is both highly accurate and highly stable, making it the safest choice for deploying into a real-world Loan Prediction Engine.
                </div>
            </div>
        """, unsafe_allow_html=True)