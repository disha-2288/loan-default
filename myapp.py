import json
import math
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import ui
from features import CAT_COLS, NUM_COLS, TARGET, add_engineered_features, clean_column_names, preprocess
from ui import AMBER, AQUA, FAINT, INK, IRIS, MINT, MUTED, ROSE

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "Loan_default.csv"
MODEL_PATH = BASE_DIR / "loan_model.pkl"
META_PATH = BASE_DIR / "model_meta.json"

REQUIRED_COLS = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
                 "InterestRate", "LoanTerm", "DTIRatio", "Education", "EmploymentType", "MaritalStatus",
                 "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner", "Default"]
EDUCATION = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
MARITAL = ["Single", "Married", "Divorced"]
PURPOSE = ["Auto", "Business", "Education", "Home", "Other"]
TERMS = [12, 24, 36, 48, 60]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FinPulse · Loan Default Risk",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_theme()


# ---------------- DATA & MODEL ----------------
@st.cache_data(show_spinner=False)
def load_data():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    # Demo dataset so the app still runs without the CSV
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame({
        "LoanID": [f"LN-{1000+i}" for i in range(n)],
        "Age": rng.integers(18, 70, n),
        "Income": rng.integers(15000, 150000, n),
        "LoanAmount": rng.integers(5000, 250000, n),
        "CreditScore": rng.integers(300, 850, n),
        "MonthsEmployed": rng.integers(0, 120, n),
        "NumCreditLines": rng.integers(1, 5, n),
        "InterestRate": np.round(rng.uniform(2, 25, n), 2),
        "LoanTerm": rng.choice(TERMS, n),
        "DTIRatio": np.round(rng.uniform(0.1, 0.9, n), 2),
        "Education": rng.choice(EDUCATION, n),
        "EmploymentType": rng.choice(EMPLOYMENT, n),
        "MaritalStatus": rng.choice(MARITAL, n),
        "HasMortgage": rng.choice(["Yes", "No"], n),
        "HasDependents": rng.choice(["Yes", "No"], n),
        "LoanPurpose": rng.choice(PURPOSE, n),
        "HasCoSigner": rng.choice(["Yes", "No"], n),
        "Default": rng.choice([0, 1], n, p=[0.88, 0.12]),
    })


@st.cache_data(show_spinner=False)
def read_upload(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def load_trained_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_model_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def calculate_emi(principal, annual_rate, term_months):
    """Monthly instalment for a fully amortising loan."""
    if annual_rate == 0 or term_months == 0:
        return principal / max(1, term_months)
    r = annual_rate / 100 / 12
    growth = math.pow(1 + r, term_months)
    return principal * r * growth / (growth - 1)


def heuristic_score(a):
    """Rule-based fallback used only if loan_model.pkl cannot be loaded."""
    pts = 0.0
    cs = a["creditscore"]
    pts += 35 if cs < 580 else 20 if cs < 670 else 8 if cs < 740 else 0
    pts += 20 if a["dtiratio"] > 0.45 else 10 if a["dtiratio"] > 0.30 else 0
    emi = calculate_emi(a["loanamount"], a["interestrate"], a["loanterm"])
    pts += 15 if emi / max(a["income"] / 12, 1) > 0.40 else 0
    pts += {"Unemployed": 30, "Part-time": 12}.get(a["employmenttype"], 0)
    pts += 10 if a["monthsemployed"] < 12 else -8 if a["monthsemployed"] > 36 else 0
    pts += -15 if a["hascosigner"] == "Yes" else 0
    pts += 5 if a["hasmortgage"] == "Yes" else 0
    pts += 4 if a["hasdependents"] == "Yes" else 0
    pts += 10 if a["interestrate"] > 15 else 0
    pts += 12 if a["loanamount"] > a["income"] * 1.5 else 0
    pts += -5 if a["education"] in ("Master's", "PhD") else 0
    return float(max(5.0, min(95.0, pts)))


def score_applicants(rows):
    """Default probability (%) for a list of applicant dicts (lower-case keys)."""
    model = load_trained_model()
    if model is not None:
        frame = add_engineered_features(pd.DataFrame(rows))
        return model.predict_proba(frame)[:, 1] * 100, "Scored by the trained Gradient Boosting pipeline · loan_model.pkl"
    return np.array([heuristic_score(r) for r in rows]), "Scored by the fallback rule engine · loan_model.pkl not loaded"


def dataset_key(df):
    return f"{len(df)}-{int(df['Default'].sum())}-{df.columns.size}"


# ---------------- SIDEBAR ----------------
def render_sidebar():
    with st.sidebar:
        logo = ui.svg('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/></svg>')
        ui.html(f"""
        <div class="fp-brand">
          <div class="fp-logo">{logo}</div>
          <div><div class="fp-brand-name">FinPulse</div><div class="fp-brand-tag">Loan Analytics &amp; Risk Portal</div></div>
        </div>
        """)

        ui.html('<div class="fp-side-label">Data source</div>')
        uploaded = st.file_uploader("Upload a custom CSV dataset", type=["csv"],
                                    help="Same columns as Loan_default.csv", label_visibility="collapsed")
        df, source = load_data(), ("Loan_default.csv" if CSV_PATH.exists() else "Demo dataset")
        if uploaded is not None:
            try:
                custom = read_upload(uploaded.getvalue())
                missing = [c for c in REQUIRED_COLS if c not in custom.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}. Using the default dataset.")
                else:
                    df, source = custom, uploaded.name
            except Exception:
                st.error("Could not read that file. Using the default dataset.")
        ui.html(f'<div class="fp-source-pill"><span class="fp-dot" style="--accent:{MINT}"></span>Active · <b>{source}</b></div>')

        rate = df["Default"].mean() * 100
        ui.html(f"""
        <div class="fp-side-label">Portfolio</div>
        <div class="fp-side-card">
          <div class="fp-side-big">{len(df):,}<small>applicants</small></div>
          <div style="margin-top:.6rem">
            <div class="fp-side-row"><span>Default rate</span><b style="color:{ROSE if rate > 15 else MINT}">{rate:.1f}%</b></div>
            <div class="fp-side-row"><span>Avg credit score</span><b>{df['CreditScore'].mean():.0f}</b></div>
            <div class="fp-side-row"><span>Avg loan</span><b>${df['LoanAmount'].mean():,.0f}</b></div>
            <div class="fp-side-row"><span>Avg income</span><b>${df['Income'].mean():,.0f}</b></div>
          </div>
        </div>
        """)

        meta = load_model_meta()
        model_ok = load_trained_model() is not None
        detail = (f"ROC-AUC {meta.get('roc_auc', 0):.3f} · Acc {meta.get('accuracy', 0) * 100:.1f}%"
                  if model_ok and meta else "Rule-based fallback active")
        ui.html(f"""
        <div class="fp-side-label">Model</div>
        <div class="fp-side-card fp-model-status {'' if model_ok else 'is-off'}">
          <span class="fp-live"></span>
          <div><b>{'Gradient Boosting · online' if model_ok else 'Model file not loaded'}</b><small>{detail}</small></div>
        </div>
        <div class="fp-side-label">Risk tiers</div>
        <div class="fp-side-card">
          <div class="fp-tier-row"><i style="background:{MINT};color:{MINT}"></i>Low risk<span>&lt; {ui.LOW_CUTOFF}%</span></div>
          <div class="fp-tier-row"><i style="background:{AMBER};color:{AMBER}"></i>Moderate<span>{ui.LOW_CUTOFF}–{ui.HIGH_CUTOFF}%</span></div>
          <div class="fp-tier-row"><i style="background:{ROSE};color:{ROSE}"></i>High risk<span>≥ {ui.HIGH_CUTOFF}%</span></div>
        </div>
        """)
    return df, source


DATA, SOURCE = None, None


def card_title(icon, title, sub=None):
    sub_html = f'<p class="fp-card-sub">{sub}</p>' if sub else ""
    ui.html(f'<div class="fp-card-title"><span class="fp-ic">{icon}</span>{title}</div>{sub_html}')


def rate_by(df, col, order=None):
    out = df.groupby(col, observed=True)["Default"].agg(["mean", "size"]).reset_index()
    out.columns = [col, "Rate", "Count"]
    out["Rate"] = out["Rate"] * 100
    if order is not None:
        out[col] = pd.Categorical(out[col], categories=order, ordered=True)
        out = out.sort_values(col)
    return out


def bar_gradient(top, bottom="rgba(139,124,255,0.08)"):
    return alt.Gradient(gradient="linear", x1=1, x2=1, y1=0, y2=1,
                        stops=[alt.GradientStop(color=top, offset=0), alt.GradientStop(color=bottom, offset=1)])


CREDIT_BINS = [0, 579, 669, 739, 900]
CREDIT_LABELS = ["Poor (<580)", "Fair (580–669)", "Good (670–739)", "Excellent (740+)"]
AGE_BINS = [17, 25, 35, 45, 55, 70]
AGE_LABELS = ["18–25", "26–35", "36–45", "46–55", "56+"]


# =====================================================
# 1. OVERVIEW
# =====================================================
def page_overview():
    df = DATA
    meta = load_model_meta()
    n, rate = len(df), df["Default"].mean() * 100
    emp = rate_by(df, "EmploymentType").sort_values("Rate", ascending=False)
    worst_emp = emp.iloc[0]

    left, right = st.columns([1.35, 1], gap="large", vertical_alignment="center")
    with left:
        auc = f"ROC-AUC {meta['roc_auc']:.3f}" if meta else "Risk engine online"
        ui.html(f"""
        <section class="fp-hero">
          <span class="fp-chip" data-reveal><b>LIVE</b>Gradient Boosting · {auc}</span>
          <h1 data-split>Credit risk, <span class="fp-serif">decoded.</span></h1>
          <p data-reveal data-delay="0.35">Explore {n:,} loan applications, see what really drives default,
          and score new applicants in real time with a tuned machine-learning model.</p>
        </section>
        """)
        with st.container(horizontal=True, gap="small"):
            with st.container(key="cta_primary", width="content"):
                st.page_link(PAGES["predict"], label="Score an applicant", icon=":material/bolt:")
            with st.container(key="cta_secondary", width="content"):
                st.page_link(PAGES["explorer"], label="Explore the data", icon=":material/table_view:")
    with right:
        ring_svg = ui.svg(f"""<svg viewBox="0 0 220 220" aria-hidden="true">
              <defs><linearGradient id="fpRing" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0%" stop-color="{IRIS}"/><stop offset="100%" stop-color="{AQUA}"/></linearGradient></defs>
              <circle cx="110" cy="110" r="103" class="fp-ring-orbit"/>
              <circle cx="110" cy="110" r="84" class="fp-ring-track"/>
              <circle cx="110" cy="110" r="84" class="fp-ring-val" pathLength="100" style="stroke-dashoffset:{100 - rate:.2f}" data-ring="{rate:.2f}"/>
            </svg>""", "fp-ring-svg")
        ui.html(f"""
        <div class="fp-orbit-card fp-spot" data-reveal data-delay="0.3">
          <div class="fp-orbit-head"><span>Portfolio default rate</span><span style="color:{MINT}">● Live</span></div>
          <div class="fp-ring-wrap">
            {ring_svg}
            <div class="fp-satellite"></div>
            <div class="fp-ring-center"><b>{ui.count(rate, 1, suffix='%')}</b><span>of loans default</span></div>
          </div>
          <div class="fp-orbit-stats">
            <div><span>Applicants</span><b>{ui.count(n / 1000, 1, suffix='k')}</b></div>
            <div><span>Avg FICO</span><b>{ui.count(df['CreditScore'].mean())}</b></div>
            <div><span>Avg loan</span><b>{ui.count(df['LoanAmount'].mean() / 1000, 0, '$', 'k')}</b></div>
          </div>
        </div>
        """)

    ui.marquee([
        ("loan applications", f"{n:,}"),
        ("overall default rate", f"{rate:.1f}%"),
        ("average loan amount", f"${df['LoanAmount'].mean():,.0f}"),
        ("average annual income", f"${df['Income'].mean():,.0f}"),
        ("average credit score", f"{df['CreditScore'].mean():.0f}"),
        (f"applicants default most · {worst_emp['Rate']:.1f}%", str(worst_emp["EmploymentType"])),
        ("average interest rate", f"{df['InterestRate'].mean():.1f}%"),
        ("models compared in Task 5", "5"),
    ])

    ui.kpi_row([
        {"label": "Total applications", "value": n, "sub": "Active portfolio count", "accent": IRIS, "bar": 100},
        {"label": "Default rate", "value": rate, "decimals": 1, "suffix": "%", "accent": ROSE if rate > 15 else MINT,
         "bar": rate, "sub": f"<b>{int(df['Default'].sum()):,}</b> loans defaulted"},
        {"label": "Avg loan amount", "value": df["LoanAmount"].mean(), "prefix": "$", "accent": AQUA,
         "bar": df["LoanAmount"].mean() / df["LoanAmount"].max() * 100, "sub": f"Avg income <b>${df['Income'].mean():,.0f}</b>"},
        {"label": "Avg credit score", "value": df["CreditScore"].mean(), "accent": AMBER,
         "bar": (df["CreditScore"].mean() - 300) / 550 * 100, "sub": "Target: <b>650+</b> FICO"},
    ])

    ui.section("Where default risk concentrates", "Default rate (%) by segment")
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="card_credit"):
        card_title("◎", "Credit score tier", "Lower tiers default noticeably more often")
        d = df.assign(Tier=pd.cut(df["CreditScore"], bins=CREDIT_BINS, labels=CREDIT_LABELS))
        d = rate_by(d, "Tier", CREDIT_LABELS)
        base = alt.Chart(d).encode(x=alt.X("Tier:N", sort=CREDIT_LABELS, title=None, axis=alt.Axis(labelAngle=0)),
                                   y=alt.Y("Rate:Q", title="Default rate (%)"))
        chart = (base.mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, size=54, color=bar_gradient(IRIS))
                 .encode(tooltip=[alt.Tooltip("Tier:N"), alt.Tooltip("Rate:Q", format=".2f", title="Default %"),
                                  alt.Tooltip("Count:Q", format=",")])
                 + base.mark_text(dy=-12, color=INK, fontSize=12, fontWeight=600).encode(text=alt.Text("Rate:Q", format=".1f")))
        ui.show_chart(chart, 290)
    with c2, st.container(key="card_emp"):
        card_title("◈", "Employment type", "Job stability is one of the strongest signals")
        base = alt.Chart(emp).encode(y=alt.Y("EmploymentType:N", sort="-x", title=None),
                                     x=alt.X("Rate:Q", title="Default rate (%)"))
        chart = (base.mark_bar(cornerRadiusTopRight=10, cornerRadiusBottomRight=10, size=30,
                               color=alt.Gradient(gradient="linear", x1=0, x2=1, y1=1, y2=1,
                                                  stops=[alt.GradientStop(color="rgba(94,234,212,0.1)", offset=0),
                                                         alt.GradientStop(color=AQUA, offset=1)]))
                 .encode(tooltip=["EmploymentType:N", alt.Tooltip("Rate:Q", format=".2f", title="Default %"),
                                  alt.Tooltip("Count:Q", format=",")])
                 + base.mark_text(dx=-24, color="#07080C", fontSize=12, fontWeight=700).encode(text=alt.Text("Rate:Q", format=".1f")))
        ui.show_chart(chart, 290)

    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(key="card_age"):
        card_title("◷", "Age band", "Younger borrowers carry the highest risk")
        d = rate_by(df.assign(Band=pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)), "Band", AGE_LABELS)
        base = alt.Chart(d).encode(x=alt.X("Band:N", sort=AGE_LABELS, title=None, axis=alt.Axis(labelAngle=0)),
                                   y=alt.Y("Rate:Q", title="Default rate (%)"))
        chart = (base.mark_area(interpolate="monotone", line={"color": IRIS, "strokeWidth": 2.5},
                                color=bar_gradient("rgba(139,124,255,0.45)", "rgba(139,124,255,0)"))
                 + base.mark_point(filled=True, size=70, color="#fff", stroke=IRIS, strokeWidth=3)
                 .encode(tooltip=["Band:N", alt.Tooltip("Rate:Q", format=".2f", title="Default %")]))
        ui.show_chart(chart, 270)
    with c4, st.container(key="card_purpose"):
        card_title("◇", "Loan purpose", "Default rate per purpose of the loan")
        d = rate_by(df, "LoanPurpose").sort_values("Rate", ascending=False)
        base = alt.Chart(d).encode(x=alt.X("LoanPurpose:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                                   y=alt.Y("Rate:Q", title="Default rate (%)", scale=alt.Scale(zero=False, padding=12)))
        chart = (base.mark_rule(color="rgba(251,191,36,0.35)", strokeWidth=2)
                 .encode(y=alt.Y("Rate:Q", scale=alt.Scale(zero=False, padding=12)), y2=alt.datum(d["Rate"].min() - 0.6))
                 + base.mark_circle(size=220, color=AMBER, opacity=1)
                 .encode(tooltip=["LoanPurpose:N", alt.Tooltip("Rate:Q", format=".2f", title="Default %"), alt.Tooltip("Count:Q", format=",")])
                 + base.mark_text(dy=-18, color=INK, fontSize=12, fontWeight=600).encode(text=alt.Text("Rate:Q", format=".1f")))
        ui.show_chart(chart, 270)

    # Insights computed from the data
    ages = rate_by(df.assign(Band=pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)), "Band", AGE_LABELS)
    young, old = ages.iloc[0]["Rate"], ages.iloc[-1]["Rate"]
    co = df.groupby("HasCoSigner")["Default"].mean() * 100
    hi_rate = df.loc[df["InterestRate"] > 20, "Default"].mean() * 100
    lo_rate = df.loc[df["InterestRate"] < 8, "Default"].mean() * 100
    ui.section("Key insights", "Calculated live from the active dataset")
    ui.html(f"""
    <div class="fp-insights">
      <div class="fp-insight fp-spot" data-reveal><span class="fp-insight-n">01 · AGE</span>
        <h4>{ui.count(young / max(old, 0.01), 1, suffix='×')}</h4>
        <p>Applicants aged <b>18–25</b> default at <b>{young:.1f}%</b>, versus <b>{old:.1f}%</b> for those aged 56+.</p></div>
      <div class="fp-insight fp-spot" data-reveal data-delay="0.08"><span class="fp-insight-n">02 · CO-SIGNER</span>
        <h4>{ui.count(abs(co.get('No', 0) - co.get('Yes', 0)), 1, '−' if co.get('No', 0) >= co.get('Yes', 0) else '+', ' pts')}</h4>
        <p>Having a co-signer lowers the default rate from <b>{co.get('No', 0):.1f}%</b> to <b>{co.get('Yes', 0):.1f}%</b>.</p></div>
      <div class="fp-insight fp-spot" data-reveal data-delay="0.16"><span class="fp-insight-n">03 · PRICING</span>
        <h4>{ui.count(hi_rate, 1, suffix='%')}</h4>
        <p>Loans priced above <b>20% interest</b> default at {hi_rate:.1f}%, compared with <b>{lo_rate:.1f}%</b> below 8%.</p></div>
    </div>
    """)


# =====================================================
# 2. DATASET EXPLORER
# =====================================================
@st.cache_data(show_spinner=False)
def preprocessing_report(_df, key):
    _, report = preprocess(_df)
    return report


def page_explorer():
    df = DATA
    ui.page_header("02", "Dataset Explorer", "Inspect every", "Inspect raw records, slice the portfolio with filters and verify data hygiene before modelling.", "record.")

    with st.container(key="card_filters"):
        card_title("⌕", "Filter the portfolio")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1.4], gap="medium")
        purpose = f1.selectbox("Loan purpose", ["All"] + sorted(df["LoanPurpose"].dropna().unique().tolist()))
        emp = f2.selectbox("Employment type", ["All"] + sorted(df["EmploymentType"].dropna().unique().tolist()))
        outcome = f3.selectbox("Outcome", ["All", "Repaid", "Defaulted"])
        lo, hi = int(df["CreditScore"].min()), int(df["CreditScore"].max())
        credit = f4.slider("Credit score range", lo, hi, (lo, hi))

    mask = df["CreditScore"].between(*credit)
    if purpose != "All":
        mask &= df["LoanPurpose"] == purpose
    if emp != "All":
        mask &= df["EmploymentType"] == emp
    if outcome != "All":
        mask &= df["Default"] == (1 if outcome == "Defaulted" else 0)
    fdf = df[mask]

    slice_rate = fdf["Default"].mean() * 100 if len(fdf) else 0
    ui.stat_strip([
        ("Matching records", len(fdf), 0, "", "", INK),
        ("Share of portfolio", len(fdf) / len(df) * 100, 1, "", "%", AQUA),
        ("Default rate in slice", slice_rate, 1, "", "%", ROSE if slice_rate > 15 else MINT),
    ])

    preview = fdf.head(5000)
    st.dataframe(
        preview, height=420, hide_index=True,
        column_config={
            "LoanID": st.column_config.TextColumn("Loan ID"),
            "Income": st.column_config.NumberColumn("Income", format="dollar"),
            "LoanAmount": st.column_config.NumberColumn("Loan amount", format="dollar"),
            "CreditScore": st.column_config.ProgressColumn("Credit score", min_value=300, max_value=850, format="%d"),
            "InterestRate": st.column_config.NumberColumn("Interest", format="%.2f%%"),
            "DTIRatio": st.column_config.NumberColumn("DTI", format="%.2f"),
            "Default": st.column_config.CheckboxColumn("Defaulted"),
        },
    )
    if len(fdf) > len(preview):
        st.caption(f"Showing the first {len(preview):,} of {len(fdf):,} matching rows. Statistics below use every matching row.")

    ui.html("<div style='height:1.2rem'></div>")
    tab_stats, tab_quality, tab_viz, tab_prep = st.tabs(["Statistical summary", "Data quality", "Visualize", "Preprocessing pipeline"])

    with tab_stats:
        num = fdf.select_dtypes(include="number")
        if num.empty:
            st.info("No numeric data in this slice.")
        else:
            st.dataframe(num.describe().T.style.format("{:,.2f}"), width="stretch")

    with tab_quality:
        missing = fdf.isnull().sum()
        ui.stat_strip([
            ("Missing cells", int(missing.sum()), 0, "", "", MINT if missing.sum() == 0 else AMBER),
            ("Duplicate rows", int(fdf.duplicated().sum()), 0, "", "", MINT if fdf.duplicated().sum() == 0 else AMBER),
            ("Columns", fdf.shape[1], 0, "", "", INK),
        ])
        quality = pd.DataFrame({
            "Column": fdf.columns,
            "Type": [str(t) for t in fdf.dtypes],
            "Missing": missing.values,
            "Missing %": (missing.values / max(len(fdf), 1)) * 100,
            "Unique values": [fdf[c].nunique() for c in fdf.columns],
        })
        st.dataframe(quality, hide_index=True, width="stretch",
                     column_config={"Missing %": st.column_config.ProgressColumn("Missing %", min_value=0, max_value=100, format="%.1f%%")})

    with tab_viz:
        if fdf.empty:
            st.info("No rows match the filters.")
        else:
            v1, v2 = st.columns(2, gap="medium")
            num_cols = [c for c in fdf.select_dtypes(include="number").columns if c != "Default"]
            cat_cols = [c for c in fdf.columns if c not in num_cols + ["Default", "LoanID"]]
            sample = fdf.sample(min(len(fdf), 4000), random_state=7).assign(
                Outcome=lambda d: np.where(d["Default"] == 1, "Defaulted", "Repaid"))
            color = alt.Color("Outcome:N", scale=alt.Scale(domain=["Repaid", "Defaulted"], range=[IRIS, ROSE]), title=None)
            with v1, st.container(key="card_hist"):
                card_title("▥", "Numerical columns", "Histogram and box plot")
                col = st.selectbox("Numeric column", num_cols, index=num_cols.index("CreditScore") if "CreditScore" in num_cols else 0)
                hist = alt.Chart(sample).mark_bar(opacity=.85, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col), y=alt.Y("count():Q", title="Count", stack=True), color=color)
                ui.show_chart(hist, 240)
                box = alt.Chart(sample).mark_boxplot(size=26, median={"color": "#fff"}, rule={"color": MUTED}, ticks=True, outliers={"color": FAINT, "size": 8}).encode(
                    x=alt.X(f"{col}:Q", title=None), y=alt.Y("Outcome:N", title=None), color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Repaid", "Defaulted"], range=[IRIS, ROSE]), legend=None))
                ui.show_chart(box, 120)
            with v2, st.container(key="card_cat"):
                card_title("◔", "Categorical columns", "Bar chart and pie chart")
                col = st.selectbox("Categorical column", cat_cols)
                counts = fdf[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                counts["Share"] = counts["Count"] / counts["Count"].sum()
                palette = [IRIS, AQUA, AMBER, ROSE, "#60A5FA", "#F0ABFC"]
                bar = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=34, color=bar_gradient(AQUA, "rgba(94,234,212,0.08)")).encode(
                    x=alt.X(f"{col}:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Count:Q"), tooltip=[f"{col}:N", alt.Tooltip("Count:Q", format=",")])
                ui.show_chart(bar, 200)
                pie = alt.Chart(counts).mark_arc(innerRadius=58, outerRadius=92, cornerRadius=6, padAngle=0.02).encode(
                    theta="Count:Q", color=alt.Color(f"{col}:N", scale=alt.Scale(range=palette), legend=alt.Legend(orient="right", title=None)),
                    tooltip=[f"{col}:N", alt.Tooltip("Share:Q", format=".1%")])
                ui.show_chart(pie, 200)

    with tab_prep:
        report = preprocessing_report(df, dataset_key(df))
        sample_cols = ", ".join(clean_column_names(df.head(0)).columns[:3])
        steps = [
            ("Understand the data", f"Loaded <b>{report['rows_loaded']:,}</b> rows × {df.shape[1]} columns and inspected <code>head()</code>, <code>info()</code>, <code>describe()</code>.", "✓ done"),
            ("Clean column names", f"Lower-cased and stripped names → <code>{sample_cols}</code> …", "✓ done"),
            ("Remove duplicates", f"<code>drop_duplicates()</code> removed <b>{report['duplicates_removed']:,}</b> duplicate rows.", f"{report['duplicates_removed']:,} rows"),
            ("Handle missing values", f"Median for numeric, mode for categorical. <b>{report['missing_values_filled']:,}</b> missing cells found.", f"{report['missing_values_filled']:,} cells"),
            ("Feature engineering", "Added <code>income_to_loan</code> and <code>credit_term_interaction</code>.", "+2 features"),
            ("Handle outliers (IQR)", f"Kept values within Q1 − 1.5·IQR and Q3 + 1.5·IQR. Removed <b>{report['outliers_removed']:,}</b> rows.", f"{report['rows_final']:,} kept"),
            ("Encode categories", f"<code>OneHotEncoder</code> on {len(CAT_COLS)} categorical columns.", "✓ done"),
            ("Scale numbers", f"<code>MinMaxScaler</code> on {len(NUM_COLS)} numeric columns inside the model pipeline.", "✓ done"),
        ]
        items = "".join(
            f'<div class="fp-step fp-spot" data-reveal data-delay="{i * 0.04:.2f}"><span class="fp-step-n">{i + 1:02d}</span>'
            f'<div><b>{t}</b><small>{d}</small></div><span class="fp-check">{c}</span></div>'
            for i, (t, d, c) in enumerate(steps))
        ui.html(f'<div class="fp-steps">{items}</div>')
        ui.note("Feature selection: <b>MaritalStatus, LoanPurpose, LoanTerm</b> and <b>DTIRatio</b> are left out of the model because "
                "their correlation with <b>Default</b> is close to zero (see <code>calc_corr.py</code>).")


# =====================================================
# 3. RISK ANALYTICS
# =====================================================
def encoded_frame(df):
    enc = df.drop(columns=["LoanID"], errors="ignore").copy()
    for c in enc.columns:
        if not pd.api.types.is_numeric_dtype(enc[c]):
            vals = set(enc[c].dropna().unique())
            enc[c] = enc[c].map({"No": 0, "Yes": 1}) if vals <= {"Yes", "No"} else enc[c].astype("category").cat.codes
    return enc


def page_analytics():
    df = DATA
    ui.page_header("03", "Risk Analytics", "Find the", "Analyse relationships between features, correlation with default and what the trained model relies on.", "signal.")

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != "Default"]
    with st.container(key="card_scatter"):
        card_title("⤢", "Feature relationship explorer", "A random sample of 3,000 applicants, coloured by outcome")
        s1, s2 = st.columns(2, gap="medium")
        x_var = s1.selectbox("X-axis feature", num_cols, index=num_cols.index("CreditScore"))
        y_var = s2.selectbox("Y-axis feature", num_cols, index=num_cols.index("InterestRate"))
        sample = df.sample(min(3000, len(df)), random_state=42).assign(
            Outcome=lambda d: np.where(d["Default"] == 1, "Defaulted", "Repaid")).sort_values("Default")
        scatter = alt.Chart(sample).mark_circle(size=34, opacity=.6, stroke="rgba(0,0,0,0.3)", strokeWidth=.5).encode(
            x=alt.X(f"{x_var}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{y_var}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Repaid", "Defaulted"], range=["rgba(139,124,255,0.75)", ROSE]), title=None),
            tooltip=[x_var, y_var, "Outcome"]).interactive()
        ui.show_chart(scatter, 380)

    enc = encoded_frame(df)
    corr_t = enc.corr()["Default"].drop("Default").sort_values(key=abs, ascending=False).reset_index()
    corr_t.columns = ["Feature", "Correlation"]
    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1, st.container(key="card_corr_target"):
        card_title("±", "Correlation with Default", "Positive raises risk, negative lowers it")
        chart = alt.Chart(corr_t).mark_bar(cornerRadius=5, size=14).encode(
            y=alt.Y("Feature:N", sort=None, title=None), x=alt.X("Correlation:Q", title="Pearson r"),
            color=alt.condition("datum.Correlation > 0", alt.value(ROSE), alt.value(MINT)),
            tooltip=["Feature", alt.Tooltip("Correlation:Q", format=".3f")])
        ui.show_chart(chart, 430)
    with c2, st.container(key="card_heatmap"):
        card_title("▦", "Correlation matrix", "Numeric features")
        corr = df[num_cols + ["Default"]].corr().round(2)
        long = corr.reset_index().melt(id_vars="index")
        long.columns = ["A", "B", "r"]
        order = corr.columns.tolist()
        base = alt.Chart(long).encode(x=alt.X("A:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40)),
                                      y=alt.Y("B:N", sort=order, title=None))
        heat = (base.mark_rect(cornerRadius=4, stroke="#07080C", strokeWidth=2).encode(
                    color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[AQUA, "#11131A", IRIS]), legend=None),
                    tooltip=["A", "B", alt.Tooltip("r:Q", format=".2f")])
                + base.mark_text(fontSize=9.5).encode(text=alt.Text("r:Q", format=".2f"),
                                                    color=alt.condition("abs(datum.r) > 0.5", alt.value("#fff"), alt.value(MUTED))))
        ui.show_chart(heat, 430)

    meta = load_model_meta()
    if meta.get("feature_importance"):
        with st.container(key="card_importance"):
            card_title("★", "What the model relies on", "Gradient Boosting feature importance (one-hot columns grouped)")
            imp = pd.DataFrame(list(meta["feature_importance"].items()), columns=["Feature", "Importance"])
            imp["Importance"] *= 100
            base = alt.Chart(imp).encode(y=alt.Y("Feature:N", sort="-x", title=None), x=alt.X("Importance:Q", title="Importance (%)"))
            chart = (base.mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, size=18,
                                   color=alt.Gradient(gradient="linear", x1=0, x2=1, y1=1, y2=1,
                                                      stops=[alt.GradientStop(color="rgba(139,124,255,0.15)", offset=0), alt.GradientStop(color=IRIS, offset=1)]))
                     .encode(tooltip=["Feature", alt.Tooltip("Importance:Q", format=".2f")])
                     + base.mark_text(dx=18, color=INK, fontSize=11).encode(text=alt.Text("Importance:Q", format=".1f")))
            ui.show_chart(chart, 360)


# =====================================================
# 4. SMART PREDICTION
# =====================================================
def applicant_factors(a, payment_ratio):
    good, bad = [], []
    if a["creditscore"] >= 670:
        good.append(("Prime credit score", f"FICO {a['creditscore']} is 670 or higher"))
    elif a["creditscore"] < 580:
        bad.append(("Poor credit score", f"FICO {a['creditscore']} is below 580"))
    if a["hascosigner"] == "Yes":
        good.append(("Co-signer present", "A second party guarantees repayment"))
    if a["dtiratio"] <= 0.35:
        good.append(("Healthy debt-to-income", f"DTI of {a['dtiratio']:.0%} is under 35%"))
    elif a["dtiratio"] > 0.40:
        bad.append(("High debt-to-income", f"DTI of {a['dtiratio']:.0%} is above 40%"))
    if a["monthsemployed"] >= 36:
        good.append(("Stable employment", f"{a['monthsemployed']} months with the current employer"))
    elif a["monthsemployed"] < 12:
        bad.append(("Short employment history", f"Only {a['monthsemployed']} months employed"))
    if a["interestrate"] > 15:
        bad.append(("High interest rate", f"{a['interestrate']:.1f}% raises the repayment burden"))
    elif a["interestrate"] < 8:
        good.append(("Low interest rate", f"{a['interestrate']:.1f}% keeps instalments affordable"))
    if a["employmenttype"] == "Unemployed":
        bad.append(("No current employment", "Unemployed applicants default most often"))
    if payment_ratio > 0.40:
        bad.append(("Heavy instalment burden", f"The EMI takes {payment_ratio:.0%} of monthly income"))
    if a["age"] < 26:
        bad.append(("Young applicant", "18–25 year-olds have the highest default rate"))
    if a["income"] >= 2 * a["loanamount"]:
        good.append(("Strong income cover", "Annual income is at least twice the loan"))
    return good, bad


def page_predict():
    ui.page_header("04", "Smart Prediction", "Score an", "Fill in the applicant's details to get an instant default-risk assessment from the trained machine-learning model.", "applicant.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(key="card_personal"):
        card_title("◉", "Personal profile")
        age = st.number_input("Age", 18, 90, 35, 1, help="Applicant age in years")
        education = st.selectbox("Education level", EDUCATION, index=1)
        marital = st.selectbox("Marital status", MARITAL, index=1)
        dependents = st.segmented_control("Has dependents?", ["No", "Yes"], default="No", required=True, key="dep")
    with c2, st.container(key="card_employment"):
        card_title("◈", "Employment & credit")
        income = st.number_input("Annual income ($)", value=65000, step=5000)
        employment = st.selectbox("Employment type", EMPLOYMENT, index=0)
        months = st.number_input("Months employed", 0, 480, 36, 6)
        credit = st.slider("Credit score (FICO)", 300, 850, 680)
        lines = st.number_input("Number of credit lines", 1, 20, 3, 1)
        dti = st.number_input("Debt-to-income (DTI) ratio", 0.01, 0.99, 0.35, 0.05, help="Existing monthly debt ÷ monthly gross income")
    with c3, st.container(key="card_loan"):
        card_title("◇", "Loan request")
        amount = st.number_input("Requested loan amount ($)", value=25000, step=2500)
        rate = st.number_input("Interest rate (%)", 1.0, 35.0, 10.5, 0.5)
        term = st.selectbox("Loan term (months)", TERMS, index=2)
        purpose = st.selectbox("Loan purpose", PURPOSE, index=0)
        mortgage = st.segmented_control("Has mortgage?", ["No", "Yes"], default="No", required=True, key="mort")
        cosigner = st.segmented_control("Has co-signer?", ["No", "Yes"], default="No", required=True, key="cos")

    if income <= 0 or amount <= 0:
        ui.note("Enter an annual income and a loan amount above zero to see the assessment.", "warn")
        return

    emi = calculate_emi(amount, rate, term)
    pay_ratio = emi / max(income / 12, 1)
    ui.stat_strip([
        ("Estimated monthly payment (EMI)", emi, 2, "$", "", IRIS),
        ("Payment-to-income ratio", pay_ratio * 100, 1, "", "%", ROSE if pay_ratio > 0.4 else MINT),
        ("Loan-to-income ratio", amount / income, 2, "", "×", INK),
    ])

    _, mid, _ = st.columns([1, 1.2, 1])
    if mid.button("Run risk assessment", type="primary", icon=":material/bolt:", width="stretch"):
        st.session_state.assessed = True
    if not st.session_state.get("assessed"):
        ui.note("Adjust the profile above, then run the assessment. Once it has run, the result updates live as you edit.", "info")
        return

    a = {"age": int(age), "income": int(income), "loanamount": int(amount), "creditscore": int(credit),
         "monthsemployed": int(months), "numcreditlines": int(lines), "interestrate": float(rate),
         "loanterm": int(term), "dtiratio": float(dti), "education": education, "employmenttype": employment,
         "maritalstatus": marital, "hasmortgage": mortgage or "No", "hasdependents": dependents or "No",
         "loanpurpose": purpose, "hascosigner": cosigner or "No"}

    levers = [("Credit score +50", {"creditscore": min(850, a["creditscore"] + 50)}),
              ("Borrow 20% less", {"loanamount": int(a["loanamount"] * 0.8)}),
              ("Interest rate −2 pts", {"interestrate": max(1.0, a["interestrate"] - 2)}),
              ("24 more months employed", {"monthsemployed": a["monthsemployed"] + 24})]
    if a["hascosigner"] == "No":
        levers.insert(0, ("Add a co-signer", {"hascosigner": "Yes"}))
    if a["employmenttype"] != "Full-time":
        levers.insert(0, ("Move to full-time work", {"employmenttype": "Full-time"}))
    levers = levers[:4]
    scores, source = score_applicants([a] + [{**a, **chg} for _, chg in levers])
    score = float(scores[0])

    ui.section("Assessment", "Updates live while you edit the profile")
    g, f = st.columns([1, 1.15], gap="medium")
    with g:
        ui.gauge(score, source, key="predict")
    with f:
        good, bad = applicant_factors(a, pay_ratio)
        fa, fb = st.columns(2, gap="small")
        with fa:
            ui.factor_list("Favourable", good, "good")
        with fb:
            ui.factor_list("Risk signals", bad, "bad")
        cards = []
        for i, ((label, _), s) in enumerate(zip(levers, scores[1:])):
            delta = s - score
            cls = "down" if delta < -0.05 else "up" if delta > 0.05 else "flat"
            cards.append(f'<div class="fp-lever fp-spot" data-reveal data-rid="lever-{i}" data-delay="{i * 0.06:.2f}"><span>{label}</span>'
                         f'<b>{ui.count(s, 1, suffix="%", key=f"lever-{i}")}<em class="{cls}">{delta:+.1f}</em></b></div>')
        ui.html(f'<div class="fp-section" style="margin-top:1.2rem"><h3>Levers that change the outcome</h3>'
                f'<span class="fp-section-hint">Model re-scored per change</span></div><div class="fp-levers">{"".join(cards)}</div>')


# =====================================================
# 5. WHAT-IF SIMULATOR
# =====================================================
# factor -> (field, sweep start, sweep end); the sweep stretches to include the current value
SWEEPS = {
    "Credit score": ("creditscore", 300, 850),
    "Interest rate": ("interestrate", 2, 25),
    "Annual income": ("income", 15000, 250000),
    "Loan amount": ("loanamount", 5000, 250000),
    "Months employed": ("monthsemployed", 0, 120),
}


def page_simulator():
    ui.page_header("05", "What-If Simulator", "Pull the", "Move the levers and watch the default probability respond instantly. See how each factor bends the risk curve.", "levers.")

    left, right = st.columns([1.05, 1], gap="medium")
    with left, st.container(key="card_levers"):
        card_title("⚙", "Simulation levers")
        l1, l2 = st.columns(2, gap="medium")
        with l1:
            credit = st.slider("Credit score (FICO)", 300, 850, 680)
            amount = st.number_input("Loan amount ($)", value=25000, step=5000, key="sim_amount")
            income = st.number_input("Annual income ($)", value=65000, step=5000, key="sim_income")
            months = st.slider("Months employed", 0, 120, 36, 6)
            age = st.slider("Age", 18, 69, 35)
        with l2:
            rate = st.slider("Interest rate (%)", 2.0, 25.0, 10.5, 0.5)
            dti = st.slider("Debt-to-income ratio", 0.05, 0.90, 0.35, 0.01)
            term = st.select_slider("Loan term (months)", TERMS, value=36)
            employment = st.selectbox("Employment type", EMPLOYMENT, index=0, key="sim_emp")
            cosigner = st.segmented_control("Co-signer", ["No", "Yes"], default="No", required=True, key="sim_cos")
            mortgage = st.segmented_control("Mortgage", ["No", "Yes"], default="No", required=True, key="sim_mort")

    if income <= 0 or amount <= 0:
        with right:
            ui.note("Enter an annual income and a loan amount above zero to run the simulation.", "warn")
        return

    a = {"age": int(age), "income": int(income), "loanamount": int(amount), "creditscore": int(credit),
         "monthsemployed": int(months), "numcreditlines": 3, "interestrate": float(rate), "loanterm": int(term),
         "dtiratio": float(dti), "education": "Bachelor's", "employmenttype": employment, "maritalstatus": "Married",
         "hasmortgage": mortgage or "No", "hasdependents": "No", "loanpurpose": "Auto", "hascosigner": cosigner or "No"}
    scores, source = score_applicants([a])
    with right:
        ui.gauge(float(scores[0]), source, key="sim", compact=False)

    with st.container(key="card_sweep"):
        card_title("∿", "Risk curve", "Default probability across the full range of one factor, everything else held at your settings")
        driver = st.segmented_control("Factor", list(SWEEPS), default="Credit score", key="sweep", label_visibility="collapsed") or "Credit score"
        field, lo, hi = SWEEPS[driver]
        cur = a[field]
        lo, hi = min(lo, cur), (hi if cur <= hi else cur * 1.25)
        cast = float if field == "interestrate" else (lambda v: int(round(v)))
        xs = list(dict.fromkeys(cast(v) for v in np.linspace(lo, hi, 60)))
        curve_scores, _ = score_applicants([{**a, field: x} for x in xs])
        curve = pd.DataFrame({"x": xs, "Risk": curve_scores})
        current = pd.DataFrame({"x": [a[field]], "Risk": [float(scores[0])]})
        bands = pd.DataFrame({"y": [0, ui.LOW_CUTOFF, ui.HIGH_CUTOFF], "y2": [ui.LOW_CUTOFF, ui.HIGH_CUTOFF, 100],
                              "c": [MINT, AMBER, ROSE]})
        top = max(float(curve["Risk"].max()), float(scores[0])) + 8
        yscale = alt.Scale(domain=[0, min(100, top)], nice=False, clamp=True)
        band_layer = alt.Chart(bands).mark_rect(opacity=.06).encode(y=alt.Y("y:Q", scale=yscale), y2="y2:Q", color=alt.Color("c:N", scale=None))
        area = alt.Chart(curve).mark_area(interpolate="monotone", line={"color": IRIS, "strokeWidth": 2.5},
                                          color=bar_gradient("rgba(139,124,255,0.4)", "rgba(139,124,255,0)")).encode(
            x=alt.X("x:Q", title=driver, scale=alt.Scale(zero=False, nice=False)),
            y=alt.Y("Risk:Q", title="Default probability (%)", scale=yscale),
            tooltip=[alt.Tooltip("x:Q", title=driver, format=",.2~f"), alt.Tooltip("Risk:Q", format=".1f", title="Risk %")])
        rule = alt.Chart(current).mark_rule(color="#fff", strokeDash=[4, 4], opacity=.5).encode(x="x:Q")
        dot = alt.Chart(current).mark_point(filled=True, size=160, color="#fff", stroke=IRIS, strokeWidth=4).encode(x="x:Q", y=alt.Y("Risk:Q", scale=yscale))
        ui.show_chart(band_layer + area + rule + dot, 320)


# =====================================================
# 6. MODEL EVALUATION (TASK 5)
# =====================================================
@st.cache_data(show_spinner=False)
def run_model_pipeline(_data, key):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    # Same cleaning + feature engineering as the deployed model (features.py)
    df_ml, _ = preprocess(_data)
    # Down-sample to keep the page responsive (10,000 records)
    if len(df_ml) > 10000:
        df_ml = df_ml.sample(n=10000, random_state=42)
    X, y = df_ml[NUM_COLS + CAT_COLS], df_ml[TARGET]

    # Split first, so the scaler / encoder only ever learn from training data (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def make_pipe(clf):
        prep = ColumnTransformer([("num", StandardScaler(), NUM_COLS),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
        return Pipeline([("prep", prep), ("clf", clf)])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, fitted = [], {}
    for name, clf in models.items():
        pipe = make_pipe(clf).fit(X_train, y_train)
        fitted[name] = pipe
        y_train_pred, y_test_pred = pipe.predict(X_train), pipe.predict(X_test)
        train_acc, test_acc = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
        scores = cross_validate(make_pipe(clf), X_train, y_train, cv=cv, scoring=["accuracy", "roc_auc"])

        fit_status = "Good fit"
        if train_acc - test_acc > 0.05:
            fit_status = "Overfitting"
        elif train_acc < 0.6 and test_acc < 0.6:
            fit_status = "Underfitting"

        results.append({
            "Model": name,
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Precision": precision_score(y_test, y_test_pred, zero_division=0),
            "Recall": recall_score(y_test, y_test_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_test_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
            "CV Accuracy": scores["test_accuracy"].mean(),
            "CV ROC-AUC": scores["test_roc_auc"].mean(),
            "CV Spread (Std)": scores["test_roc_auc"].std(),
            "Fit Status": fit_status,
        })

    results = pd.DataFrame(results)
    # Accuracy is misleading here (88% by always predicting "no default"), so rank by CV ROC-AUC
    best_name = results.sort_values("CV ROC-AUC", ascending=False).iloc[0]["Model"]

    grids = {
        "Random Forest": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]},
        "Gradient Boosting": {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1, 0.2], "clf__max_depth": [3, 4, 5]},
        "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 0.5, 1.0]},
        "Decision Tree": {"clf__max_depth": [None, 5, 10, 20], "clf__min_samples_split": [2, 5, 10]},
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    }
    search = RandomizedSearchCV(make_pipe(models[best_name]), grids[best_name], n_iter=5, cv=3,
                                scoring="roc_auc", random_state=42)
    search.fit(X_train, y_train)
    tuned_prob = search.predict_proba(X_test)[:, 1]
    tuned_pred = search.predict(X_test)
    tuning = {
        "Best Params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "Tuned Test Acc": accuracy_score(y_test, tuned_pred),
        "Tuned ROC-AUC": roc_auc_score(y_test, tuned_prob),
        "Confusion": confusion_matrix(y_test, tuned_pred).tolist(),
        "Train rows": len(X_train), "Test rows": len(X_test),
    }
    return results, best_name, tuning


MODEL_NOTES = [
    ("Logistic Regression", "Baseline", "A statistical baseline that estimates default probability with a linear equation. Fast and easy to interpret, but it struggles when relationships are non-linear."),
    ("Decision Tree", "Interpretable", "A flowchart of questions such as <i>“Is income &lt; $50k?”</i>. Very readable, but prone to <b>overfitting</b>: it memorises the training data instead of learning general rules."),
    ("Random Forest", "Bagging", "Hundreds of decision trees trained on random subsets, voting together. <b>Why choose it:</b> it fixes the overfitting of single trees and is robust out of the box."),
    ("AdaBoost", "Boosting", "Trees are built one after another, each focusing on the applicants the previous ones misclassified. <b>Why choose it:</b> strong on borderline, hard-to-predict cases."),
    ("Gradient Boosting", "Advanced boosting", "Like AdaBoost, but each new tree corrects the remaining error using gradient descent. <b>Why choose it:</b> often the most accurate, with careful hyperparameter tuning."),
]


def page_models():
    df = DATA
    ui.page_header("06", "Task 5 · Model Evaluation & Tuning", "Model", "Evaluation, overfitting checks, 5-fold cross-validation, comparison of five models and hyperparameter tuning.", "lab.")

    meta = load_model_meta()
    if meta:
        ui.kpi_row([
            {"label": "Deployed model", "value": meta["params"]["n_estimators"], "suffix": " trees", "accent": IRIS, "sub": "Gradient Boosting · depth 5 · lr 0.05"},
            {"label": "Validation accuracy", "value": meta["accuracy"] * 100, "decimals": 2, "suffix": "%", "accent": MINT, "bar": meta["accuracy"] * 100, "sub": f"On <b>{meta['test_rows']:,}</b> held-out loans"},
            {"label": "Validation ROC-AUC", "value": meta["roc_auc"], "decimals": 3, "accent": AQUA, "bar": meta["roc_auc"] * 100, "sub": "0.5 = random · 1.0 = perfect"},
            {"label": "Training rows", "value": meta["train_rows"], "accent": AMBER, "sub": "After cleaning &amp; IQR outlier removal"},
        ])

    holder = st.empty()
    holder.html('<div class="fp-loader"><div class="fp-loader-orb"></div><div><b style="color:#fff">Training and cross-validating five models…</b>'
                '<div style="color:#8A90A2;font-size:.85rem">Runs once on a 10,000-row sample, then it is cached. This takes about a minute.</div></div></div>')
    results, best, tuning = run_model_pipeline(df, dataset_key(df))
    holder.empty()

    ui.section("1–4 · Evaluation, overfitting check, cross-validation & comparison", "Test set = 20% stratified split")
    pct = lambda c, fmt="%.3f": st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format=fmt)
    st.dataframe(results, hide_index=True, width="stretch", column_config={
        "Train Acc": pct("Train Acc"), "Test Acc": pct("Test Acc"), "Precision": pct("Precision"), "Recall": pct("Recall"),
        "F1-Score": pct("F1-Score"), "ROC-AUC": pct("ROC-AUC"), "CV Accuracy": pct("CV Accuracy"), "CV ROC-AUC": pct("CV ROC-AUC"),
        "CV Spread (Std)": st.column_config.NumberColumn("CV Spread (Std)", format="%.4f"),
    })

    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1, st.container(key="card_metrics"):
        card_title("▤", "Metric comparison", "Test-set scores per model")
        long = results.melt(id_vars="Model", value_vars=["Test Acc", "ROC-AUC", "F1-Score", "Recall"], var_name="Metric", value_name="Score")
        chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=90)), xOffset="Metric:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[IRIS, AQUA, AMBER, ROSE]), title=None),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")])
        ui.show_chart(chart, 320)
    with c2, st.container(key="card_fit"):
        card_title("⇅", "Overfitting check", "Train vs test accuracy · a large gap means overfitting")
        long = results.melt(id_vars="Model", value_vars=["Train Acc", "Test Acc"], var_name="Split", value_name="Accuracy")
        chart = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5).encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=80)), y=alt.Y("Accuracy:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Split:N", scale=alt.Scale(range=[ROSE, MINT]), title=None),
            tooltip=["Model", "Split", alt.Tooltip("Accuracy:Q", format=".4f")])
        ui.show_chart(chart, 320)

    best_row = results[results["Model"] == best].iloc[0]
    ui.section("5 · Hyperparameter tuning", f"RandomizedSearchCV on {best}")
    params = " · ".join(f"<b>{k}</b> = {v}" for k, v in tuning["Best Params"].items())
    ui.html(f"""
    <div class="fp-winner" data-reveal>
      <div class="fp-eyebrow"><span class="fp-index">WINNER</span><span class="fp-dash"></span>Highest 5-fold cross-validated ROC-AUC</div>
      <h2>{best}</h2>
      <p>Best parameters found: {params}</p>
      <div class="fp-winner-stats">
        <div><span>CV ROC-AUC</span><b>{ui.count(best_row['CV ROC-AUC'], 4)}</b></div>
        <div><span>Baseline test AUC</span><b>{ui.count(best_row['ROC-AUC'], 4)}</b></div>
        <div><span>Tuned test AUC</span><b style="color:{AQUA}">{ui.count(tuning['Tuned ROC-AUC'], 4)}</b></div>
        <div><span>Baseline test acc</span><b>{ui.count(best_row['Test Acc'] * 100, 2, suffix='%')}</b></div>
        <div><span>Tuned test acc</span><b style="color:{MINT}">{ui.count(tuning['Tuned Test Acc'] * 100, 2, suffix='%')}</b></div>
      </div>
    </div>
    """)

    if meta and best != meta.get("model"):
        ui.note(f"<b>Why is a different model deployed?</b> This comparison runs on a 10,000-row sample so the page stays fast. "
                f"The deployed <b>{meta['model']}</b> was trained on all {meta['train_rows']:,} training rows, where it reaches a test "
                f"ROC-AUC of <b>{meta['roc_auc']:.3f}</b>. Boosting keeps improving with more data, while {best} levels off sooner.", "info")
        ui.html("<div style='height:.6rem'></div>")

    c3, c4 = st.columns([1, 1.3], gap="medium")
    with c3, st.container(key="card_confusion"):
        card_title("▩", "Confusion matrix", f"Tuned {best} on the test set")
        (tn, fp), (fn, tp) = tuning["Confusion"]
        cm = pd.DataFrame({"Actual": ["Repaid", "Repaid", "Defaulted", "Defaulted"],
                           "Predicted": ["Repaid", "Defaulted", "Repaid", "Defaulted"], "Count": [tn, fp, fn, tp]})
        base = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=["Repaid", "Defaulted"], axis=alt.Axis(labelAngle=0, orient="top")),
                                    y=alt.Y("Actual:N", sort=["Repaid", "Defaulted"]))
        chart = (base.mark_rect(cornerRadius=10, stroke="#07080C", strokeWidth=4).encode(
                    color=alt.Color("Count:Q", scale=alt.Scale(type="log", range=["#151827", IRIS]), legend=None))
                 + base.mark_text(fontSize=20, fontWeight=600, color="#fff").encode(text=alt.Text("Count:Q", format=",")))
        ui.show_chart(chart, 260)
    with c4:
        ui.note(f"<b>Reading the scores.</b> Only about {df['Default'].mean():.0%} of applicants default, so a model that always "
                f"predicts “repaid” already reaches ~{1 - df['Default'].mean():.0%} accuracy. That is why <b>recall</b> for defaulters looks low at "
                "the standard 0.5 cut-off, and why models are ranked by <b>ROC-AUC</b>: it measures how well defaulters are ranked above "
                "non-defaulters across every threshold. The prediction pages use the full probability with risk tiers instead of a 0.5 cut-off.", "info")
        ui.html("<div style='height:.8rem'></div>")
        ui.note(f"<b>Cross-validation.</b> Instead of testing once, the training data is cut into 5 folds and every model is trained 5 times, "
                f"each time validated on a different fold. {best} averages <b>{best_row['CV ROC-AUC']:.4f}</b> ROC-AUC with a spread of only "
                f"<b>{best_row['CV Spread (Std)']:.4f}</b>, so its score is stable rather than a lucky split.", "good")

    ui.section("6 · Advanced models implemented", "Which to choose and why")
    cards = "".join(
        f'<div class="fp-model fp-spot {"is-best" if name == best else ""}" data-reveal data-delay="{i * 0.06:.2f}">'
        f'{"<span class=fp-best-ribbon>Selected</span>" if name == best else ""}'
        f'<span class="fp-model-tag">0{i + 1} · {tag}</span><h4>{name}</h4><p>{text}</p></div>'
        for i, (name, tag, text) in enumerate(MODEL_NOTES))
    ui.html(f'<div class="fp-models">{cards}</div>')


def footer():
    ui.html(f"""
    <div class="fp-foot"><span><b>FinPulse</b> · Loan Default Prediction &amp; Risk Analytics</span>
    <span>{SOURCE} · {len(DATA):,} records · Streamlit · scikit-learn</span></div>
    """)


# ---------------- NAVIGATION ----------------
PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    "explorer": st.Page(page_explorer, title="Dataset Explorer", icon=":material/table_view:", url_path="explorer"),
    "analytics": st.Page(page_analytics, title="Risk Analytics", icon=":material/insights:", url_path="analytics"),
    "predict": st.Page(page_predict, title="Smart Prediction", icon=":material/bolt:", url_path="predict"),
    "simulator": st.Page(page_simulator, title="What-If Simulator", icon=":material/tune:", url_path="simulator"),
    "models": st.Page(page_models, title="Model Evaluation", icon=":material/science:", url_path="models"),
}
pg = st.navigation(list(PAGES.values()), position="top")

DATA, SOURCE = render_sidebar()
pg.run()
footer()
