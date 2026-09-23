import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aegis | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ['Time', 'Transaction_Amount_Ratio', 'Transaction_Frequency',
            'Merchant_Risk_Score', 'Customer_Risk_Score', 'Account_Age_Months',
            'Device_Trust_Score', 'Location_Risk_Score', 'Transaction_Velocity',
            'Cardholder_Behavior_Score', 'Fraud_Risk_Index', 'Amount']

ACCENT = "#7C4DFF"
ACCENT2 = "#00E5FF"
DANGER = "#FF3B6B"
SAFE = "#00E88F"
PLOT_BG = "rgba(0,0,0,0)"
FONT_COLOR = "#EAEDF7"

# ──────────────────────────────────────────────────────────────────────────
# CSS — the whole "post-worthy" look
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
    --accent: {ACCENT};
    --accent2: {ACCENT2};
    --danger: {DANGER};
    --safe: {SAFE};
}}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background: radial-gradient(ellipse 120% 80% at 15% -10%, rgba(124,77,255,0.16) 0%, rgba(5,7,13,0) 55%),
                radial-gradient(ellipse 100% 70% at 100% 0%, rgba(0,229,255,0.10) 0%, rgba(5,7,13,0) 55%),
                #05070D;
}}
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }}

/* Hero */
.hero-wrap {{
    padding: 34px 40px; border-radius: 22px; margin-bottom: 28px;
    background: linear-gradient(135deg, rgba(124,77,255,0.16), rgba(0,229,255,0.06));
    border: 1px solid rgba(124,77,255,0.35);
    position: relative; overflow: hidden;
}}
.hero-wrap::after {{
    content:""; position:absolute; top:-60%; right:-10%; width:380px; height:380px; border-radius:50%;
    background: radial-gradient(circle, rgba(0,229,255,0.25) 0%, rgba(0,0,0,0) 70%);
}}
.hero-title {{
    font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:2.7rem; margin:0;
    background: linear-gradient(90deg, #FFFFFF 20%, var(--accent2) 60%, var(--accent) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: -0.5px;
}}
.hero-sub {{ color:#9FA8C7; font-size:1.02rem; margin-top:10px; max-width:620px; line-height:1.55;}}
.badge-row {{ margin-top:18px; display:flex; gap:10px; flex-wrap: wrap;}}
.pill {{
    display:inline-block; padding:6px 14px; border-radius:100px; font-size:0.78rem; font-weight:600;
    background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:#C9CFE8;
}}
.pill.live {{ background: rgba(0,232,143,0.12); border-color: rgba(0,232,143,0.4); color: var(--safe);}}
.pill.live::before {{ content:"●"; margin-right:6px; color: var(--safe); animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.25}} 100%{{opacity:1}} }}

/* KPI / metric cards */
.kpi-card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px; padding: 20px 22px; height: 100%;
    transition: all .25s ease;
}}
.kpi-card:hover {{ border-color: rgba(124,77,255,0.55); transform: translateY(-3px); }}
.kpi-label {{ color:#8D95B8; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.6px;}}
.kpi-value {{ font-family:'Space Grotesk', sans-serif; font-size:2.05rem; font-weight:700; color:#F5F6FB; margin-top:6px;}}
.kpi-delta {{ font-size:0.82rem; margin-top:6px; font-weight:600;}}
.kpi-delta.up {{ color: var(--safe); }}
.kpi-delta.down {{ color: var(--danger); }}

/* Section title */
.sec-title {{
    font-family:'Space Grotesk', sans-serif; font-weight:600; font-size:1.35rem; color:#F1F3F9;
    margin: 6px 0 16px 0; display:flex; align-items:center; gap:10px;
}}
.sec-title .bar {{ width:5px; height:22px; border-radius:4px; background: linear-gradient(180deg, var(--accent), var(--accent2)); display:inline-block;}}

/* Card container */
.glass-card {{
    background: rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.09);
    border-radius: 18px; padding: 22px 24px; margin-bottom: 18px;
}}

/* Verdict boxes */
.verdict-safe, .verdict-fraud {{
    border-radius: 18px; padding: 28px; text-align:center; margin-top:10px;
}}
.verdict-safe {{ background: rgba(0,232,143,0.08); border: 1px solid rgba(0,232,143,0.45); }}
.verdict-fraud {{ background: rgba(255,59,107,0.08); border: 1px solid rgba(255,59,107,0.5); }}
.verdict-title {{ font-family:'Space Grotesk', sans-serif; font-size:1.6rem; font-weight:700; }}
.verdict-safe .verdict-title {{ color: var(--safe); }}
.verdict-fraud .verdict-title {{ color: var(--danger); }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080A13 0%, #05070D 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
.brand {{ display:flex; align-items:center; gap:10px; padding: 6px 0 22px 0; }}
.brand-icon {{ font-size:1.8rem; }}
.brand-name {{ font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.25rem; color:#F5F6FB;}}
.brand-tag {{ color:#7C86AE; font-size:0.72rem; letter-spacing: 1.5px; text-transform:uppercase; margin-top:-4px;}}

div[role="radiogroup"] > label {{
    background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 10px 14px !important; margin-bottom: 8px; width: 100%;
    transition: all .2s ease;
}}
div[role="radiogroup"] > label:hover {{ background: rgba(124,77,255,0.12); border-color: rgba(124,77,255,0.4); }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #05070D; font-weight: 700; border: none; border-radius: 12px;
    padding: 0.6rem 1.4rem; transition: all .2s ease;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ filter: brightness(1.12); transform: translateY(-1px); }}

/* Dataframe */
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow:hidden; border:1px solid rgba(255,255,255,0.08);}}

/* Slider accent */
.stSlider [data-baseweb="slider"] div div {{ background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }}

hr {{ border-color: rgba(255,255,255,0.08); }}
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-thumb {{ background: rgba(124,77,255,0.4); border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING + CLEANING  (mirrors the notebook's cleaning logic)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    df.replace(['?', 'error'], np.nan, inplace=True)
    for col in df.columns:
        if col != 'Class':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train_s, y_train)
        y_pred = model.predict(x_test_s)
        y_proba = model.predict_proba(x_test_s)[:, 1] if hasattr(model, "predict_proba") else y_pred
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
        }
    best_name = max(results, key=lambda k: results[k]["f1"])
    return results, scaler, best_name, (x_test, y_test)


def plotly_theme(fig, height=380):
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_PATH = "creditcard_data.csv"
df = load_and_clean(DEFAULT_PATH)
results, scaler, best_name, (x_test, y_test) = train_models(df)
best = results[best_name]

fraud_count = int(df['Class'].sum())
genuine_count = int((df['Class'] == 0).sum())
fraud_rate = fraud_count / len(df) * 100
amount_at_risk = df.loc[df['Class'] == 1, 'Amount'].sum()

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">Aegis</div>
            <div class="brand-tag">Fraud Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Data Explorer",
        "🤖  Model Performance",
        "🔮  Live Fraud Check",
        "📁  Batch Prediction",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#7C86AE; line-height:1.7;">
    <b style="color:#C9CFE8;">Best model</b><br>{best_name}<br><br>
    <b style="color:#C9CFE8;">Dataset</b><br>{len(df):,} transactions<br><br>
    <b style="color:#C9CFE8;">Features</b><br>{len(FEATURES)} risk signals
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# HERO (shown on every page, compact)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">Credit Card Fraud Detection</div>
    <div class="hero-sub">A real-time ML risk engine that scores every transaction across
    {len(FEATURES)} behavioral signals — flagging fraud before it costs you.</div>
    <div class="badge-row">
        <span class="pill live">Model Live</span>
        <span class="pill">🧠 {best_name}</span>
        <span class="pill">🎯 F1 {best['f1']*100:.1f}%</span>
        <span class="pill">📈 AUC {best['auc']:.3f}</span>
    </div>
</div>
""", unsafe_allow_html=True)


def kpi(col, label, value, delta=None, delta_up=True):
    delta_html = ""
    if delta is not None:
        cls = "up" if delta_up else "down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────
if page.startswith("🏠"):
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Transactions", f"{len(df):,}")
    kpi(c2, "Fraud Detected", f"{fraud_count}", f"{fraud_rate:.2f}% of volume", delta_up=False)
    kpi(c3, "Amount at Risk", f"${amount_at_risk:,.0f}")
    kpi(c4, "Model Accuracy", f"{best['accuracy']*100:.1f}%", best_name, delta_up=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown('<div class="sec-title"><span class="bar"></span>Class Balance</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=["Genuine", "Fraud"], values=[genuine_count, fraud_count],
            hole=0.62, marker=dict(colors=[ACCENT2, DANGER], line=dict(color="#05070D", width=3)),
            textinfo="percent", textfont=dict(color="#05070D", size=14, family="Space Grotesk")
        )])
        fig.add_annotation(text=f"{fraud_rate:.1f}%<br><span style='font-size:11px;color:#8D95B8'>Fraud Rate</span>",
                            showarrow=False, font=dict(size=22, color="#F5F6FB", family="Space Grotesk"))
        fig = plotly_theme(fig, height=340)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="sec-title"><span class="bar"></span>Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Amount", color=df['Class'].map({0: "Genuine", 1: "Fraud"}),
                            nbins=40, color_discrete_map={"Genuine": ACCENT2, "Fraud": DANGER},
                            barmode="overlay", opacity=0.75)
        fig.update_layout(legend_title_text="")
        fig = plotly_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title"><span class="bar"></span>Risk Signal Correlation</div>', unsafe_allow_html=True)
    corr = df[FEATURES + ['Class']].corr()
    fig = px.imshow(corr, color_continuous_scale=[[0, "#0D111C"], [0.5, ACCENT], [1, ACCENT2]],
                     aspect="auto")
    fig = plotly_theme(fig, height=480)
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────
elif page.startswith("📊"):
    st.markdown('<div class="sec-title"><span class="bar"></span>Cleaned Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=320)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-title"><span class="bar"></span>Feature Distribution Explorer</div>', unsafe_allow_html=True)
        feat = st.selectbox("Choose a feature", FEATURES, index=FEATURES.index("Fraud_Risk_Index"))
        fig = px.histogram(df, x=feat, color=df['Class'].map({0: "Genuine", 1: "Fraud"}),
                            color_discrete_map={"Genuine": ACCENT2, "Fraud": DANGER},
                            barmode="overlay", opacity=0.75, nbins=40)
        fig.update_layout(legend_title_text="")
        fig = plotly_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-title"><span class="bar"></span>Feature vs Fraud Risk</div>', unsafe_allow_html=True)
        feat2 = st.selectbox("Compare against Fraud_Risk_Index", [f for f in FEATURES if f != "Fraud_Risk_Index"],
                              index=0)
        fig = px.scatter(df, x=feat2, y="Fraud_Risk_Index", color=df['Class'].map({0: "Genuine", 1: "Fraud"}),
                          color_discrete_map={"Genuine": ACCENT2, "Fraud": DANGER}, opacity=0.7)
        fig.update_layout(legend_title_text="")
        fig = plotly_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-title"><span class="bar"></span>Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[FEATURES + ['Class']].describe().T.style.background_gradient(cmap="Purples"),
                 use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────
elif page.startswith("🤖"):
    st.markdown('<div class="sec-title"><span class="bar"></span>Model Comparison</div>', unsafe_allow_html=True)
    st.caption("Note: the bundled dataset is a small, synthetic sample built for demo purposes — "
               "metrics reflect the actual signal present in that sample. Swap in your own labeled "
               "transaction data to see real-world performance.")

    metric_df = pd.DataFrame({
        name: {"Accuracy": r["accuracy"], "Precision": r["precision"],
               "Recall": r["recall"], "F1-Score": r["f1"], "AUC": r["auc"]}
        for name, r in results.items()
    }).T.reset_index().rename(columns={"index": "Model"})

    fig = go.Figure()
    for metric, color in zip(["Accuracy", "Precision", "Recall", "F1-Score"],
                              [ACCENT2, ACCENT, "#FFD166", SAFE]):
        fig.add_trace(go.Bar(name=metric, x=metric_df["Model"], y=metric_df[metric], marker_color=color))
    fig.update_layout(barmode="group", yaxis_tickformat=".0%")
    fig = plotly_theme(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metric_df.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]})
                 .background_gradient(cmap="PuBu", subset=["F1-Score"]), use_container_width=True)

    st.markdown('<div class="sec-title"><span class="bar"></span>Select a Model to Inspect</div>', unsafe_allow_html=True)
    chosen = st.selectbox("Model", list(results.keys()), index=list(results.keys()).index(best_name))
    r = results[chosen]

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Accuracy", f"{r['accuracy']*100:.2f}%")
    kpi(c2, "Precision", f"{r['precision']*100:.2f}%")
    kpi(c3, "Recall", f"{r['recall']*100:.2f}%")
    kpi(c4, "F1-Score", f"{r['f1']*100:.2f}%")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="sec-title"><span class="bar"></span>Confusion Matrix</div>', unsafe_allow_html=True)
        cm = r["cm"]
        fig = px.imshow(cm, text_auto=True, x=["Genuine", "Fraud"], y=["Genuine", "Fraud"],
                         color_continuous_scale=[[0, "#0D111C"], [1, ACCENT]])
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        fig = plotly_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        st.markdown('<div class="sec-title"><span class="bar"></span>ROC Curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"AUC = {r['auc']:.3f}",
                                  line=dict(color=ACCENT2, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                  line=dict(color="#555b78", dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        fig = plotly_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: LIVE FRAUD CHECK
# ──────────────────────────────────────────────────────────────────────────
elif page.startswith("🔮"):
    st.markdown('<div class="sec-title"><span class="bar"></span>Score a Single Transaction</div>', unsafe_allow_html=True)
    st.caption("Adjust the risk signals below — Aegis scores the transaction instantly using the best-performing model.")

    model_name = st.selectbox("Scoring model", list(results.keys()), index=list(results.keys()).index(best_name))
    model = results[model_name]["model"]

    with st.form("live_form"):
        cols = st.columns(3)
        inputs = {}
        defaults = df[FEATURES].median()
        mins = df[FEATURES].min()
        maxs = df[FEATURES].max()
        for i, feat in enumerate(FEATURES):
            with cols[i % 3]:
                inputs[feat] = st.slider(feat.replace("_", " "), float(mins[feat]), float(maxs[feat]),
                                          float(defaults[feat]))
        submitted = st.form_submit_button("🛡️  Run Fraud Check", use_container_width=True)

    if submitted:
        X_new = pd.DataFrame([inputs])[FEATURES]
        X_new_s = scaler.transform(X_new)
        pred = model.predict(X_new_s)[0]
        proba = model.predict_proba(X_new_s)[0][1] if hasattr(model, "predict_proba") else float(pred)

        gcol, vcol = st.columns([1, 1.3])
        with gcol:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "#F5F6FB"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7C86AE"},
                    "bar": {"color": DANGER if proba > 0.5 else SAFE},
                    "bgcolor": "rgba(255,255,255,0.04)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(0,232,143,0.15)"},
                        {"range": [40, 70], "color": "rgba(255,209,102,0.15)"},
                        {"range": [70, 100], "color": "rgba(255,59,107,0.15)"},
                    ],
                },
                title={"text": "Fraud Probability", "font": {"size": 14, "color": "#8D95B8"}}
            ))
            fig = plotly_theme(fig, height=280)
            st.plotly_chart(fig, use_container_width=True)

        with vcol:
            if pred == 1:
                st.markdown(f"""
                <div class="verdict-fraud">
                    <div style="font-size:2.4rem;">🚨</div>
                    <div class="verdict-title">High Risk — Likely Fraud</div>
                    <div style="color:#C9CFE8; margin-top:8px;">Flagged by <b>{model_name}</b> with
                    {proba*100:.1f}% fraud probability. Recommend manual review or transaction hold.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-safe">
                    <div style="font-size:2.4rem;">✅</div>
                    <div class="verdict-title">Low Risk — Genuine Transaction</div>
                    <div style="color:#C9CFE8; margin-top:8px;">Cleared by <b>{model_name}</b> with only
                    {proba*100:.1f}% fraud probability. Safe to approve.</div>
                </div>
                """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# PAGE: BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────
elif page.startswith("📁"):
    st.markdown('<div class="sec-title"><span class="bar"></span>Batch Score a CSV of Transactions</div>', unsafe_allow_html=True)
    st.caption(f"Upload a CSV containing these columns: {', '.join(FEATURES)}")

    model_name = st.selectbox("Scoring model", list(results.keys()), index=list(results.keys()).index(best_name))
    model = results[model_name]["model"]

    upload = st.file_uploader("Drop your CSV here", type=["csv"])
    if upload is not None:
        try:
            new_df = load_and_clean(upload)
            missing = [f for f in FEATURES if f not in new_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                X_new = new_df[FEATURES]
                X_new_s = scaler.transform(X_new)
                preds = model.predict(X_new_s)
                probs = model.predict_proba(X_new_s)[:, 1] if hasattr(model, "predict_proba") else preds
                out = new_df.copy()
                out["Fraud_Probability"] = (probs * 100).round(2)
                out["Prediction"] = np.where(preds == 1, "🚨 Fraud", "✅ Genuine")

                c1, c2, c3 = st.columns(3)
                kpi(c1, "Rows Scored", f"{len(out):,}")
                kpi(c2, "Flagged Fraud", f"{int((preds==1).sum())}", delta_up=False)
                kpi(c3, "Avg Fraud Probability", f"{probs.mean()*100:.1f}%")

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(out.sort_values("Fraud_Probability", ascending=False),
                             use_container_width=True, height=420)

                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️  Download Scored Results", csv, "scored_transactions.csv",
                                    "text/csv", use_container_width=False)
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")
    else:
        st.info("No file uploaded yet — showing a live demo on the built-in dataset instead.")
        demo = df.sample(min(15, len(df)), random_state=1).copy()
        X_new_s = scaler.transform(demo[FEATURES])
        preds = model.predict(X_new_s)
        probs = model.predict_proba(X_new_s)[:, 1] if hasattr(model, "predict_proba") else preds
        demo["Fraud_Probability"] = (probs * 100).round(2)
        demo["Prediction"] = np.where(preds == 1, "🚨 Fraud", "✅ Genuine")
        st.dataframe(demo.sort_values("Fraud_Probability", ascending=False), use_container_width=True, height=420)

st.markdown("""
<div style="text-align:center; color:#5B6284; font-size:0.78rem; margin-top:40px;">
Built with Streamlit · scikit-learn · Plotly &nbsp;•&nbsp; Aegis Fraud Intelligence Demo
</div>
""", unsafe_allow_html=True)
