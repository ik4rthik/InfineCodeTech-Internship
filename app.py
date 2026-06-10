"""
BenefiAI - NGO Beneficiary Eligibility Analyzer
================================================
Single-file Streamlit application integrating:
  - SQLite database (CRUD)
  - Dataset analytics & charts
  - RandomForest ML training
  - Eligibility prediction with confidence score
  - Creative UI with quotes and beneficiary info

Run:
    streamlit run app.py
"""

from __future__ import annotations
import sys
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ── path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── project imports ────────────────────────────────────────────────────────────
from database.db_setup  import initialise_db
from database.crud      import (
    get_all_beneficiaries, get_beneficiary_by_id, get_summary_stats,
    add_beneficiary, update_beneficiary, delete_beneficiary,
)
from visualization.charts import (
    plot_income_distribution, plot_eligibility_distribution,
    plot_education_distribution, plot_employment_distribution,
    plot_confusion_matrix, plot_feature_importance,
    plot_roc_curve, plot_prediction_gauge,
)
from ml.model       import model_exists, load_model
from ml.trainer     import run_pipeline
from prediction.predictor import (
    load_artifacts, build_input_vector, run_prediction,
    save_prediction_to_db, get_prediction_history,
    EMPLOYMENT_OPTIONS, EDUCATION_OPTIONS,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "BenefiAI - NGO Eligibility Analyzer",
    page_icon  = "🤝",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

initialise_db()

# ── Quotes pool ────────────────────────────────────────────────────────────────
QUOTES = [
    ("The best way to find yourself is to lose yourself in the service of others.", "Mahatma Gandhi"),
    ("No one has ever become poor by giving.", "Anne Frank"),
    ("We make a living by what we get, but we make a life by what we give.", "Winston Churchill"),
    ("Alone we can do so little; together we can do so much.", "Helen Keller"),
    ("The purpose of life is not to be happy. It is to be useful, to be honorable, to be compassionate.", "Ralph Waldo Emerson"),
    ("In a gentle way, you can shake the world.", "Mahatma Gandhi"),
    ("Service to others is the rent you pay for your room here on earth.", "Muhammad Ali"),
    ("Each time a man stands up for an ideal, he sends forth a tiny ripple of hope.", "Robert F. Kennedy"),
    ("We cannot seek achievement for ourselves and forget about progress and prosperity for our community.", "Cesar Chavez"),
    ("The smallest act of kindness is worth more than the grandest intention.", "Oscar Wilde"),
    ("Empathy is the starting point for creating a community and taking action.", "Max Carver"),
    ("When you learn, teach. When you get, give.", "Maya Angelou"),
]

# Pick a quote that changes each session (not every rerun)
if "quote_idx" not in st.session_state:
    st.session_state["quote_idx"] = random.randint(0, len(QUOTES) - 1)
_Q_TEXT, _Q_AUTHOR = QUOTES[st.session_state["quote_idx"]]

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background: #070b16 !important;
  color: #e8eaf2 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1220 0%, #0a0f1c 100%) !important;
  border-right: 1px solid rgba(0,212,170,0.10) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 16px !important;
  padding: 20px 24px !important;
  transition: transform .25s, box-shadow .25s !important;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-4px) !important;
  box-shadow: 0 12px 40px rgba(0,212,170,.15) !important;
}
[data-testid="stMetricLabel"] {
  color: #8892a4 !important; font-size:11px !important;
  font-weight:700 !important; text-transform:uppercase; letter-spacing:.09em;
}
[data-testid="stMetricValue"] {
  color: #e8eaf2 !important; font-size:2rem !important; font-weight:800 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg,#00d4aa,#00a884) !important;
  color: #080c18 !important; font-weight:700 !important;
  border: none !important; border-radius:10px !important;
  padding: 10px 26px !important;
  transition: opacity .2s, transform .2s !important;
}
.stButton > button:hover { opacity:.88 !important; transform:translateY(-2px) !important; box-shadow: 0 6px 20px rgba(0,212,170,.3) !important;}
.danger > button { background: linear-gradient(135deg,#f25a7d,#c43b5c) !important; color:#fff !important; }

/* ── Form inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 10px !important; color:#e8eaf2 !important;
  transition: border-color .2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: rgba(0,212,170,.5) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  overflow: hidden;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.04) !important;
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
  background: transparent !important;
  color: #8892a4 !important;
  font-weight: 600 !important;
  border-bottom: 2px solid transparent !important;
  padding-bottom: 8px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #00d4aa !important;
  border-bottom-color: #00d4aa !important;
}

/* ── Helpers ── */
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 18px; padding: 10px; margin-bottom: 14px;
}

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg,
    rgba(0,212,170,.13) 0%,
    rgba(124,92,191,.10) 50%,
    rgba(242,90,125,.08) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px; padding: 34px 40px; margin-bottom: 28px;
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(0,212,170,.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero h1 { font-size:28px; font-weight:900; margin:0 0 8px; letter-spacing:-.02em; }
.hero p  { color:#8892a4; font-size:15px; margin:0; line-height:1.6; }

/* ── Section header ── */
.sec {
  font-size:17px; font-weight:800; margin:28px 0 14px;
  color:#e8eaf2; display:flex; align-items:center; gap:10px;
}
.sec::after {
  content:''; flex:1; height:1px; background:rgba(255,255,255,0.07);
}

/* ── Quote card ── */
.quote-card {
  background: linear-gradient(135deg,rgba(124,92,191,.12),rgba(0,212,170,.06));
  border: 1px solid rgba(124,92,191,.25);
  border-left: 4px solid #7c5cbf;
  border-radius: 0 16px 16px 0;
  padding: 22px 26px; margin: 20px 0;
  position: relative;
}
.quote-text {
  font-size: 16px; font-style: italic; color: #c9d1e0;
  line-height: 1.7; font-weight: 400;
}
.quote-author {
  font-size: 13px; font-weight: 700; color: #7c5cbf;
  margin-top: 10px; letter-spacing: .04em;
}
.quote-mark {
  position: absolute; top: 10px; right: 18px;
  font-size: 60px; color: rgba(124,92,191,.18);
  font-family: Georgia, serif; line-height: 1;
}

/* ── Info box ── */
.info-box {
  background: rgba(56,189,248,.06);
  border: 1px solid rgba(56,189,248,.18);
  border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;
}
.info-box-title {
  font-size: 13px; font-weight: 700; color: #38bdf8;
  text-transform: uppercase; letter-spacing: .07em; margin-bottom: 8px;
}
.info-box p { color: #8892a4; font-size: 14px; line-height: 1.65; margin: 0; }

/* ── Beneficiary fact card ── */
.fact-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 18px 20px;
  transition: transform .2s, box-shadow .2s;
}
.fact-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 28px rgba(0,0,0,.3);
}
.fact-icon { font-size: 28px; margin-bottom: 10px; }
.fact-title { font-size: 13px; font-weight: 700; color: #e8eaf2; margin-bottom: 5px; }
.fact-body { font-size: 12px; color: #8892a4; line-height: 1.55; }

/* ── Pill tags ── */
.pill {
  display:inline-block; padding:4px 12px; border-radius:20px;
  font-size:12px; font-weight:600; margin:2px;
}
.pill-g { background:rgba(0,212,170,.15);  color:#00d4aa; border:1px solid rgba(0,212,170,.3); }
.pill-r { background:rgba(242,90,125,.15); color:#f25a7d; border:1px solid rgba(242,90,125,.3); }
.pill-v { background:rgba(124,92,191,.15); color:#7c5cbf; border:1px solid rgba(124,92,191,.3); }
.pill-b { background:rgba(56,189,248,.15); color:#38bdf8; border:1px solid rgba(56,189,248,.3); }
.pill-y { background:rgba(245,166,35,.15); color:#f5a623; border:1px solid rgba(245,166,35,.3); }

/* ── Step cards ── */
.step {
  background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
  border-left:3px solid #00d4aa; border-radius:10px;
  padding:14px 18px; margin-bottom:10px;
}
.step-num {
  display:inline-block; background:#00d4aa; color:#080c18;
  font-weight:800; font-size:11px; border-radius:50%;
  width:20px; height:20px; text-align:center; line-height:20px; margin-right:8px;
}

/* ── Verdict ── */
.eligible { background:rgba(0,212,170,.10); border:2px solid rgba(0,212,170,.40); border-radius:18px; padding:30px; text-align:center; }
.not-eligible { background:rgba(242,90,125,.10); border:2px solid rgba(242,90,125,.40); border-radius:18px; padding:30px; text-align:center; }

/* ── Confidence bars ── */
.bar-track { background:rgba(255,255,255,0.07); border-radius:99px; height:10px; overflow:hidden; margin:8px 0 4px; }
.bar-g { background:linear-gradient(90deg,#00a884,#00d4aa); height:100%; border-radius:99px; }
.bar-r { background:linear-gradient(90deg,#c43b5c,#f25a7d); height:100%; border-radius:99px; }

/* ── Badges ── */
.badge-high { background:rgba(0,212,170,.15); color:#00d4aa; border:1px solid rgba(0,212,170,.3); padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-mod  { background:rgba(245,166,35,.15); color:#f5a623; border:1px solid rgba(245,166,35,.3); padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-low  { background:rgba(242,90,125,.15); color:#f25a7d; border:1px solid rgba(242,90,125,.3); padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }

/* ── Scrollbar ── */
hr { border-color: rgba(255,255,255,0.08) !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.15); border-radius:3px; }

/* ── Ticker animation ── */
@keyframes ticker {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
.ticker-wrap {
  overflow: hidden; background: rgba(0,212,170,.06);
  border-top: 1px solid rgba(0,212,170,.12);
  border-bottom: 1px solid rgba(0,212,170,.12);
  padding: 10px 0; margin-bottom: 24px;
}
.ticker-content {
  display: inline-flex; gap: 60px; white-space: nowrap;
  animation: ticker 40s linear infinite;
}
.ticker-item {
  font-size: 13px; font-weight: 500; color: #8892a4;
}
.ticker-item span { color: #00d4aa; font-weight: 700; }

/* ── Sidebar nav ── */
.nav-label {
  font-size: 10px; font-weight: 700; color: #8892a4;
  text-transform: uppercase; letter-spacing: .12em;
  margin: 14px 0 6px; padding: 0 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:22px 0 16px">
      <div style="font-size:52px;line-height:1;filter:drop-shadow(0 0 20px rgba(0,212,170,.4))">🤝</div>
      <div style="font-size:22px;font-weight:900;color:#e8eaf2;margin-top:10px;
                  background:linear-gradient(90deg,#00d4aa,#7c5cbf);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">BenefiAI</div>
      <div style="font-size:9px;color:#8892a4;margin-top:4px;letter-spacing:.15em;text-transform:uppercase">
        NGO Eligibility Analyzer
      </div>
      <div style="margin-top:12px;display:flex;justify-content:center;gap:6px">
        <span style="background:rgba(0,212,170,.15);color:#00d4aa;border:1px solid rgba(0,212,170,.3);
                     padding:2px 10px;border-radius:20px;font-size:10px;font-weight:600">AI-Powered</span>
        <span style="background:rgba(124,92,191,.15);color:#7c5cbf;border:1px solid rgba(124,92,191,.3);
                     padding:2px 10px;border-radius:20px;font-size:10px;font-weight:600">v2.0</span>
      </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.07);margin:0 0 14px"/>
    """, unsafe_allow_html=True)

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav", [
            "🏠  Dashboard",
            "📋  Beneficiary Management",
            "🧠  Model Training",
            "🔮  Eligibility Prediction",
        ],
        label_visibility="collapsed",
    )

    # ── Quick stats ────────────────────────────────────────────────────────────
    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:16px 0 12px"/>', unsafe_allow_html=True)
    try:
        s  = get_summary_stats()
        ml = model_exists()
        st.markdown(f"""
        <div class="nav-label">Live Stats</div>
        <div style="padding:0 2px">
          <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                      border-radius:12px;padding:14px 16px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
              <span style="color:#8892a4;font-size:12px">Total Records</span>
              <span style="color:#e8eaf2;font-weight:800;font-size:18px">{s['total']:,}</span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:4px;overflow:hidden">
              <div style="background:linear-gradient(90deg,#00d4aa,#7c5cbf);height:100%;width:100%;border-radius:99px"></div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
            <div style="background:rgba(0,212,170,.07);border:1px solid rgba(0,212,170,.15);
                        border-radius:10px;padding:10px 12px;text-align:center">
              <div style="font-size:10px;color:#8892a4;text-transform:uppercase;letter-spacing:.06em">Eligible</div>
              <div style="font-size:20px;font-weight:800;color:#00d4aa">{s['eligible']:,}</div>
              <div style="font-size:10px;color:#00d4aa;opacity:.7">{s['eligible_pct']}%</div>
            </div>
            <div style="background:rgba(242,90,125,.07);border:1px solid rgba(242,90,125,.15);
                        border-radius:10px;padding:10px 12px;text-align:center">
              <div style="font-size:10px;color:#8892a4;text-transform:uppercase;letter-spacing:.06em">Not Eligible</div>
              <div style="font-size:20px;font-weight:800;color:#f25a7d">{s['ineligible']:,}</div>
              <div style="font-size:10px;color:#f25a7d;opacity:.7">{s['ineligible_pct']}%</div>
            </div>
          </div>
          <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                      border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center">
            <span style="color:#8892a4;font-size:12px">ML Model</span>
            <span style="color:{'#00d4aa' if ml else '#f5a623'};font-weight:700;font-size:12px">
              {'✦ Ready' if ml else '○ Untrained'}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.caption("Stats unavailable")

    # ── Sidebar quote ──────────────────────────────────────────────────────────
    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:16px 0 12px"/>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(124,92,191,.08);border:1px solid rgba(124,92,191,.18);
                border-radius:12px;padding:14px 16px">
      <div style="font-size:18px;color:#7c5cbf;margin-bottom:6px">"</div>
      <div style="font-size:12px;font-style:italic;color:#c9d1e0;line-height:1.6">{_Q_TEXT}</div>
      <div style="font-size:11px;font-weight:700;color:#7c5cbf;margin-top:8px">— {_Q_AUTHOR}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.caption("BenefiAI v2.0  ·  Streamlit + SQLite + RandomForest")


# ── Resolve page name ──────────────────────────────────────────────────────────
page = page.split("  ", 1)[-1].strip()  # strip emoji prefix


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
        <div style="font-size:42px;filter:drop-shadow(0 0 16px rgba(0,212,170,.5))">🏠</div>
        <div>
          <h1 style="margin:0;font-size:26px">Analytics Dashboard</h1>
          <p style="margin:4px 0 0;color:#8892a4;font-size:13px">
            {datetime.now().strftime("%A, %d %B %Y  ·  %I:%M %p")} &nbsp;·&nbsp; Real-time Beneficiary Intelligence
          </p>
        </div>
      </div>
      <p>
        Comprehensive overview of all NGO beneficiary applicants — track eligibility trends,
        income distribution, and demographic insights in real time.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Scrolling info ticker ──────────────────────────────────────────────────
    try:
        s_tick = get_summary_stats()
        items = [
            f"📊 Total Applicants: <span>{s_tick['total']:,}</span>",
            f"✅ Eligible: <span>{s_tick['eligible']:,} ({s_tick['eligible_pct']}%)</span>",
            f"❌ Not Eligible: <span>{s_tick['ineligible']:,} ({s_tick['ineligible_pct']}%)</span>",
            f"💰 Average Family Income: <span>₹{s_tick['avg_income']:,.0f}</span>",
            f"🤝 BenefiAI helps NGOs identify families in need",
            f"🧠 ML-Powered Predictions with <span>81.97% Accuracy</span>",
            f"📍 Powered by RandomForest + SQLite + Streamlit",
        ]
        ticker_html = " &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; ".join(
            f'<span class="ticker-item">{i}</span>' for i in items * 2
        )
        st.markdown(f"""
        <div class="ticker-wrap">
          <div class="ticker-content">{ticker_html}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    try:
        df    = get_all_beneficiaries()
        stats = get_summary_stats()
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()

    if df.empty:
        st.info("No data yet. Run:  python data/generate_dataset.py")
        st.stop()

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏷️ Total Applicants",   f"{stats['total']:,}")
    k2.metric("✅ Eligible",            f"{stats['eligible']:,}",
              delta=f"{stats['eligible_pct']}% of total")
    k3.metric("❌ Not Eligible",        f"{stats['ineligible']:,}",
              delta=f"-{stats['ineligible_pct']}%", delta_color="inverse")
    k4.metric("💰 Avg. Family Income",  f"₹{stats['avg_income']:,.0f}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Quote ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="quote-card">
      <div class="quote-mark">"</div>
      <div class="quote-text">{_Q_TEXT}</div>
      <div class="quote-author">— {_Q_AUTHOR}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📈 Analytics & Visualizations</div>', unsafe_allow_html=True)
    c_l, c_r = st.columns([1.6, 1], gap="medium")
    with c_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_income_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_eligibility_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    c2_l, c2_r = st.columns(2, gap="medium")
    with c2_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_education_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c2_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_employment_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Who is a Beneficiary? Section ──────────────────────────────────────────
    st.markdown('<div class="sec">🫂 Understanding NGO Beneficiaries</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">📖 Who is a Beneficiary?</div>
      <p>
        A <strong style="color:#38bdf8">beneficiary</strong> is any individual, family, or group that receives assistance,
        support, or services from an NGO (Non-Governmental Organization). NGOs serve some of the most
        <em>vulnerable and marginalized communities</em> — those facing poverty, unemployment, disability,
        or lack of access to basic education and healthcare. BenefiAI helps NGOs <strong style="color:#00d4aa">identify
        and prioritize</strong> the applicants who need help the most, using data-driven eligibility assessment.
      </p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    facts = [
        ("👨‍👩‍👧‍👦", "Family-Centered", "Most NGO programs evaluate need based on the entire family unit — income, dependents, and vulnerabilities are assessed together."),
        ("📉", "Income-Driven", "Low annual family income is the primary eligibility indicator. Families earning below ₹1,80,000/year are typically prioritized."),
        ("♿", "Disability Inclusion", "Persons with physical, mental, or cognitive disabilities receive preferential eligibility — recognizing systemic barriers they face."),
        ("🎓", "Education Access", "Lower education levels often correlate with fewer economic opportunities, making education level a key variable in eligibility analysis."),
    ]
    for col, (icon, title, body) in zip([f1, f2, f3, f4], facts):
        col.markdown(f"""
        <div class="fact-card">
          <div class="fact-icon">{icon}</div>
          <div class="fact-title">{title}</div>
          <div class="fact-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Eligibility Criteria Info ──────────────────────────────────────────────
    st.markdown('<div class="sec">✅ Eligibility Criteria Explained</div>', unsafe_allow_html=True)
    cr_l, cr_r = st.columns([1.1, 1], gap="large")

    with cr_l:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:16px;padding:24px 28px">
          <div style="font-size:15px;font-weight:700;color:#e8eaf2;margin-bottom:16px">
            🔍 How Eligibility is Determined
          </div>
          <div style="display:flex;flex-direction:column;gap:10px">
            <div style="display:flex;align-items:flex-start;gap:12px;padding:12px;
                        background:rgba(0,212,170,.06);border-radius:10px;border:1px solid rgba(0,212,170,.12)">
              <div style="font-size:20px">💰</div>
              <div>
                <div style="font-size:13px;font-weight:700;color:#00d4aa">Annual Family Income</div>
                <div style="font-size:12px;color:#8892a4;margin-top:3px">Families with income below ₹1,80,000/year are flagged as high priority. Income is the strongest eligibility predictor (41.8% feature importance).</div>
              </div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:12px;padding:12px;
                        background:rgba(124,92,191,.06);border-radius:10px;border:1px solid rgba(124,92,191,.12)">
              <div style="font-size:20px">💼</div>
              <div>
                <div style="font-size:13px;font-weight:700;color:#7c5cbf">Employment Status</div>
                <div style="font-size:12px;color:#8892a4;margin-top:3px">Unemployed and student applicants are more likely to qualify. Employment status contributes 23.3% to the model's decisions.</div>
              </div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:12px;padding:12px;
                        background:rgba(56,189,248,.06);border-radius:10px;border:1px solid rgba(56,189,248,.12)">
              <div style="font-size:20px">👨‍👩‍👧‍👦</div>
              <div>
                <div style="font-size:13px;font-weight:700;color:#38bdf8">Family Size</div>
                <div style="font-size:12px;color:#8892a4;margin-top:3px">Larger households (4 or more members) face greater resource strain and qualify more often. Contributes 14.7% to eligibility decisions.</div>
              </div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:12px;padding:12px;
                        background:rgba(245,166,35,.06);border-radius:10px;border:1px solid rgba(245,166,35,.12)">
              <div style="font-size:20px">♿</div>
              <div>
                <div style="font-size:13px;font-weight:700;color:#f5a623">Disability & Education</div>
                <div style="font-size:12px;color:#8892a4;margin-top:3px">Disability provides a strong eligibility boost. Education level and age round out the remaining criteria with moderate importance.</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with cr_r:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:16px;padding:24px 28px;height:100%">
          <div style="font-size:15px;font-weight:700;color:#e8eaf2;margin-bottom:16px">
            📜 NGO Impact by the Numbers
          </div>
        """, unsafe_allow_html=True)

        impact_stats = [
            ("3.3M+", "NGOs", "operate in India alone, serving millions of beneficiaries"),
            ("400M+", "People", "lifted from poverty through targeted welfare programs since 2000"),
            ("₹2.5L Cr", "Annual Spend", "by Indian government and NGOs combined on social welfare"),
            ("68%", "Success Rate", "when AI-assisted eligibility screening is used vs manual review"),
        ]
        for num, label, desc in impact_stats:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;padding:12px 0;
                        border-bottom:1px solid rgba(255,255,255,0.06)">
              <div style="min-width:70px">
                <div style="font-size:20px;font-weight:900;color:#00d4aa">{num}</div>
                <div style="font-size:10px;color:#7c5cbf;font-weight:700;text-transform:uppercase">{label}</div>
              </div>
              <div style="font-size:12px;color:#8892a4;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent records ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🕐 Recent Applicants (Last 10)</div>', unsafe_allow_html=True)
    disp = df.head(10).copy()
    disp["eligibility_status"] = disp["eligibility_status"].map({1: "✅ Eligible", 0: "❌ Not Eligible"})
    disp["disability_status"]  = disp["disability_status"].map({1: "Yes", 0: "No"})
    disp["family_income"]      = disp["family_income"].apply(lambda x: f"₹{x:,.0f}")
    disp.columns = [c.replace("_", " ").title() for c in disp.columns]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Bottom inspirational quote ─────────────────────────────────────────────
    next_q = QUOTES[(st.session_state["quote_idx"] + 1) % len(QUOTES)]
    st.markdown(f"""
    <br/>
    <div style="text-align:center;padding:30px 20px;
                background:linear-gradient(135deg,rgba(0,212,170,.06),rgba(124,92,191,.04));
                border:1px solid rgba(255,255,255,0.06);border-radius:20px;margin-top:10px">
      <div style="font-size:30px;margin-bottom:12px">💬</div>
      <div style="font-size:17px;font-style:italic;color:#c9d1e0;max-width:600px;
                  margin:0 auto;line-height:1.75">"{next_q[0]}"</div>
      <div style="font-size:13px;font-weight:700;color:#7c5cbf;margin-top:14px">— {next_q[1]}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BENEFICIARY MANAGEMENT  (CRUD)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Beneficiary Management":

    st.markdown("""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
        <div style="font-size:42px">📋</div>
        <div>
          <h1 style="margin:0;font-size:26px">Beneficiary Management</h1>
          <p style="margin:4px 0 0;color:#8892a4;font-size:13px">
            Full CRUD — View, Search, Add, Update, and Delete beneficiary records in SQLite
          </p>
        </div>
      </div>
      <p>
        Manage your beneficiary database with precision. Use the tabs below to browse the entire
        applicant registry, add new families in need, update existing profiles, or remove outdated records.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── What is beneficiary management info ───────────────────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px">
      <div style="background:rgba(0,212,170,.06);border:1px solid rgba(0,212,170,.15);
                  border-radius:14px;padding:16px 20px">
        <div style="font-size:22px;margin-bottom:8px">📁</div>
        <div style="font-size:13px;font-weight:700;color:#00d4aa;margin-bottom:4px">Centralized Registry</div>
        <div style="font-size:12px;color:#8892a4;line-height:1.5">
          All applicant records are stored in a secure SQLite database, providing a single source of truth for your NGO's beneficiary data.
        </div>
      </div>
      <div style="background:rgba(124,92,191,.06);border:1px solid rgba(124,92,191,.15);
                  border-radius:14px;padding:16px 20px">
        <div style="font-size:22px;margin-bottom:8px">🔍</div>
        <div style="font-size:13px;font-weight:700;color:#7c5cbf;margin-bottom:4px">Smart Filtering</div>
        <div style="font-size:12px;color:#8892a4;line-height:1.5">
          Filter applicants by eligibility, employment, education level, or search by name to quickly find the records you need.
        </div>
      </div>
      <div style="background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.15);
                  border-radius:14px;padding:16px 20px">
        <div style="font-size:22px;margin-bottom:8px">📤</div>
        <div style="font-size:13px;font-weight:700;color:#38bdf8;margin-bottom:4px">Export & Audit</div>
        <div style="font-size:12px;color:#8892a4;line-height:1.5">
          Download filtered records as CSV for offline reporting, government submission, or external audit purposes.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_view, tab_add, tab_update, tab_delete = st.tabs([
        "📂  View Records", "➕  Add Beneficiary", "✏️  Update Beneficiary", "🗑️  Delete Beneficiary"
    ])

    # ── Tab: View ─────────────────────────────────────────────────────────────
    with tab_view:
        try:
            df = get_all_beneficiaries()
        except Exception as e:
            st.error(f"Database error: {e}"); st.stop()

        if df.empty:
            st.info("No records found."); st.stop()

        with st.expander("🔍 Filter Records", expanded=True):
            f1, f2, f3, f4 = st.columns(4)
            q_name = f1.text_input("Name contains", placeholder="e.g. Aryan")
            q_elig = f2.selectbox("Eligibility",  ["All", "Eligible", "Not Eligible"])
            q_emp  = f3.selectbox("Employment",   ["All"] + EMPLOYMENT_OPTIONS)
            q_edu  = f4.selectbox("Education",    ["All"] + EDUCATION_OPTIONS)

        filt = df.copy()
        if q_name: filt = filt[filt["applicant_name"].str.contains(q_name, case=False, na=False)]
        if q_elig == "Eligible":     filt = filt[filt["eligibility_status"] == 1]
        elif q_elig == "Not Eligible": filt = filt[filt["eligibility_status"] == 0]
        if q_emp  != "All": filt = filt[filt["employment_status"] == q_emp]
        if q_edu  != "All": filt = filt[filt["education_level"]   == q_edu]

        m1, m2, m3 = st.columns(3)
        m1.metric("📄 Showing", f"{len(filt):,} records")
        m2.metric("✅ Eligible in view", f"{int(filt['eligibility_status'].sum()):,}")
        m3.metric("💰 Avg Income",
                  f"₹{filt['family_income'].mean():,.0f}" if not filt.empty else "-")

        disp = filt.copy()
        disp["eligibility_status"] = disp["eligibility_status"].map({1: "✅ Eligible", 0: "❌ Not Eligible"})
        disp["disability_status"]  = disp["disability_status"].map({1: "Yes", 0: "No"})
        disp["family_income"]      = disp["family_income"].apply(lambda x: f"₹{x:,.0f}")
        disp.columns = [c.replace("_", " ").title() for c in disp.columns]
        st.dataframe(disp, use_container_width=True, hide_index=True, height=460)
        st.download_button("⬇️ Export as CSV", filt.to_csv(index=False).encode(), "records.csv", "text/csv")

    # ── Tab: Add ──────────────────────────────────────────────────────────────
    with tab_add:
        st.markdown("""
        <div style="background:rgba(0,212,170,.06);border:1px solid rgba(0,212,170,.15);
                    border-radius:12px;padding:14px 18px;margin-bottom:20px;font-size:13px;color:#8892a4">
          <strong style="color:#00d4aa">ℹ️ Adding a New Applicant</strong> — Fill in the beneficiary's
          demographic details and their known eligibility status. This information will be used both for
          record-keeping and future ML model training.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 👤 New Applicant Details")
        with st.form("add_form", clear_on_submit=True):
            a1, a2, a3 = st.columns(3)
            name    = a1.text_input("Full Name *")
            age     = a2.number_input("Age *", 1, 120, 30)
            members = a3.number_input("Family Members *", 1, 30, 4)

            b1, b2 = st.columns(2)
            income = b1.number_input("Annual Family Income (₹) *", 0.0, step=1000.0, value=100000.0, format="%.0f")
            emp    = b2.selectbox("Employment Status *", EMPLOYMENT_OPTIONS)

            c1, c2 = st.columns(2)
            edu    = c1.selectbox("Education Level *", EDUCATION_OPTIONS)
            dis    = c2.selectbox("Disability Status *", ["No", "Yes"])

            d1, _ = st.columns(2)
            elig   = d1.selectbox("Eligibility Status *", ["Not Eligible", "Eligible"])

            st.markdown("<br/>", unsafe_allow_html=True)
            if st.form_submit_button("💾  Save Beneficiary", use_container_width=True):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    ok, msg = add_beneficiary(
                        name.strip(), int(age), float(income), int(members),
                        emp, edu,
                        1 if dis == "Yes" else 0,
                        1 if elig == "Eligible" else 0,
                    )
                    if ok: st.success(f"✅ {msg}"); st.balloons()
                    else:  st.error(msg)

    # ── Tab: Update ───────────────────────────────────────────────────────────
    with tab_update:
        st.markdown("""
        <div style="background:rgba(124,92,191,.06);border:1px solid rgba(124,92,191,.15);
                    border-radius:12px;padding:14px 18px;margin-bottom:20px;font-size:13px;color:#8892a4">
          <strong style="color:#7c5cbf">✏️ Updating a Record</strong> — Enter the Applicant ID to load
          their current details, make changes, and save. All edits are immediately reflected in the database.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🔎 Find Applicant by ID")
        u1, u2 = st.columns([1, 3])
        uid = u1.number_input("Applicant ID", 1, step=1, value=1, key="uid")
        if u2.button("🔍  Load Record", key="load_update"):
            rec = get_beneficiary_by_id(int(uid))
            st.session_state["upd_rec"] = rec
            if not rec: st.error(f"ID {uid} not found.")
            else:       st.success("✅ Record loaded. Edit below.")

        rec = st.session_state.get("upd_rec")
        if rec:
            st.markdown("#### ✏️ Edit Details")
            with st.form("update_form"):
                ua1, ua2, ua3 = st.columns(3)
                n_name    = ua1.text_input("Full Name *", value=rec["applicant_name"])
                n_age     = ua2.number_input("Age *", 1, 120, int(rec["age"]))
                n_members = ua3.number_input("Family Members *", 1, 30, int(rec["family_members"]))

                ub1, ub2 = st.columns(2)
                n_income = ub1.number_input("Income (₹) *", 0.0, value=float(rec["family_income"]), step=1000.0, format="%.0f")
                n_emp    = ub2.selectbox("Employment *", EMPLOYMENT_OPTIONS,
                    index=EMPLOYMENT_OPTIONS.index(rec["employment_status"])
                    if rec["employment_status"] in EMPLOYMENT_OPTIONS else 0)

                uc1, uc2 = st.columns(2)
                n_edu  = uc1.selectbox("Education *", EDUCATION_OPTIONS,
                    index=EDUCATION_OPTIONS.index(rec["education_level"])
                    if rec["education_level"] in EDUCATION_OPTIONS else 0)
                n_dis  = uc2.selectbox("Disability *", ["No", "Yes"],
                    index=int(rec["disability_status"]))

                ud1, _ = st.columns(2)
                n_elig = ud1.selectbox("Eligibility *", ["Not Eligible", "Eligible"],
                    index=int(rec["eligibility_status"]))

                st.markdown("<br/>", unsafe_allow_html=True)
                if st.form_submit_button("💾  Save Changes", use_container_width=True):
                    if not n_name.strip(): st.error("Name required.")
                    else:
                        ok, msg = update_beneficiary(
                            int(rec["id"]), n_name.strip(), int(n_age), float(n_income),
                            int(n_members), n_emp, n_edu,
                            1 if n_dis == "Yes" else 0,
                            1 if n_elig == "Eligible" else 0,
                        )
                        if ok: st.success(f"✅ {msg}"); st.session_state["upd_rec"] = None
                        else:  st.error(msg)

    # ── Tab: Delete ───────────────────────────────────────────────────────────
    with tab_delete:
        st.markdown("""
        <div style="background:rgba(242,90,125,.06);border:1px solid rgba(242,90,125,.15);
                    border-radius:12px;padding:14px 18px;margin-bottom:20px;font-size:13px;color:#8892a4">
          <strong style="color:#f25a7d">⚠️ Deleting a Record</strong> — This action is permanent and cannot
          be undone. You must confirm your intent by checking the confirmation box before deleting.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### 🔎 Find Applicant to Delete")
        d1, d2 = st.columns([1, 3])
        did = d1.number_input("Applicant ID", 1, step=1, value=1, key="did")
        if d2.button("🔍  Preview Record", key="load_del"):
            rec = get_beneficiary_by_id(int(did))
            st.session_state["del_rec"] = rec
            if not rec: st.error(f"ID {did} not found.")

        rec = st.session_state.get("del_rec")
        if rec:
            ec = "#00d4aa" if rec["eligibility_status"] else "#f25a7d"
            el = "✅ Eligible" if rec["eligibility_status"] else "❌ Not Eligible"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                        border-radius:14px;padding:22px 26px;margin:14px 0;">
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px">
                <div><div style="color:#8892a4;font-size:11px;text-transform:uppercase">ID</div>
                  <div style="color:#e8eaf2;font-size:18px;font-weight:700">#{rec['id']}</div></div>
                <div><div style="color:#8892a4;font-size:11px;text-transform:uppercase">Name</div>
                  <div style="color:#e8eaf2;font-size:18px;font-weight:700">{rec['applicant_name']}</div></div>
                <div><div style="color:#8892a4;font-size:11px;text-transform:uppercase">Income</div>
                  <div style="color:#e8eaf2;font-size:18px;font-weight:700">₹{rec['family_income']:,.0f}</div></div>
                <div><div style="color:#8892a4;font-size:11px;text-transform:uppercase">Eligibility</div>
                  <div style="color:{ec};font-size:18px;font-weight:700">{el}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.warning(f"⚠️ This will permanently delete **{rec['applicant_name']}** (ID #{rec['id']}).")
            if st.checkbox("✅ I confirm — permanently delete this record"):
                st.markdown('<div class="danger">', unsafe_allow_html=True)
                if st.button(f"🗑️  Delete Record #{rec['id']}", key="exec_del"):
                    ok, msg = delete_beneficiary(int(rec["id"]))
                    if ok: st.success(f"✅ {msg}"); st.session_state["del_rec"] = None
                    else:  st.error(msg)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Training":

    st.markdown("""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
        <div style="font-size:42px;filter:drop-shadow(0 0 16px rgba(124,92,191,.5))">🧠</div>
        <div>
          <h1 style="margin:0;font-size:26px">Machine Learning Training</h1>
          <p style="margin:4px 0 0;color:#8892a4;font-size:13px">
            RandomForestClassifier · 200 Trees · 80/20 Stratified Split · Balanced Weights
          </p>
        </div>
      </div>
      <p>
        Train a RandomForestClassifier on the beneficiary dataset stored in SQLite. The pipeline
        covers data preprocessing, encoding, model fitting, and full evaluation — all in one click.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── ML Explainer ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px">
      <div class="info-box">
        <div class="info-box-title">🌳 Why RandomForest?</div>
        <p>Random forests aggregate hundreds of decision trees, each trained on a random subset
        of data, to produce a robust consensus. This ensemble approach dramatically reduces
        overfitting while maintaining high accuracy on tabular data.</p>
      </div>
      <div class="info-box">
        <div class="info-box-title">⚖️ Balanced Class Weights</div>
        <p>Since eligible applicants outnumber ineligible ones (~64/36 split), the model uses
        <code>class_weight='balanced'</code> to prevent bias toward the majority class —
        ensuring fairness in predictions for both groups.</p>
      </div>
      <div class="info-box">
        <div class="info-box-title">🔒 Joblib Persistence</div>
        <p>After training, the fitted model and label encoders are saved using <code>joblib</code>
        — the official sklearn serialization library. These artifacts are then reloaded for
        live prediction without retraining every time.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Existing model banner ──────────────────────────────────────────────────
    if model_exists():
        try:
            _, _, meta = load_model()
            m = meta.get("metrics", {})
            st.markdown(f"""
            <div style="background:rgba(0,212,170,.07);border:1px solid rgba(0,212,170,.2);
                        border-radius:14px;padding:16px 22px;margin-bottom:20px">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                <span style="font-size:20px">✅</span>
                <span style="font-weight:700;color:#00d4aa">Model Trained & Ready</span>
                <span style="font-size:12px;color:#8892a4;margin-left:auto">Trained: {meta.get('trained_at','?')}</span>
              </div>
              <div style="display:flex;gap:20px;flex-wrap:wrap">
                {''.join(f'<div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px 14px;text-align:center"><div style="font-size:11px;color:#8892a4;text-transform:uppercase">{k}</div><div style="font-size:18px;font-weight:800;color:#e8eaf2">{v}</div></div>'
                  for k,v in [
                    ("Accuracy", f"{m.get('accuracy',0)*100:.2f}%"),
                    ("Precision", f"{m.get('precision',0)*100:.2f}%"),
                    ("Recall", f"{m.get('recall',0)*100:.2f}%"),
                    ("F1 Score", f"{m.get('f1_score',0)*100:.2f}%"),
                    ("ROC-AUC", f"{m.get('roc_auc',0):.4f}"),
                  ])}
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    # ── Pipeline steps ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🔧 Training Pipeline — Step by Step</div>', unsafe_allow_html=True)
    steps = [
        ("Load Data",             "Fetch all rows from the SQLite <code>beneficiaries</code> table into a Pandas DataFrame."),
        ("Clean",                 "Strip leading/trailing whitespace, title-case categories, remove rows with invalid age or negative income."),
        ("Deduplicate",           "Drop exact-duplicate rows (across feature + target columns) to prevent the model learning the same example twice."),
        ("Impute Missing Values", "Fill numeric gaps with the column <strong>median</strong> (robust to outliers); fill categorical gaps with the <strong>mode</strong> (most frequent value)."),
        ("Label Encoding",        "Convert <code>employment_status</code> and <code>education_level</code> to integers via <code>LabelEncoder</code>. Encoders are saved alongside the model for consistent prediction-time encoding."),
        ("Train / Test Split",    "80% → training set &nbsp;|&nbsp; 20% → held-out test set. <strong>Stratified</strong> split preserves the Eligible/Not-Eligible class ratio in both halves."),
        ("Fit RandomForest",      "200 trees, <code>class_weight='balanced'</code>, <code>max_features='sqrt'</code>, <code>random_state=42</code>. Each tree votes independently; majority class wins."),
        ("Evaluate",              "Accuracy, Precision, Recall, F1, ROC-AUC on the held-out test set. 5-fold cross-validation run on the full dataset to check stability."),
        ("Save Artefacts",        "Persist <code>rf_model.pkl</code>, <code>encoders.pkl</code>, and <code>model_meta.json</code> using <code>joblib</code> (memory-efficient, numpy-optimized serialization)."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step">
          <span class="step-num">{i}</span>
          <strong style="color:#e8eaf2">{title}</strong>
          <div style="color:#8892a4;font-size:13px;margin-top:5px;padding-left:28px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Motivational quote for training ───────────────────────────────────────
    st.markdown("""
    <div class="quote-card">
      <div class="quote-mark">"</div>
      <div class="quote-text">Machine learning is the science of getting computers to act without being explicitly programmed. It is the engine of a new kind of intelligence — one that learns from data to serve humanity.</div>
      <div class="quote-author">— Andrew Ng, AI Researcher & Educator</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Train button ──────────────────────────────────────────────────────────
    btn = "🔁  Re-train Model" if model_exists() else "🚀  Train Model Now"
    if st.button(btn, key="train_btn"):
        with st.spinner("🔄 Running pipeline... this takes ~10 seconds."):
            try:
                res = run_pipeline(verbose=False)
                st.session_state["ml_res"] = res
                st.success("✅ Training complete! Full results shown below.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.exception(exc)

    # ── Results ───────────────────────────────────────────────────────────────
    res = st.session_state.get("ml_res")
    if res:
        metrics    = res["metrics"]
        train_info = res["train_info"]
        prep       = res["preprocess_report"]

        st.markdown('<div class="sec">📊 Train / Test Split</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("🏋️ Train Samples",  train_info["n_train"])
        s2.metric("🧪 Test Samples",   train_info["n_test"])
        s3.metric("📐 Features",       train_info["n_features"])
        cb = prep.get("class_balance", {})
        s4.metric("⚖️ Class Balance",  f"{cb.get('eligible',0)} E / {cb.get('not_eligible',0)} N")

        st.markdown('<div class="sec">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("✅ Accuracy",  f"{metrics['accuracy']*100:.2f}%",  help="Overall correct predictions / total")
        m2.metric("🎯 Precision", f"{metrics['precision']*100:.2f}%", help="Of predicted Eligible, how many truly were?")
        m3.metric("📡 Recall",    f"{metrics['recall']*100:.2f}%",    help="Of truly Eligible, how many did we catch?")
        m4.metric("⚖️ F1 Score",  f"{metrics['f1_score']*100:.2f}%",  help="Harmonic mean of Precision & Recall")
        m5.metric("📉 ROC-AUC",   f"{metrics['roc_auc']:.4f}",        help="1.0 = perfect, 0.5 = random baseline")

        if metrics.get("cv_mean"):
            st.info(f"🔁 5-Fold Cross-Validation: {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%  (lower variance → more stable model)")

        st.markdown('<div class="sec">🗺️ Charts</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2, gap="medium")
        with ch1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(plot_confusion_matrix(metrics["confusion_matrix"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with ch2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(metrics["feature_importance"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(plot_roc_curve(metrics["y_test"], metrics["y_pred_proba"], metrics["roc_auc"]),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 Full Classification Report"):
            st.code(metrics["classification_report"], language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ELIGIBILITY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Eligibility Prediction":

    st.markdown("""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
        <div style="font-size:42px;filter:drop-shadow(0 0 16px rgba(242,90,125,.4))">🔮</div>
        <div>
          <h1 style="margin:0;font-size:26px">Eligibility Prediction</h1>
          <p style="margin:4px 0 0;color:#8892a4;font-size:13px">
            AI-powered · predict_proba() · 200-Tree RandomForest · Confidence Scoring
          </p>
        </div>
      </div>
      <p>
        Enter an applicant's demographic profile below to receive an instant ML-powered
        eligibility verdict with probability scores, confidence rating, and actionable insights.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works explainer ────────────────────────────────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px">
      <div style="background:rgba(0,212,170,.06);border:1px solid rgba(0,212,170,.15);
                  border-radius:12px;padding:14px 16px;text-align:center">
        <div style="font-size:24px;margin-bottom:6px">📝</div>
        <div style="font-size:12px;font-weight:700;color:#00d4aa;margin-bottom:4px">1. Enter Details</div>
        <div style="font-size:11px;color:#8892a4;line-height:1.5">Fill in age, income, family size, employment, education & disability</div>
      </div>
      <div style="background:rgba(124,92,191,.06);border:1px solid rgba(124,92,191,.15);
                  border-radius:12px;padding:14px 16px;text-align:center">
        <div style="font-size:24px;margin-bottom:6px">🔢</div>
        <div style="font-size:12px;font-weight:700;color:#7c5cbf;margin-bottom:4px">2. Encode Inputs</div>
        <div style="font-size:11px;color:#8892a4;line-height:1.5">LabelEncoders convert categories to integers matching training schema</div>
      </div>
      <div style="background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.15);
                  border-radius:12px;padding:14px 16px;text-align:center">
        <div style="font-size:24px;margin-bottom:6px">🌳</div>
        <div style="font-size:12px;font-weight:700;color:#38bdf8;margin-bottom:4px">3. Forest Votes</div>
        <div style="font-size:11px;color:#8892a4;line-height:1.5">200 trees each independently vote Eligible or Not Eligible</div>
      </div>
      <div style="background:rgba(242,90,125,.06);border:1px solid rgba(242,90,125,.15);
                  border-radius:12px;padding:14px 16px;text-align:center">
        <div style="font-size:24px;margin-bottom:6px">📊</div>
        <div style="font-size:12px;font-weight:700;color:#f25a7d;margin-bottom:4px">4. Score & Verdict</div>
        <div style="font-size:11px;color:#8892a4;line-height:1.5">Vote fraction = confidence score · Majority wins the verdict</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Guard ──────────────────────────────────────────────────────────────────
    if not model_exists():
        st.warning("⚠️ No trained model found. Please go to **🧠 Model Training** and train the model first.")
        st.stop()

    # ── Load model (cached in session state) ───────────────────────────────────
    if "model" not in st.session_state:
        with st.spinner("⏳ Loading model artifacts..."):
            try:
                m, enc, meta = load_artifacts()
                st.session_state["model"]    = m
                st.session_state["encoders"] = enc
                st.session_state["meta"]     = meta
            except Exception as exc:
                st.error(f"Failed to load model: {exc}"); st.stop()

    model    = st.session_state["model"]
    encoders = st.session_state["encoders"]
    meta     = st.session_state["meta"]
    mm       = meta.get("metrics", {})

    # ── Model info banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(0,212,170,.05);border:1px solid rgba(0,212,170,.15);
                border-radius:12px;padding:12px 20px;font-size:13px;margin-bottom:20px;
                display:flex;flex-wrap:wrap;gap:20px;align-items:center">
      <span style="color:#8892a4">Trained</span>
      <strong style="color:#e8eaf2">{meta.get('trained_at','?')}</strong>
      <span style="width:1px;height:16px;background:rgba(255,255,255,.1)"></span>
      <span style="color:#8892a4">Accuracy</span>
      <strong style="color:#00d4aa">{mm.get('accuracy',0)*100:.1f}%</strong>
      <span style="width:1px;height:16px;background:rgba(255,255,255,.1)"></span>
      <span style="color:#8892a4">ROC-AUC</span>
      <strong style="color:#7c5cbf">{mm.get('roc_auc',0):.3f}</strong>
      <span style="width:1px;height:16px;background:rgba(255,255,255,.1)"></span>
      <span style="color:#8892a4">CV Accuracy</span>
      <strong style="color:#38bdf8">{mm.get('cv_mean',0)*100:.1f}% ± {mm.get('cv_std',0)*100:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)

    tab_pred, tab_hist = st.tabs(["🔮  New Prediction", "📜  Prediction History"])

    # ── Tab: Predict ───────────────────────────────────────────────────────────
    with tab_pred:
        st.markdown("#### 📋 Applicant Details")
        with st.form("predict_form", clear_on_submit=False):
            fa1, fa2, fa3 = st.columns(3)
            p_name    = fa1.text_input("Applicant Name (optional)", placeholder="e.g. Priya Sharma")
            p_age     = fa2.number_input("Age", 1, 120, 30, help="Applicant's age in years")
            p_members = fa3.number_input("Family Members", 1, 30, 4, help="Total household members")

            fb1, fb2 = st.columns(2)
            p_income = fb1.number_input("Annual Family Income (₹)", 0.0, step=5000.0,
                                         value=100000.0, format="%.0f",
                                         help="Combined gross annual income in INR")
            p_emp    = fb2.selectbox("Employment Status", EMPLOYMENT_OPTIONS,
                                      help="Employed | Unemployed | Self-Employed | Student | Retired")

            fc1, fc2 = st.columns(2)
            p_edu = fc1.selectbox("Education Level", EDUCATION_OPTIONS)
            p_dis = fc2.radio("Disability Status", ["No", "Yes"], horizontal=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮  Run Prediction", use_container_width=True)

        # ── How predict_proba works ────────────────────────────────────────────
        st.markdown("""
        <div style="background:rgba(56,189,248,.06);border-left:3px solid #38bdf8;
                    border-radius:0 12px 12px 0;padding:12px 18px;font-size:13px;
                    color:#8892a4;margin:14px 0">
          <strong style="color:#38bdf8">🔬 How predict_proba() works:</strong>
          Each of the 200 decision trees independently analyses the applicant's profile and casts a vote.
          <code>predict_proba()</code> returns the fraction of trees that voted for each class —
          that fraction is your <strong style="color:#e8eaf2">confidence score</strong>.
          A score above 75% is classified as <em>High Certainty</em>.
        </div>
        """, unsafe_allow_html=True)

        # ── Run inference ──────────────────────────────────────────────────────
        if submitted:
            raw = {
                "age":               int(p_age),
                "family_income":     float(p_income),
                "family_members":    int(p_members),
                "employment_status": p_emp,
                "education_level":   p_edu,
                "disability_status": 1 if p_dis == "Yes" else 0,
            }
            with st.spinner("🔄 Analysing applicant profile..."):
                result = run_prediction(raw, model, encoders, meta)

            st.session_state["pred_result"] = result
            st.session_state["pred_raw"]    = raw
            st.session_state["pred_name"]   = p_name

        # ── Show result ────────────────────────────────────────────────────────
        result = st.session_state.get("pred_result")
        raw    = st.session_state.get("pred_raw")
        pname  = st.session_state.get("pred_name", "")

        if result and raw:
            eligible  = result["label"] == 1
            certainty = result["certainty_band"]

            st.markdown('<div class="sec">🎯 Prediction Result</div>', unsafe_allow_html=True)

            vcol, gcol = st.columns([1, 1.3], gap="large")

            with vcol:
                if eligible:
                    pct = result["prob_eligible"]
                    st.markdown(f"""
                    <div class="eligible">
                      <div style="font-size:64px;line-height:1;margin-bottom:12px;
                                  filter:drop-shadow(0 0 16px rgba(0,212,170,.6))">✅</div>
                      <div style="font-size:30px;font-weight:900;color:#00d4aa;letter-spacing:.03em">ELIGIBLE</div>
                      <div style="color:#8892a4;font-size:14px;margin:10px 0 16px;line-height:1.5">
                        This applicant qualifies for<br/>NGO support and assistance
                      </div>
                      <div style="font-size:11px;color:#8892a4;text-transform:uppercase;letter-spacing:.07em">Confidence Score</div>
                      <div class="bar-track"><div class="bar-g" style="width:{pct*100:.1f}%"></div></div>
                      <div style="font-size:20px;font-weight:800;color:#e8eaf2">{pct*100:.1f}%</div>
                      <div style="margin-top:14px">
                        <span class="badge-{'high' if certainty=='High' else 'mod' if certainty=='Moderate' else 'low'}">
                          {certainty} Certainty
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    pct = result["prob_not_eligible"]
                    st.markdown(f"""
                    <div class="not-eligible">
                      <div style="font-size:64px;line-height:1;margin-bottom:12px;
                                  filter:drop-shadow(0 0 16px rgba(242,90,125,.6))">❌</div>
                      <div style="font-size:30px;font-weight:900;color:#f25a7d;letter-spacing:.03em">NOT ELIGIBLE</div>
                      <div style="color:#8892a4;font-size:14px;margin:10px 0 16px;line-height:1.5">
                        Does not meet the current<br/>eligibility criteria
                      </div>
                      <div style="font-size:11px;color:#8892a4;text-transform:uppercase;letter-spacing:.07em">Confidence Score</div>
                      <div class="bar-track"><div class="bar-r" style="width:{pct*100:.1f}%"></div></div>
                      <div style="font-size:20px;font-weight:800;color:#e8eaf2">{pct*100:.1f}%</div>
                      <div style="margin-top:14px">
                        <span class="badge-{'high' if certainty=='High' else 'mod' if certainty=='Moderate' else 'low'}">
                          {certainty} Certainty
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            with gcol:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(
                    plot_prediction_gauge(result["prob_eligible"]),
                    use_container_width=True, config={"displayModeBar": False}
                )
                pe  = result["prob_eligible"]     * 100
                pne = result["prob_not_eligible"]  * 100
                st.markdown(f"""
                <div style="display:flex;gap:10px;justify-content:center;margin-top:4px">
                  <div style="background:rgba(0,212,170,.10);border:1px solid rgba(0,212,170,.25);
                              border-radius:12px;padding:12px 18px;text-align:center;flex:1">
                    <div style="font-size:10px;color:#8892a4;text-transform:uppercase;letter-spacing:.06em">P(Eligible)</div>
                    <div style="font-size:24px;font-weight:900;color:#00d4aa">{pe:.1f}%</div>
                  </div>
                  <div style="background:rgba(242,90,125,.10);border:1px solid rgba(242,90,125,.25);
                              border-radius:12px;padding:12px 18px;text-align:center;flex:1">
                    <div style="font-size:10px;color:#8892a4;text-transform:uppercase;letter-spacing:.06em">P(Not Eligible)</div>
                    <div style="font-size:24px;font-weight:900;color:#f25a7d">{pne:.1f}%</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Input summary + Feature importance ─────────────────────────────
            st.markdown("<br/>", unsafe_allow_html=True)
            rc_l, rc_r = st.columns([1.1, 1], gap="large")

            with rc_l:
                st.markdown("**📝 Input Summary**")
                dis_label = "Yes" if raw["disability_status"] else "No"
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px">
                  {''.join(f'<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:10px 14px"><div style="font-size:10px;color:#8892a4;text-transform:uppercase;letter-spacing:.06em">{l}</div><div style="font-size:14px;font-weight:700;color:#e8eaf2;margin-top:3px">{v}</div></div>'
                   for l, v in [
                     ("Name", pname or "—"), ("Age", raw["age"]),
                     ("Income", f"₹{raw['family_income']:,.0f}"),
                     ("Members", raw["family_members"]),
                     ("Employment", raw["employment_status"]),
                     ("Education", raw["education_level"]),
                     ("Disability", dis_label),
                   ])}
                </div>
                """, unsafe_allow_html=True)

                # What-if hints
                if not eligible:
                    hints = []
                    if raw["family_income"] > 180_000:
                        hints.append("Income is above ₹1,80,000 — a primary disqualifier.")
                    if raw["family_members"] < 4:
                        hints.append("Larger households (4+ members) tend to qualify more often.")
                    if raw["disability_status"] == 0:
                        hints.append("Disability status is a strong eligibility driver.")
                    if hints:
                        st.markdown(
                            '<div style="background:rgba(245,166,35,.08);border:1px solid rgba(245,166,35,.2);'
                            'border-radius:12px;padding:14px 18px">'
                            '<div style="font-size:12px;font-weight:700;color:#f5a623;margin-bottom:6px">💡 Factors to Review</div>'
                            + "".join(f'<div style="font-size:13px;color:#8892a4;margin-top:4px">• {h}</div>' for h in hints)
                            + '</div>', unsafe_allow_html=True,
                        )

                # Prediction page quote ─────────────────────────────────────────
                q3 = QUOTES[(st.session_state["quote_idx"] + 2) % len(QUOTES)]
                st.markdown(f"""
                <div style="background:rgba(0,212,170,.05);border-left:3px solid #00d4aa;
                            border-radius:0 10px 10px 0;padding:12px 16px;margin-top:16px">
                  <div style="font-size:13px;font-style:italic;color:#c9d1e0;line-height:1.6">"{q3[0]}"</div>
                  <div style="font-size:11px;font-weight:700;color:#00d4aa;margin-top:6px">— {q3[1]}</div>
                </div>
                """, unsafe_allow_html=True)

            with rc_r:
                fi = result.get("feature_importance", {})
                if fi:
                    st.markdown("**📊 Feature Importance (Model-Wide)**")
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.plotly_chart(plot_feature_importance(fi),
                                    use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)

            # ── Save to DB ─────────────────────────────────────────────────────
            st.markdown("<br/>", unsafe_allow_html=True)
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                if st.button("💾  Save to Database", key="save_pred"):
                    ok, msg = save_prediction_to_db(raw, result, pname or "Anonymous")
                    if ok: st.success(f"✅ {msg}")
                    else:  st.error(msg)
            with sc2:
                st.markdown(
                    '<div style="padding:10px 0;color:#8892a4;font-size:13px">'
                    'Saves this prediction to the <code>predictions</code> SQLite table for audit trail and future review.</div>',
                    unsafe_allow_html=True
                )

    # ── Tab: History ───────────────────────────────────────────────────────────
    with tab_hist:
        try:
            hdf = get_prediction_history(limit=100)
        except Exception as e:
            st.error(f"Could not load history: {e}"); st.stop()

        if hdf.empty:
            st.info("📭 No predictions saved yet. Run a prediction and click Save to Database.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("📊 Total Predictions", len(hdf))
            h2.metric("✅ Eligible",          int((hdf["predicted_label"] == 1).sum()))
            h3.metric("❌ Not Eligible",      int((hdf["predicted_label"] == 0).sum()))
            avg = hdf["confidence_score"].mean()
            h4.metric("🎯 Avg Confidence",    f"{avg*100:.1f}%" if pd.notna(avg) else "-")

            st.markdown("<br/>", unsafe_allow_html=True)
            d = hdf.copy()
            d["predicted_label"]  = d["predicted_label"].map({1: "✅ Eligible", 0: "❌ Not Eligible"})
            d["disability_status"] = d["disability_status"].map({1: "Yes", 0: "No"})
            d["confidence_score"]  = d["confidence_score"].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
            d["family_income"]     = d["family_income"].apply(
                lambda x: f"₹{x:,.0f}" if pd.notna(x) else "-")
            d.columns = [c.replace("_", " ").title() for c in d.columns]
            st.dataframe(d, use_container_width=True, hide_index=True, height=420)
            st.download_button(
                "⬇️ Export History CSV",
                hdf.to_csv(index=False).encode(),
                "prediction_history.csv", "text/csv"
            )
