"""
app.py
══════════════════════════════════════════════════════════════════════════════
BenefiAI – NGO Beneficiary Eligibility Analyzer
Integrated Streamlit Application  |  v1.0.0

Pages
─────
  1. Dashboard            – KPI cards + analytics charts
  2. Beneficiary Mgmt     – Full CRUD (View / Add / Update / Delete tabs)
  3. Model Training       – Train RF, view metrics + confusion matrix
  4. Eligibility Predict  – Live predict_proba() with confidence UI

Run:  streamlit run app.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(__file__))

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ── project modules ───────────────────────────────────────────────────────────
from database.db_setup  import initialize_database
from database.crud      import (fetch_all, fetch_by_id, insert,
                                 update, delete, get_summary_stats)
from modules.visualization import (eligibility_pie, income_distribution,
                                    education_breakdown, employment_breakdown,
                                    age_distribution)
from modules.ml_model   import (run_full_pipeline, load_model, model_exists,
                                 get_feature_importance, MODEL_PATH,
                                 FEATURE_COLS, CAT_COLS)
from modules.prediction import (log_prediction, get_prediction_log,
                                 refresh_bundle)

# ── Bootstrap DB on startup ───────────────────────────────────────────────────
initialize_database()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "BenefiAI – NGO Eligibility Analyzer",
    page_icon  = "🌱",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #060D1A 0%, #0F172A 55%, #131E30 100%);
    color: #E2E8F0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1526 0%, #0A1020 100%);
    border-right: 1px solid #1E3A5F;
}

/* ── KPI grid ── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
.kpi-card {
    background: linear-gradient(135deg,#1E293B 0%,#162032 100%);
    border: 1px solid #1E3A5F; border-radius:16px;
    padding: 1.3rem 1.5rem; position:relative; overflow:hidden;
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(56,189,248,.12); }
.kpi-icon  { font-size:1.8rem; margin-bottom:.4rem; }
.kpi-label { font-size:.7rem; font-weight:600; color:#64748B;
             text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { font-size:2rem; font-weight:800; color:#F1F5F9; line-height:1.1; }
.kpi-sub   { font-size:.75rem; margin-top:.25rem; }
.green { color:#22C55E; } .red { color:#EF4444; } .blue { color:#38BDF8; }

/* ── Section titles ── */
.sec { font-size:1rem; font-weight:600; color:#CBD5E1;
       border-bottom:1px solid #1E3A5F; padding-bottom:.5rem; margin:1rem 0; }

/* ── Tab overrides ── */
[data-testid="stTabs"] button { color:#64748B !important; font-weight:500; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#38BDF8 !important; border-bottom-color:#38BDF8 !important; }

/* ── Buttons ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg,#2563EB,#7C3AED) !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
}
/* ── Sidebar brand ── */
.brand { text-align:center; padding:.5rem 0 1rem; }
.brand-title {
    font-size:1.5rem; font-weight:800;
    background:linear-gradient(90deg,#38BDF8,#818CF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:.3rem 0 0;
}
.brand-sub { color:#475569; font-size:.72rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EMP_OPTIONS  = ["Unemployed", "Part-time", "Full-time", "Self-employed"]
EDU_OPTIONS  = ["No Formal", "Primary", "Secondary", "Graduate", "Post-Graduate"]
DIS_OPTIONS  = ["No", "Yes"]
ELIG_OPTIONS = ["Eligible", "Not Eligible"]

# Design tokens
SURFACE   = "#1E293B"
SURFACE2  = "#263348"
BORDER    = "#1E3A5F"
TEXT      = "#E2E8F0"
SUBTEXT   = "#64748B"
ELIGIBLE  = "#22C55E"
INELIG    = "#EF4444"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Always-fresh data from SQLite (no cache, so CRUD changes appear instantly)."""
    return fetch_all()

def eligibility_heuristic(income, members, emp, edu, dis) -> str:
    flags = 0
    if (income / max(members, 1)) < 30_000: flags += 1
    if emp in ("Unemployed", "Part-time"):   flags += 1
    if edu in ("No Formal", "Primary"):      flags += 1
    if dis == "Yes":                         flags += 1
    return "Eligible" if flags >= 2 else "Not Eligible"

@st.cache_resource(show_spinner=False)
def get_bundle():
    """Load joblib model bundle (cached across reruns)."""
    return load_model()

def reload_bundle():
    get_bundle.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div class="brand">
  <div style="font-size:2.8rem">🌱</div>
  <div class="brand-title">BenefiAI</div>
  <div class="brand-sub">NGO Beneficiary Eligibility Analyzer</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()

PAGE = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "👥 Beneficiary Management",
     "🤖 Model Training", "🔍 Eligibility Prediction"],
    label_visibility="collapsed",
)

st.sidebar.divider()
db_total = get_summary_stats().get("total", 0)
model_ok = model_exists()
st.sidebar.markdown(f"""
<div style="font-size:.75rem;color:{SUBTEXT};">
  <div>📦 Records in DB: <b style="color:{TEXT};">{db_total}</b></div>
  <div style="margin-top:.3rem;">🤖 Model: <b style="color:{'#22C55E' if model_ok else '#EF4444'};">
    {'Trained ✓' if model_ok else 'Not trained'}</b></div>
  <div style="margin-top:.8rem;color:#334155;">v1.0.0 · Submission Ready</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Dashboard":

    st.markdown("## 🌱 &nbsp;BenefiAI — Beneficiary Eligibility Analyzer")
    st.markdown(f"<p style='color:{SUBTEXT};margin-top:-.5rem;'>Real-time analytics dashboard · SQLite-backed · ML-powered</p>",
                unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.warning("No records found. Add beneficiaries or run: `python data/generate_dataset.py` → `python database/db_setup.py`")
        st.stop()

    stats = get_summary_stats()
    total  = stats.get("total", 0)
    elig   = stats.get("eligible", 0)
    nelig  = stats.get("not_eligible", 0)
    avg_inc= stats.get("avg_income", 0) or 0
    pct    = round(elig / total * 100, 1) if total else 0

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-icon">👥</div>
        <div class="kpi-label">Total Applicants</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-sub blue">All registered beneficiaries</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">✅</div>
        <div class="kpi-label">Eligible</div>
        <div class="kpi-value">{elig:,}</div>
        <div class="kpi-sub green">▲ {pct}% of total</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">❌</div>
        <div class="kpi-label">Not Eligible</div>
        <div class="kpi-value">{nelig:,}</div>
        <div class="kpi-sub red">▼ {100-pct}% of total</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-label">Avg Family Income</div>
        <div class="kpi-value">₹{avg_inc:,.0f}</div>
        <div class="kpi-sub blue">Annual household income</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📊 Dataset Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.plotly_chart(eligibility_pie(df), use_container_width=True)
    with c2:
        st.plotly_chart(income_distribution(df), use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    st.plotly_chart(education_breakdown(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(employment_breakdown(df), use_container_width=True)
    with c4:
        st.plotly_chart(age_distribution(df), use_container_width=True)

    # ── Quick stats table ──────────────────────────────────────────────────────
    st.markdown('<div class="sec">📋 Recent Records (latest 8)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – BENEFICIARY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "👥 Beneficiary Management":

    st.markdown("## 👥 &nbsp;Beneficiary Management")
    st.markdown(f"<p style='color:{SUBTEXT};margin-top:-.5rem;'>Create, read, update and delete beneficiary records in SQLite.</p>",
                unsafe_allow_html=True)

    tab_view, tab_add, tab_update, tab_delete = st.tabs(
        ["📋 View All", "➕ Add", "✏️ Update", "🗑️ Delete"]
    )

    # ── TAB: VIEW ─────────────────────────────────────────────────────────────
    with tab_view:
        df = load_data()
        if df.empty:
            st.info("No records yet.")
        else:
            with st.expander("🔍 Filter Records", expanded=False):
                f1, f2, f3 = st.columns(3)
                elig_f = f1.multiselect("Eligibility", ELIG_OPTIONS, ELIG_OPTIONS)
                emp_f  = f2.multiselect("Employment",  EMP_OPTIONS,  EMP_OPTIONS)
                edu_f  = f3.multiselect("Education",   EDU_OPTIONS,  EDU_OPTIONS)
                lo, hi = int(df["family_income"].min()), int(df["family_income"].max())
                inc_r  = st.slider("Income Range (INR)", lo, hi, (lo, hi), step=1000)

            mask = (
                df["eligibility_status"].isin(elig_f) &
                df["employment_status"].isin(emp_f) &
                df["education_level"].isin(edu_f) &
                df["family_income"].between(*inc_r)
            )
            filtered = df[mask]
            st.markdown(f"<p style='color:{SUBTEXT};'>Showing <b style='color:{TEXT};'>{len(filtered)}</b> of {len(df)} records</p>",
                        unsafe_allow_html=True)

            def hl(val):
                if val == "Eligible":   return "background:#14532D;color:#86EFAC;font-weight:600"
                if val == "Not Eligible": return "background:#450A0A;color:#FCA5A5;font-weight:600"
                return ""

            st.dataframe(filtered.style.map(hl, subset=["eligibility_status"]),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download CSV",
                               filtered.to_csv(index=False).encode(),
                               "beneficiaries.csv", "text/csv")

    # ── TAB: ADD ──────────────────────────────────────────────────────────────
    with tab_add:
        st.markdown(f"<p style='color:{SUBTEXT};'>Fill in all fields. Eligibility is computed automatically.</p>",
                    unsafe_allow_html=True)
        with st.form("add_form", clear_on_submit=True):
            a1, a2 = st.columns(2)
            with a1:
                a_name = st.text_input("Full Name *", placeholder="e.g. Priya Sharma")
                a_age  = st.number_input("Age *", 1, 120, 30)
                a_emp  = st.selectbox("Employment Status *", EMP_OPTIONS)
            with a2:
                a_inc  = st.number_input("Annual Family Income (INR) *", 0, 10_000_000, 50_000, 1_000)
                a_mem  = st.number_input("Family Members *", 1, 20, 3)
                a_edu  = st.selectbox("Education Level *", EDU_OPTIONS)
            a_dis = st.radio("Disability Status *", DIS_OPTIONS, horizontal=True)
            add_btn = st.form_submit_button("💾 Save Beneficiary", type="primary", use_container_width=True)

        if add_btn:
            if not a_name.strip():
                st.error("Full name is required.")
            else:
                elig = eligibility_heuristic(a_inc, a_mem, a_emp, a_edu, a_dis)
                new_id = insert({"applicant_name": a_name.strip(), "age": int(a_age),
                                  "family_income": float(a_inc), "family_members": int(a_mem),
                                  "employment_status": a_emp, "education_level": a_edu,
                                  "disability_status": a_dis, "eligibility_status": elig})
                if elig == "Eligible":
                    st.success(f"✅ **{a_name}** saved (ID {new_id}) — **Eligible**")
                else:
                    st.warning(f"⚠️ **{a_name}** saved (ID {new_id}) — **Not Eligible**")

    # ── TAB: UPDATE ───────────────────────────────────────────────────────────
    with tab_update:
        df = load_data()
        if df.empty:
            st.info("No records to update.")
        else:
            us1, us2 = st.columns([3, 1])
            name_q = us1.text_input("Search by name", placeholder="Type to filter…", key="upd_q")
            id_pool = df[df["applicant_name"].str.contains(name_q, case=False, na=False)]["id"].tolist() \
                      if name_q else df["id"].tolist()
            sel_id = us2.selectbox("Record ID", id_pool, key="upd_id")
            rec = fetch_by_id(sel_id) if sel_id else None

            if rec:
                st.markdown(f"<p style='color:{SUBTEXT};'>Editing: <b style='color:#38BDF8;'>"
                            f"{rec['applicant_name']}</b> (ID {sel_id})</p>", unsafe_allow_html=True)
                with st.form("upd_form"):
                    u1, u2 = st.columns(2)
                    with u1:
                        u_name = st.text_input("Full Name *", rec["applicant_name"])
                        u_age  = st.number_input("Age *", 1, 120, int(rec["age"]))
                        u_emp  = st.selectbox("Employment *", EMP_OPTIONS,
                                               index=EMP_OPTIONS.index(rec["employment_status"]))
                    with u2:
                        u_inc  = st.number_input("Income (INR) *", 0, 10_000_000,
                                                  int(rec["family_income"]), 1_000)
                        u_mem  = st.number_input("Members *", 1, 20, int(rec["family_members"]))
                        u_edu  = st.selectbox("Education *", EDU_OPTIONS,
                                               index=EDU_OPTIONS.index(rec["education_level"]))
                    u_dis = st.radio("Disability *", DIS_OPTIONS,
                                     index=DIS_OPTIONS.index(rec["disability_status"]), horizontal=True)
                    upd_btn = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)

                if upd_btn:
                    if not u_name.strip():
                        st.error("Name required.")
                    else:
                        u_elig = eligibility_heuristic(u_inc, u_mem, u_emp, u_edu, u_dis)
                        ok = update(sel_id, {"applicant_name": u_name.strip(), "age": int(u_age),
                                              "family_income": float(u_inc), "family_members": int(u_mem),
                                              "employment_status": u_emp, "education_level": u_edu,
                                              "disability_status": u_dis, "eligibility_status": u_elig})
                        if ok:
                            st.success(f"✅ Record {sel_id} updated — **{u_elig}**")
                        else:
                            st.error("Update failed.")

    # ── TAB: DELETE ───────────────────────────────────────────────────────────
    with tab_delete:
        df = load_data()
        if df.empty:
            st.info("No records to delete.")
        else:
            ds1, ds2 = st.columns([3, 1])
            del_q  = ds1.text_input("Search by name", placeholder="Type to filter…", key="del_q")
            del_pool = df[df["applicant_name"].str.contains(del_q, case=False, na=False)]["id"].tolist() \
                       if del_q else df["id"].tolist()
            del_id = ds2.selectbox("Record ID", del_pool, key="del_id")
            rec = fetch_by_id(del_id) if del_id else None

            if rec:
                is_e  = rec["eligibility_status"] == "Eligible"
                b_bg  = "#14532D" if is_e else "#450A0A"
                b_tx  = "#86EFAC" if is_e else "#FCA5A5"
                st.markdown(f"""
                <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
                            padding:1.2rem 1.5rem;margin:1rem 0;">
                  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
                    <div><div style="color:{SUBTEXT};font-size:.7rem;text-transform:uppercase;">Name</div>
                         <div style="color:{TEXT};font-weight:600;">{rec['applicant_name']}</div></div>
                    <div><div style="color:{SUBTEXT};font-size:.7rem;text-transform:uppercase;">Age / Income</div>
                         <div style="color:{TEXT};font-weight:600;">{rec['age']} yrs · ₹{rec['family_income']:,.0f}</div></div>
                    <div><div style="color:{SUBTEXT};font-size:.7rem;text-transform:uppercase;">Eligibility</div>
                         <span style="background:{b_bg};color:{b_tx};padding:2px 10px;border-radius:99px;
                                      font-size:.78rem;font-weight:600;">{rec['eligibility_status']}</span></div>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.error(f"⚠️ Permanently delete **{rec['applicant_name']}** (ID {del_id})? This cannot be undone.")
                if st.button("🗑️ Confirm Delete", type="primary", key="confirm_del"):
                    if delete(del_id):
                        st.success(f"✅ Record {del_id} deleted.")
                        st.balloons()
                    else:
                        st.error("Deletion failed.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🤖 Model Training":

    st.markdown("## 🤖 &nbsp;Model Training")
    st.markdown(f"<p style='color:{SUBTEXT};margin-top:-.5rem;'>Train a Random Forest classifier on the beneficiary database. "
                "Every step is explained below.</p>", unsafe_allow_html=True)

    # ── Pipeline explanation ──────────────────────────────────────────────────
    with st.expander("📖 ML Pipeline — Step-by-Step Explanation", expanded=False):
        st.markdown("""
| Step | Function | What happens |
|------|----------|-------------|
| **1 · Load** | `load_raw_data()` | Pulls all rows from SQLite — always reflects latest CRUD changes |
| **2 · Clean** | `clean(df)` | Drops duplicates & NaN rows; strips whitespace; clamps income to [₹1k–₹1Cr] |
| **3 · Encode** | `encode(df)` | Label-encodes all categoricals with **fixed vocabularies** (train-test safe) |
| **4 · Split** | `split(X, y)` | Stratified 80/20 split — preserves Eligible / Not-Eligible ratio |
| **5 · Train** | `train(X_train, y_train)` | `RandomForestClassifier(n_estimators=200, class_weight='balanced')` |
| **6 · Evaluate** | `evaluate(...)` | Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold CV, Confusion Matrix |
| **7 · Save** | `save_model(bundle)` | `joblib.dump({model, encoders, features, metrics})` → `models/eligibility_model.pkl` |
        """)
        st.markdown("""
**Why Random Forest?**
- Handles mixed features (numeric + label-encoded categoricals) natively
- `class_weight='balanced'` auto-corrects for class imbalance
- Provides `feature_importances_` for explainability
- No feature scaling required
- 200 trees with bootstrap bagging minimises overfitting
        """)

    # ── Train button ──────────────────────────────────────────────────────────
    col_btn, col_stat = st.columns([1, 3])
    with col_btn:
        train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)
    with col_stat:
        if model_exists():
            st.success(f"✅ Model saved at `{MODEL_PATH}`")
        else:
            st.warning("No model yet — click **Train Model** to begin.")

    if train_btn:
        with st.spinner("Training RandomForestClassifier (200 trees) …"):
            result = run_full_pipeline()
        if not result["success"]:
            st.error(f"Training failed: {result.get('error')}")
            st.stop()
        reload_bundle()
        st.session_state["ml_result"] = result
        st.success("✅ Model trained and saved successfully!")

    # ── Load results (from session or disk) ───────────────────────────────────
    if "ml_result" not in st.session_state:
        b = load_model()
        if b:
            st.session_state["ml_result"] = {
                "success": True, "metrics": b["metrics"],
                "bundle": b, "cleaning_log": {},
            }

    if "ml_result" not in st.session_state:
        st.info("Click **Train Model** above to see evaluation results.")
        st.stop()

    res     = st.session_state["ml_result"]
    metrics = res["metrics"]
    bundle  = res["bundle"]

    st.divider()

    # ── Cleaning summary ──────────────────────────────────────────────────────
    cl = res.get("cleaning_log", {})
    if cl:
        st.markdown('<div class="sec">🧹 Data Cleaning Summary</div>', unsafe_allow_html=True)
        x1, x2, x3 = st.columns(3)
        x1.metric("Duplicates Removed", cl.get("duplicates_removed", 0))
        x2.metric("Null Rows Removed",  cl.get("nulls_removed", 0))
        x3.metric("Clean Records Used", cl.get("final_rows", 0))
        st.divider()

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📊 Evaluation Metrics</div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%",
              help="Overall % correct on the test split")
    m2.metric("Precision", f"{metrics['precision']*100:.2f}%",
              help="Of predicted Eligible, how many truly are? (minimises wasted aid)")
    m3.metric("Recall",    f"{metrics['recall']*100:.2f}%",
              help="Of truly Eligible, how many did we catch? (minimises missed beneficiaries)")
    m4.metric("F1 Score",  f"{metrics['f1_score']*100:.2f}%",
              help="Harmonic mean of Precision & Recall")
    m5.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}",
              help="1.0 = perfect; 0.5 = random")

    st.markdown(
        f"<p style='color:{SUBTEXT};font-size:.82rem;margin-top:.5rem;'>"
        f"Train accuracy: <b style='color:#38BDF8;'>{metrics['train_accuracy']*100:.2f}%</b> &nbsp;|&nbsp; "
        f"Test accuracy: <b style='color:#22C55E;'>{metrics['accuracy']*100:.2f}%</b> &nbsp;|&nbsp; "
        f"5-Fold CV: <b style='color:#818CF8;'>{metrics['cv_mean_accuracy']*100:.2f}% "
        f"(±{metrics['cv_std']*100:.2f}%)</b> &nbsp;|&nbsp; "
        f"Train: {metrics['train_size']} rows · Test: {metrics['test_size']} rows</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Confusion matrix + explanation ────────────────────────────────────────
    st.markdown('<div class="sec">🧮 Confusion Matrix</div>', unsafe_allow_html=True)
    cm     = np.array(metrics["confusion_matrix"])
    labels = ["Not Eligible", "Eligible"]
    ann    = [[f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
              [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]]

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Predicted: {l}" for l in labels],
        y=[f"Actual: {l}"    for l in labels],
        colorscale=[[0,"#0F172A"],[.5,"#1D4ED8"],[1,"#22C55E"]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    for i in range(2):
        for j in range(2):
            fig_cm.add_annotation(x=j, y=i, text=ann[i][j], showarrow=False,
                                   font=dict(size=18, color="#F1F5F9", family="Inter"))
    fig_cm.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          font=dict(color=TEXT, family="Inter"),
                          margin=dict(l=20,r=20,t=40,b=20), height=320,
                          xaxis=dict(side="bottom"))

    cm1, cm2 = st.columns([1,1])
    with cm1:
        st.plotly_chart(fig_cm, use_container_width=True)
    with cm2:
        st.markdown("""
**Reading the Matrix**

| Cell | Meaning |
|------|---------|
| **TN** (top-left) | Correctly rejected — Not Eligible predicted correctly |
| **FP** (top-right) | False alarm — wrongly approved (costs resources) |
| **FN** (bottom-left) | Miss — truly Eligible but rejected (**most harmful for NGO**) |
| **TP** (bottom-right) | Correctly approved — Eligible predicted correctly |

> High **Recall** → fewer FN (fewer missed beneficiaries).  
> High **Precision** → fewer FP (less wasted aid).
        """)

    st.divider()

    # ── Classification report ─────────────────────────────────────────────────
    st.markdown('<div class="sec">📄 Classification Report</div>', unsafe_allow_html=True)
    rpt = metrics["classification_report"]
    rpt_rows = []
    for cls in ["Not Eligible", "Eligible", "macro avg", "weighted avg"]:
        if cls in rpt:
            r = rpt[cls]
            rpt_rows.append({"Class": cls,
                              "Precision": f"{r['precision']:.4f}",
                              "Recall":    f"{r['recall']:.4f}",
                              "F1-Score":  f"{r['f1-score']:.4f}",
                              "Support":   int(r.get("support", 0))})
    st.dataframe(pd.DataFrame(rpt_rows), use_container_width=True, hide_index=True)
    st.divider()

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown('<div class="sec">🌲 Feature Importance (Mean Decrease in Gini Impurity)</div>',
                unsafe_allow_html=True)
    fi = get_feature_importance(bundle)
    fig_fi = go.Figure(go.Bar(
        x=fi["importance_pct"], y=fi["feature"], orientation="h",
        marker=dict(color=fi["importance_pct"],
                    colorscale=[[0,"#1D4ED8"],[.5,"#38BDF8"],[1,"#22C55E"]],
                    showscale=False),
        text=[f"{v:.1f}%" for v in fi["importance_pct"]],
        textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
    ))
    fig_fi.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          font=dict(color=TEXT, family="Inter", size=13),
                          margin=dict(l=20,r=60,t=20,b=20),
                          xaxis=dict(gridcolor="#334155", ticksuffix="%"),
                          yaxis=dict(gridcolor="#334155", autorange="reversed"),
                          height=280)
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("Higher % = more important for splitting decisions across all 200 trees.")

    # ── Prediction audit log ──────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sec">📜 Prediction Audit Log</div>', unsafe_allow_html=True)
    log_df = get_prediction_log()
    if log_df.empty:
        st.info("No predictions yet. Use the **🔍 Eligibility Prediction** page.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Log", log_df.to_csv(index=False).encode(),
                           "prediction_log.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – ELIGIBILITY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔍 Eligibility Prediction":

    st.markdown("## 🔍 &nbsp;Eligibility Prediction")
    st.markdown(f"<p style='color:{SUBTEXT};margin-top:-.5rem;'>Enter applicant details. The trained Random Forest calls "
                "<code>predict_proba()</code> and returns eligibility with a confidence score.</p>",
                unsafe_allow_html=True)

    # ── Guard: model must exist ───────────────────────────────────────────────
    if not model_exists():
        st.error("⚠️ **No model found.** Go to **🤖 Model Training** and click **Train Model** first.")
        st.stop()

    bundle = get_bundle()
    if bundle is None:
        st.error("Model file found but could not be loaded. Try retraining.")
        st.stop()

    # ── Model info strip ──────────────────────────────────────────────────────
    m = bundle.get("metrics", {})
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:12px;
                padding:.75rem 1.2rem;margin-bottom:1.2rem;display:flex;gap:2rem;
                flex-wrap:wrap;font-size:.8rem;color:{SUBTEXT};">
      <span>🤖 <b style='color:{TEXT};'>RandomForest</b> · 200 trees</span>
      <span>🎯 Test Accuracy: <b style='color:#38BDF8;'>{m.get('accuracy',0)*100:.1f}%</b></span>
      <span>📐 Precision: <b style='color:#22C55E;'>{m.get('precision',0)*100:.1f}%</b></span>
      <span>📡 Recall: <b style='color:#818CF8;'>{m.get('recall',0)*100:.1f}%</b></span>
      <span>🏆 ROC-AUC: <b style='color:#F59E0B;'>{m.get('roc_auc',0):.4f}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works (collapsed) ──────────────────────────────────────────────
    with st.expander("ℹ️ How prediction works", expanded=False):
        st.markdown("""
| Step | Code | Detail |
|------|------|--------|
| **1** | `joblib.load(MODEL_PATH)` | Load trained model + LabelEncoders from disk |
| **2** | `pd.DataFrame(inputs)` | Build a 1-row DataFrame matching training feature order |
| **3** | `le.transform(df[col])` | Encode each categorical using the **same** saved LabelEncoder |
| **4** | `model.predict_proba(X)[0]` | 200 trees vote → `[P(Not Eligible), P(Eligible)]` |
| **5** | `argmax(proba)` | Winning class; confidence = `proba[1]` |
| **6** | SQLite log | Prediction written to `prediction_log` table for audit |
        """)

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">👤 Applicant Information</div>', unsafe_allow_html=True)
    with st.form("pred_form", clear_on_submit=False):
        f1, f2 = st.columns(2)
        with f1:
            p_name = st.text_input("Full Name (for audit log)",
                                    placeholder="e.g. Anjali Mehta")
            p_age  = st.number_input("Age ★", 1, 120, 30,
                                      help="Age of the primary applicant")
            p_inc  = st.number_input("Annual Family Income (INR) ★",
                                      0, 10_000_000, 50_000, 1_000,
                                      help="Total household income per year")
        with f2:
            p_mem  = st.number_input("Family Members ★", 1, 20, 3,
                                      help="Total people in the household")
            p_emp  = st.selectbox("Employment Status ★", EMP_OPTIONS)
            p_edu  = st.selectbox("Education Level ★", EDU_OPTIONS)
        p_dis = st.radio("Disability Status ★", DIS_OPTIONS, horizontal=True)

        st.markdown("---")
        pred_btn = st.form_submit_button(
            "🔮 Predict Eligibility", type="primary", use_container_width=True
        )

    if not pred_btn:
        st.stop()

    # ── Run inference ─────────────────────────────────────────────────────────
    model    = bundle["model"]
    encoders = bundle["encoders"]

    inputs = {
        "age":               int(p_age),
        "family_income":     float(p_inc),
        "family_members":    int(p_mem),
        "employment_status": p_emp,
        "education_level":   p_edu,
        "disability_status": p_dis,
    }

    # Step 2: one-row DataFrame
    row_df = pd.DataFrame({col: [inputs[col]] for col in FEATURE_COLS})

    # Step 3: encode categoricals
    for col in CAT_COLS:
        row_df[col] = encoders[col].transform(row_df[col])

    # Step 4: predict_proba()
    with st.spinner("Running predict_proba() …"):
        proba       = model.predict_proba(row_df)[0]   # [P(Not Eligible), P(Eligible)]
    p_elig      = float(proba[1])
    p_not_elig  = float(proba[0])
    label_idx   = int(np.argmax(proba))
    label       = encoders["eligibility_status"].inverse_transform([label_idx])[0]
    is_eligible = label == "Eligible"
    timestamp   = datetime.now().isoformat(timespec="seconds")

    # Vulnerability flags
    per_cap = inputs["family_income"] / max(inputs["family_members"], 1)
    flags = {
        "low_income":      per_cap < 30_000,
        "weak_employment": p_emp in ("Unemployed", "Part-time"),
        "low_education":   p_edu in ("No Formal", "Primary"),
        "has_disability":  p_dis == "Yes",
    }
    flags["total"] = sum(v for k,v in flags.items() if k != "total")

    # ── Result card ───────────────────────────────────────────────────────────
    bg  = "#14532D" if is_eligible else "#450A0A"
    bdr = ELIGIBLE  if is_eligible else INELIG
    fg  = "#86EFAC" if is_eligible else "#FCA5A5"
    ico = "✅"      if is_eligible else "❌"

    st.markdown(f"""
    <div style="background:{bg};border:2px solid {bdr};border-radius:20px;
                padding:2rem 2.5rem;margin:1.2rem 0 1.5rem;">
      <div style="display:flex;align-items:center;gap:1.2rem;">
        <div style="font-size:4rem;">{ico}</div>
        <div>
          <div style="color:{fg};font-size:2.2rem;font-weight:800;">{label}</div>
          <div style="color:{fg};opacity:.85;font-size:.92rem;">
            Confidence: <b>{p_elig*100:.1f}%</b> &nbsp;·&nbsp;
            Applicant: <b>{p_name.strip() or 'Anonymous'}</b>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bar ───────────────────────────────────────────────────────
    st.markdown(f"<p style='color:{SUBTEXT};font-size:.78rem;margin-bottom:.2rem;'>"
                "predict_proba() output — class probability distribution</p>", unsafe_allow_html=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="P(Eligible)", x=[round(p_elig*100,2)], y=[""],
                              orientation="h", marker_color=ELIGIBLE,
                              text=[f"Eligible: {p_elig*100:.1f}%"],
                              textposition="inside", insidetextanchor="middle",
                              textfont=dict(color="#F0FDF4", size=13)))
    fig_bar.add_trace(go.Bar(name="P(Not Eligible)", x=[round(p_not_elig*100,2)], y=[""],
                              orientation="h", marker_color=INELIG,
                              text=[f"Not Eligible: {p_not_elig*100:.1f}%"],
                              textposition="inside", insidetextanchor="middle",
                              textfont=dict(color="#FFF1F2", size=13)))
    fig_bar.update_layout(barmode="stack", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                           font=dict(color=TEXT), height=80, showlegend=False,
                           margin=dict(l=0,r=0,t=5,b=0),
                           xaxis=dict(range=[0,100],showgrid=False,showticklabels=False),
                           yaxis=dict(showgrid=False,showticklabels=False))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Gauge + flags ─────────────────────────────────────────────────────────
    gc, fc = st.columns([1, 1])

    with gc:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(p_elig*100, 1),
            delta=dict(reference=50, increasing=dict(color=ELIGIBLE),
                       decreasing=dict(color=INELIG)),
            title=dict(text="P(Eligible) — Confidence Score",
                       font=dict(color=TEXT, size=13)),
            number=dict(suffix="%", font=dict(color=TEXT, size=32)),
            gauge=dict(
                axis=dict(range=[0,100],
                          tickvals=[0,25,50,75,100],
                          ticktext=["0%","25%","50%","75%","100%"],
                          tickcolor=SUBTEXT, tickfont=dict(color=SUBTEXT,size=10)),
                bar=dict(color=bdr, thickness=0.22),
                bgcolor="#0F172A", bordercolor=BORDER,
                steps=[dict(range=[0,40],  color="#450A0A"),
                       dict(range=[40,60], color="#713F12"),
                       dict(range=[60,100],color="#14532D")],
                threshold=dict(line=dict(color=bdr,width=3),
                               thickness=0.85, value=p_elig*100),
            ),
        ))
        fig_g.update_layout(paper_bgcolor=SURFACE, font=dict(color=TEXT, family="Inter"),
                             height=270, margin=dict(l=15,r=15,t=30,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

    with fc:
        st.markdown(f"<b style='color:{TEXT};font-size:.9rem;'>Vulnerability Flag Analysis</b>",
                    unsafe_allow_html=True)
        flag_defs = [
            ("Low per-capita income (< ₹30k/member)", flags["low_income"],
             f"₹{per_cap:,.0f}/member  (Income ₹{p_inc:,.0f} ÷ {p_mem} members)"),
            ("Weak employment (Unemployed / Part-time)", flags["weak_employment"], p_emp),
            ("Low education (No Formal / Primary)",      flags["low_education"],   p_edu),
            ("Has a disability",                          flags["has_disability"],  p_dis),
        ]
        for lbl, fired, detail in flag_defs:
            op  = "1"    if fired else "0.42"
            bl  = INELIG if fired else BORDER
            ico2 = "🔴"  if fired else "🟢"
            st.markdown(
                f"<div style='opacity:{op};margin:.45rem 0;padding:.4rem .7rem;"
                f"border-radius:7px;border-left:3px solid {bl};"
                f"background:{'#1a0a0a' if fired else 'transparent'};'>"
                f"{ico2} <b style='color:{TEXT};font-size:.87rem;'>{lbl}</b><br>"
                f"<span style='color:{SUBTEXT};font-size:.78rem;padding-left:1.4rem;'>{detail}</span></div>",
                unsafe_allow_html=True,
            )
        total_f = flags["total"]
        fc_col  = ELIGIBLE if total_f >= 2 else INELIG
        st.markdown(
            f"<div style='margin-top:.7rem;padding:.5rem .9rem;border-radius:8px;"
            f"background:{SURFACE2};border:1px solid {fc_col};'>"
            f"<b style='color:{fc_col};'>Flags: {total_f} / 4</b>"
            f" — <span style='color:{SUBTEXT};font-size:.84rem;'>"
            f"{'≥ 2 → Eligible' if total_f >= 2 else '< 2 → Not Eligible'}</span></div>",
            unsafe_allow_html=True,
        )

    # ── Per-class probability table ───────────────────────────────────────────
    st.markdown('<div class="sec">📊 predict_proba() — Raw Output</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Class":         ["Not Eligible",        "Eligible"],
        "Probability %": [f"{p_not_elig*100:.2f}%", f"{p_elig*100:.2f}%"],
        "Raw Score":     [f"{p_not_elig:.6f}",    f"{p_elig:.6f}"],
        "Predicted":     ["✅" if label=="Not Eligible" else "", "✅" if label=="Eligible" else ""],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # ── Step 6: Log to SQLite ─────────────────────────────────────────────────
    log_prediction(
        {**inputs, "applicant_name": p_name.strip() or "Anonymous"},
        {"label": label, "confidence": p_elig,
         "flags": flags, "timestamp": timestamp},
    )
    st.caption(f"✔ Prediction logged at {timestamp}  ·  View full audit log in **🤖 Model Training**.")
