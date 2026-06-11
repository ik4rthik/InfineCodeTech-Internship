"""
pages/prediction.py
════════════════════════════════════════════════════════════════════════════════
BenefiAI – Standalone Prediction Page

Can be run independently:
    streamlit run pages/prediction.py

Or called via render() from the main app.py router.

HOW IT WORKS (step-by-step)
────────────────────────────
  Step 1 │ Load model bundle from disk using joblib.load()
          │   bundle = { model, encoders, features, metrics }
  Step 2 │ Accept 6 user inputs via Streamlit widgets
  Step 3 │ Build a one-row Pandas DataFrame from the inputs
  Step 4 │ Label-encode each categorical column using the
          │   saved LabelEncoder objects (same mapping as training)
  Step 5 │ Call model.predict_proba(X) → [P(Not Eligible), P(Eligible)]
  Step 6 │ Extract label = argmax, confidence = P(Eligible)
  Step 7 │ Render result card, confidence gauge, flag breakdown,
          │   probability bar, and per-class probability table
  Step 8 │ Log prediction to SQLite audit table
"""

import os
import sys

# Allow running standalone from the pages/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from modules.ml_model import MODEL_PATH, FEATURE_COLS, CAT_COLS
from modules.prediction import log_prediction, get_prediction_log
from datetime import datetime

# ── Design tokens (mirrored from app.py) ──────────────────────────────────────
SURFACE    = "#1E293B"
SURFACE2   = "#263348"
BORDER     = "#1E3A5F"
TEXT       = "#E2E8F0"
SUBTEXT    = "#64748B"
ELIGIBLE   = "#22C55E"
INELIGIBLE = "#EF4444"
ACCENT     = "#38BDF8"

EMP_OPTIONS  = ["Unemployed", "Part-time", "Full-time", "Self-employed"]
EDU_OPTIONS  = ["No Formal", "Primary", "Secondary", "Graduate", "Post-Graduate"]
DIS_OPTIONS  = ["No", "Yes"]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model from disk…")
def load_bundle() -> dict | None:
    """
    Load the joblib model bundle from disk.

    Returns
    -------
    dict  { model, encoders, features, metrics }   or  None if not found.

    Why @st.cache_resource?
    ───────────────────────
    cache_resource keeps the loaded object alive across reruns without
    re-reading the file each time a widget changes.  Unlike cache_data it
    does NOT serialize the object, which is correct for sklearn estimators.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5-6 – INFERENCE CORE
# ══════════════════════════════════════════════════════════════════════════════

def predict(bundle: dict, inputs: dict) -> dict:
    """
    Run prediction for one applicant.

    Pipeline
    ────────
    1. Build a 1-row DataFrame from raw inputs.
    2. Label-encode each categorical with the stored LabelEncoder.
    3. Call model.predict_proba(X)  →  [[P(Not Eligible), P(Eligible)]]
    4. Determine label and confidence.
    5. Compute vulnerability flags for transparency.

    Parameters
    ──────────
    bundle : dict   Loaded from joblib (model + encoders + features)
    inputs : dict   Raw user-supplied values:
                    age, family_income, family_members,
                    employment_status, education_level, disability_status

    Returns
    ───────
    dict {
      label         : "Eligible" | "Not Eligible"
      confidence    : float  0–1  P(Eligible)
      proba_eligible    : float  P(Eligible)
      proba_ineligible  : float  P(Not Eligible)
      flags         : dict   four vulnerability indicators
      per_capita    : float  income per family member
      timestamp     : str    ISO-8601
    }
    """
    model    = bundle["model"]
    encoders = bundle["encoders"]

    # ── Step 3: Build DataFrame ───────────────────────────────────────────────
    row = {col: [inputs[col]] for col in FEATURE_COLS}
    df  = pd.DataFrame(row)

    # ── Step 4: Label-encode categoricals ────────────────────────────────────
    for col in CAT_COLS:
        le = encoders[col]
        df[col] = le.transform(df[col])

    # ── Step 5: predict_proba ────────────────────────────────────────────────
    # Returns shape (1, 2):  col-0 = P(Not Eligible),  col-1 = P(Eligible)
    proba = model.predict_proba(df)[0]          # e.g. [0.08, 0.92]
    p_not_eligible = float(proba[0])
    p_eligible     = float(proba[1])

    # ── Step 6: Derive label ──────────────────────────────────────────────────
    label_idx = int(np.argmax(proba))           # 0 or 1
    label     = encoders["eligibility_status"].inverse_transform([label_idx])[0]

    # ── Vulnerability flags (for explainability) ──────────────────────────────
    per_capita = inputs["family_income"] / max(inputs["family_members"], 1)
    flags = {
        "low_income":      per_capita < 30_000,
        "weak_employment": inputs["employment_status"] in ("Unemployed", "Part-time"),
        "low_education":   inputs["education_level"]   in ("No Formal", "Primary"),
        "has_disability":  inputs["disability_status"] == "Yes",
    }
    flags["total"] = sum(flags[k] for k in ("low_income","weak_employment",
                                             "low_education","has_disability"))

    return {
        "label":             label,
        "confidence":        round(p_eligible, 4),
        "proba_eligible":    round(p_eligible, 4),
        "proba_ineligible":  round(p_not_eligible, 4),
        "flags":             flags,
        "per_capita":        round(per_capita, 2),
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _result_card(result: dict, name: str) -> None:
    """Render the big Eligible / Not Eligible result card."""
    label     = result["label"]
    conf      = result["confidence"] * 100
    is_elig   = label == "Eligible"
    bg        = "#14532D" if is_elig else "#450A0A"
    border    = ELIGIBLE  if is_elig else INELIGIBLE
    fg        = "#86EFAC" if is_elig else "#FCA5A5"
    icon      = "✅"      if is_elig else "❌"

    st.markdown(f"""
    <div style="background:{bg};border:2px solid {border};border-radius:20px;
                padding:2rem 2.5rem;margin:1.2rem 0 1.8rem;">
      <div style="display:flex;align-items:center;gap:1.2rem;">
        <div style="font-size:4rem;line-height:1;">{icon}</div>
        <div>
          <div style="color:{fg};font-size:2.2rem;font-weight:800;
                      letter-spacing:-0.03em;">{label}</div>
          <div style="color:{fg};opacity:.85;font-size:.95rem;margin-top:.3rem;">
            ML Confidence: <b>{conf:.1f}%</b>
            &nbsp;·&nbsp; Applicant: <b>{name or 'Anonymous'}</b>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _gauge(confidence_pct: float, is_eligible: bool) -> go.Figure:
    """Render a Plotly gauge showing P(Eligible) 0-100%."""
    bar_color = ELIGIBLE if is_eligible else INELIGIBLE
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_pct,
        delta=dict(
            reference=50,
            increasing=dict(color=ELIGIBLE),
            decreasing=dict(color=INELIGIBLE),
        ),
        title=dict(text="P(Eligible) — Confidence Score",
                   font=dict(color=TEXT, size=14)),
        number=dict(suffix="%", font=dict(color=TEXT, size=32)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0%", "25%", "50%", "75%", "100%"],
                tickcolor=SUBTEXT, tickfont=dict(color=SUBTEXT, size=11),
            ),
            bar=dict(color=bar_color, thickness=0.22),
            bgcolor="#0F172A",
            bordercolor=BORDER,
            steps=[
                dict(range=[0,  40], color="#450A0A"),
                dict(range=[40, 60], color="#713F12"),
                dict(range=[60, 100], color="#14532D"),
            ],
            threshold=dict(
                line=dict(color=bar_color, width=3),
                thickness=0.85,
                value=confidence_pct,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=SURFACE, font=dict(color=TEXT, family="Inter"),
        height=280, margin=dict(l=20, r=20, t=30, b=0),
    )
    return fig


def _prob_bar(p_eligible: float, p_ineligible: float) -> go.Figure:
    """Horizontal stacked bar showing P(Eligible) vs P(Not Eligible)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="P(Eligible)",
        x=[round(p_eligible * 100, 2)],
        y=["Probability"],
        orientation="h",
        marker_color=ELIGIBLE,
        text=[f"Eligible: {p_eligible*100:.1f}%"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#F0FDF4", size=13, family="Inter"),
    ))
    fig.add_trace(go.Bar(
        name="P(Not Eligible)",
        x=[round(p_ineligible * 100, 2)],
        y=["Probability"],
        orientation="h",
        marker_color=INELIGIBLE,
        text=[f"Not Eligible: {p_ineligible*100:.1f}%"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#FFF1F2", size=13, family="Inter"),
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Inter"),
        height=90,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis=dict(range=[0, 100], showgrid=False,
                   showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def _flag_breakdown(result: dict, inputs: dict) -> None:
    """Render the 4 vulnerability flag rows with icons and detail."""
    flags    = result["flags"]
    per_cap  = result["per_capita"]

    rows = [
        (
            "Low per-capita income (< Rs.30,000 per member)",
            flags["low_income"],
            f"Rs.{per_cap:,.0f} per member  ·  "
            f"Income Rs.{inputs['family_income']:,.0f}  /  "
            f"{inputs['family_members']} members",
        ),
        (
            "Weak employment (Unemployed or Part-time)",
            flags["weak_employment"],
            inputs["employment_status"],
        ),
        (
            "Low education (No Formal or Primary)",
            flags["low_education"],
            inputs["education_level"],
        ),
        (
            "Has a disability",
            flags["has_disability"],
            inputs["disability_status"],
        ),
    ]

    for label, fired, detail in rows:
        icon    = "🔴" if fired else "🟢"
        opacity = "1"   if fired else "0.45"
        st.markdown(
            f"<div style='opacity:{opacity};margin:.5rem 0;"
            f"padding:.5rem .8rem;border-radius:8px;"
            f"background:{SURFACE2 if fired else 'transparent'};"
            f"border-left:3px solid {INELIGIBLE if fired else BORDER};'>"
            f"{icon} &nbsp;<b style='color:{TEXT};font-size:.9rem;'>{label}</b><br>"
            f"<span style='color:{SUBTEXT};font-size:.8rem;"
            f"padding-left:1.5rem;'>{detail}</span></div>",
            unsafe_allow_html=True,
        )

    total = flags["total"]
    color  = ELIGIBLE if total >= 2 else INELIGIBLE
    st.markdown(
        f"<div style='margin-top:.8rem;padding:.6rem 1rem;border-radius:8px;"
        f"background:{SURFACE2};border:1px solid {color};'>"
        f"<b style='color:{color};'>Flags fired: {total} / 4</b>"
        f"&nbsp; — &nbsp;"
        f"<span style='color:{SUBTEXT};font-size:.87rem;'>"
        f"{'≥ 2 flags → Eligible' if total >= 2 else '< 2 flags → Not Eligible'}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def _class_table(result: dict) -> None:
    """Render per-class probability table."""
    df = pd.DataFrame({
        "Class":       ["Not Eligible", "Eligible"],
        "Probability": [
            f"{result['proba_ineligible']*100:.2f}%",
            f"{result['proba_eligible']*100:.2f}%",
        ],
        "Raw Score": [
            f"{result['proba_ineligible']:.6f}",
            f"{result['proba_eligible']:.6f}",
        ],
        "Verdict": [
            "✅ Predicted" if result["label"] == "Not Eligible" else "",
            "✅ Predicted" if result["label"] == "Eligible"     else "",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION (called by app.py)
# ══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    """Full prediction page — call this from app.py's page router."""

    st.markdown("## 🔍 &nbsp;Eligibility Prediction")
    st.markdown(
        "<p style='color:#64748B;margin-top:-0.5rem;'>"
        "Fill in the applicant's details. The trained Random Forest model will "
        "call <code>predict_proba()</code> and return the eligibility label with "
        "a confidence score.</p>",
        unsafe_allow_html=True,
    )

    # ── Step 1: Load model ────────────────────────────────────────────────────
    bundle = load_bundle()
    if bundle is None:
        st.error(
            "⚠️ **No trained model found.**  \n"
            "Go to **🤖 ML Insights** → click **🚀 Train Model** first."
        )
        st.stop()

    # Show model info banner
    m = bundle.get("metrics", {})
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:12px;
                padding:.8rem 1.2rem;margin-bottom:1.2rem;display:flex;gap:2rem;
                flex-wrap:wrap;font-size:.82rem;color:{SUBTEXT};">
      <span>🤖 <b style='color:{TEXT};'>Random Forest</b> · 200 trees</span>
      <span>🎯 Test Accuracy: <b style='color:#38BDF8;'>{m.get('accuracy',0)*100:.1f}%</b></span>
      <span>📐 Precision: <b style='color:#22C55E;'>{m.get('precision',0)*100:.1f}%</b></span>
      <span>📡 Recall: <b style='color:#818CF8;'>{m.get('recall',0)*100:.1f}%</b></span>
      <span>🏆 ROC-AUC: <b style='color:#F59E0B;'>{m.get('roc_auc',0):.4f}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 2: Input form ────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;color:#CBD5E1;"
        "border-bottom:1px solid #1E3A5F;padding-bottom:.5rem;margin-bottom:1rem;'>"
        "👤 Applicant Details</div>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            p_name    = st.text_input(
                "Full Name (optional)",
                placeholder="e.g. Anjali Mehta",
                help="Used for display only — not a model feature",
            )
            p_age     = st.number_input(
                "Age ★",
                min_value=1, max_value=120, value=30, step=1,
                help="Integer age of the applicant (1–120)",
            )
            p_income  = st.number_input(
                "Annual Family Income (INR) ★",
                min_value=0, max_value=10_000_000, value=50_000, step=1_000,
                help="Total annual household income in Indian Rupees",
            )
        with c2:
            p_members = st.number_input(
                "Family Members ★",
                min_value=1, max_value=20, value=3, step=1,
                help="Total number of people in the household",
            )
            p_emp     = st.selectbox(
                "Employment Status ★",
                EMP_OPTIONS,
                help="Current employment situation of the primary applicant",
            )
            p_edu     = st.selectbox(
                "Education Level ★",
                EDU_OPTIONS,
                help="Highest education qualification attained",
            )

        p_dis = st.radio(
            "Disability Status ★",
            DIS_OPTIONS,
            horizontal=True,
            help="Whether the applicant or any immediate family member has a disability",
        )

        st.markdown("---")
        submitted = st.form_submit_button(
            "🔮 &nbsp;Predict Eligibility",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        # ── Show feature guide before first submission ─────────────────────
        with st.expander("ℹ️ How predictions work", expanded=True):
            st.markdown("""
| Step | What happens |
|------|-------------|
| **1. Load** | `joblib.load("models/eligibility_model.pkl")` reads the saved Random Forest + LabelEncoders |
| **2. Encode** | Each categorical input is transformed using the stored `LabelEncoder` — same mapping as training |
| **3. predict_proba()** | The forest's 200 trees each vote; votes are averaged → `[P(Not Eligible), P(Eligible)]` |
| **4. Label** | `argmax(proba)` selects the winning class; confidence = `proba[1]` (P(Eligible)) |
| **5. Flags** | 4 rule-based vulnerability flags explain *why* the model leaned one way |
| **6. Log** | Every prediction is written to the SQLite `prediction_log` table for audit |
            """)
        return

    # ── Step 3-6: Run prediction ──────────────────────────────────────────────
    inputs = {
        "age":               int(p_age),
        "family_income":     float(p_income),
        "family_members":    int(p_members),
        "employment_status": p_emp,
        "education_level":   p_edu,
        "disability_status": p_dis,
    }

    with st.spinner("Running predict_proba()…"):
        result = predict(bundle, inputs)

    # ── Step 7: Render results ────────────────────────────────────────────────

    # Result card
    _result_card(result, p_name)

    # Probability stacked bar
    st.markdown(
        "<p style='color:#64748B;font-size:.8rem;margin-bottom:.3rem;'>"
        "Class probability distribution (predict_proba output)</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_prob_bar(result["proba_eligible"], result["proba_ineligible"]),
                    use_container_width=True)

    # Gauge + flags side by side
    col_gauge, col_flags = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(
            _gauge(result["confidence"] * 100, result["label"] == "Eligible"),
            use_container_width=True,
        )
    with col_flags:
        st.markdown(
            "<b style='color:#CBD5E1;font-size:.9rem;'>Vulnerability Flag Analysis</b>",
            unsafe_allow_html=True,
        )
        _flag_breakdown(result, inputs)

    # Per-class probability table
    st.markdown(
        "<div style='font-size:.9rem;font-weight:600;color:#CBD5E1;"
        "border-bottom:1px solid #1E3A5F;padding-bottom:.4rem;margin:.8rem 0;'>"
        "📊 Per-class Probability Table (predict_proba raw output)</div>",
        unsafe_allow_html=True,
    )
    _class_table(result)

    # ── Step 8: Log to SQLite ─────────────────────────────────────────────────
    applicant_for_log = {**inputs, "applicant_name": p_name or "Anonymous"}
    log_prediction(applicant_for_log, {
        "label":      result["label"],
        "confidence": result["confidence"],
        "flags":      result["flags"],
        "timestamp":  result["timestamp"],
    })
    st.caption(
        f"✔ Prediction logged at {result['timestamp']}  ·  "
        "View full audit log in **🤖 ML Insights**."
    )

    # ── Recent prediction history (this session) ──────────────────────────────
    with st.expander("📜 Prediction History (last 5)", expanded=False):
        log_df = get_prediction_log()
        if not log_df.empty:
            st.dataframe(
                log_df[["id","applicant_name","predicted_label",
                         "confidence","flag_count","predicted_at"]].head(5),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT (streamlit run pages/prediction.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or True:
    # When run standalone, set up the page config and CSS
    try:
        st.set_page_config(
            page_title="BenefiAI – Eligibility Prediction",
            page_icon="🔍",
            layout="wide",
        )
    except Exception:
        pass   # already configured when called from app.py

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg,#060D1A 0%,#0F172A 55%,#131E30 100%);
        color:#E2E8F0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#111827 0%,#0F172A 100%);
        border-right:1px solid #1E3A5F;
    }
    </style>
    """, unsafe_allow_html=True)

    render()
