"""
pages/predict_page.py
─────────────────────
Standalone Streamlit Prediction Page for BenefiAI.

Can be imported and rendered by app.py via:
    from pages.predict_page import render
    render()

Structure of this page
───────────────────────
  Tab 1 — Predict        : input form → verdict card + gauge + confidence bar
                           + feature importance mini-chart + save to DB
  Tab 2 — History        : table of all past predictions from the DB
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prediction.predictor import (
    EDUCATION_OPTIONS,
    EMPLOYMENT_OPTIONS,
    get_prediction_history,
    load_artifacts,
    run_prediction,
    save_prediction_to_db,
)
from visualization.charts import plot_feature_importance, plot_prediction_gauge


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — injected once by the parent app, repeated here as a safety net
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
/* Verdict cards */
.verdict-eligible {
  background: linear-gradient(135deg,rgba(0,212,170,0.14),rgba(0,168,132,0.06));
  border: 2px solid rgba(0,212,170,0.45);
  border-radius: 16px; padding: 32px 28px; text-align: center;
}
.verdict-not-eligible {
  background: linear-gradient(135deg,rgba(242,90,125,0.14),rgba(180,50,80,0.06));
  border: 2px solid rgba(242,90,125,0.45);
  border-radius: 16px; padding: 32px 28px; text-align: center;
}
/* Confidence bar track */
.conf-track {
  background: rgba(255,255,255,0.07);
  border-radius: 99px;
  height: 10px;
  overflow: hidden;
  margin: 8px 0 4px;
}
.conf-fill-elig   { background:linear-gradient(90deg,#00a884,#00d4aa); height:100%; border-radius:99px; }
.conf-fill-noelig { background:linear-gradient(90deg,#c43b5c,#f25a7d); height:100%; border-radius:99px; }
/* Input summary grid */
.summary-grid {
  display: grid;
  grid-template-columns: repeat(3,1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.summary-cell {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px; padding: 10px 14px;
}
.summary-label { font-size:11px; color:#8892a4; text-transform:uppercase; letter-spacing:0.06em; }
.summary-value { font-size:15px; font-weight:600; color:#e8eaf2; margin-top:3px; }
/* Certainty badge */
.badge-high     { background:rgba(0,212,170,0.15); color:#00d4aa; border:1px solid rgba(0,212,170,0.3); padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-moderate { background:rgba(245,166,35,0.15); color:#f5a623; border:1px solid rgba(245,166,35,0.3); padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-low      { background:rgba(242,90,125,0.15); color:#f25a7d; border:1px solid rgba(242,90,125,0.3); padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
/* History table label */
.lbl-eligible   { color:#00d4aa; font-weight:600; }
.lbl-noteligible{ color:#f25a7d; font-weight:600; }
/* Form section label */
.form-section {
  font-size: 13px; font-weight: 600; color: #8892a4;
  text-transform: uppercase; letter-spacing: 0.07em;
  margin: 20px 0 8px; border-bottom: 1px solid rgba(255,255,255,0.07);
  padding-bottom: 6px;
}
/* Tip box */
.tip-box {
  background: rgba(56,189,248,0.07);
  border-left: 3px solid #38bdf8;
  border-radius: 0 10px 10px 0;
  padding: 10px 16px; font-size: 13px; color: #8892a4;
  margin-bottom: 14px;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _certainty_badge(band: str) -> str:
    css = {"High": "badge-high", "Moderate": "badge-moderate", "Low": "badge-low"}.get(band, "badge-low")
    return f'<span class="{css}">{band} Certainty</span>'


def _confidence_bar(pct: float, eligible: bool) -> str:
    fill_cls = "conf-fill-elig" if eligible else "conf-fill-noelig"
    return f"""
    <div style="font-size:12px;color:#8892a4;margin-top:6px;">Confidence Score</div>
    <div class="conf-track"><div class="{fill_cls}" style="width:{pct*100:.1f}%"></div></div>
    <div style="font-size:13px;font-weight:700;color:#e8eaf2;">{pct*100:.1f}%</div>
    """


def _input_summary_card(raw_inputs: dict, applicant_name: str) -> str:
    dis = "Yes" if raw_inputs.get("disability_status") == 1 else "No"
    cells = [
        ("Name",           applicant_name or "—"),
        ("Age",            raw_inputs.get("age", "—")),
        ("Family Income",  f"₹{raw_inputs['family_income']:,.0f}"),
        ("Family Members", raw_inputs.get("family_members", "—")),
        ("Employment",     raw_inputs.get("employment_status", "—")),
        ("Education",      raw_inputs.get("education_level", "—")),
        ("Disability",     dis),
    ]
    html = '<div class="summary-grid">'
    for label, value in cells:
        html += f"""
        <div class="summary-cell">
          <div class="summary-label">{label}</div>
          <div class="summary-value">{value}</div>
        </div>
        """
    html += "</div>"
    return html


# ══════════════════════════════════════════════════════════════════════════════
#  Main render function (called from app.py)
# ══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    """Render the full Prediction page including tabs."""
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(0,212,170,0.12),
                rgba(124,92,191,0.10),rgba(242,90,125,0.08));
                border:1px solid rgba(255,255,255,0.08);border-radius:16px;
                padding:30px 34px;margin-bottom:24px;">
      <h1 style="font-size:28px;font-weight:800;margin:0 0 6px;">🔮 Predict Eligibility</h1>
      <p style="color:#8892a4;font-size:15px;margin:0;">
        Enter an applicant's details to get an instant ML-powered eligibility
        prediction with probability scores and confidence rating.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Guard: model must be trained ───────────────────────────────────────────
    from ml.model import model_exists
    if not model_exists():
        st.warning("""
        **No trained model found.**
        Please open the **🧠 ML Training** page and click **Train Model** first.
        """)
        return

    # ── Load artefacts (cached in session state so we don't reload on every rerun)
    if "pred_model" not in st.session_state:
        with st.spinner("Loading model…"):
            try:
                m, enc, meta = load_artifacts()
                st.session_state["pred_model"]    = m
                st.session_state["pred_encoders"] = enc
                st.session_state["pred_meta"]     = meta
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                return

    model    = st.session_state["pred_model"]
    encoders = st.session_state["pred_encoders"]
    meta     = st.session_state["pred_meta"]
    m_metrics = meta.get("metrics", {})

    # ── Tiny model-info banner ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.18);
                border-radius:10px;padding:10px 18px;font-size:13px;margin-bottom:20px;
                display:flex;gap:24px;align-items:center;">
      <span style="color:#8892a4;">Model trained</span>
      <strong style="color:#e8eaf2;">{meta.get('trained_at','—')}</strong>
      <span style="color:#8892a4;">Accuracy</span>
      <strong style="color:#00d4aa;">{m_metrics.get('accuracy',0)*100:.1f}%</strong>
      <span style="color:#8892a4;">ROC-AUC</span>
      <strong style="color:#7c5cbf;">{m_metrics.get('roc_auc',0):.3f}</strong>
      <span style="color:#8892a4;">CV Accuracy</span>
      <strong style="color:#38bdf8;">{m_metrics.get('cv_mean',0)*100:.1f}% ± {m_metrics.get('cv_std',0)*100:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Two tabs ───────────────────────────────────────────────────────────────
    tab_predict, tab_history = st.tabs(["🔮  New Prediction", "📜  Prediction History"])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — Predict
    # ══════════════════════════════════════════════════════════════════════════
    with tab_predict:

        # ── Input form ─────────────────────────────────────────────────────────
        with st.form("predict_form", clear_on_submit=False):

            # Applicant name (stored to DB, not used by model)
            st.markdown('<div class="form-section">Applicant Identity</div>', unsafe_allow_html=True)
            applicant_name = st.text_input(
                "Applicant Name (optional — for record keeping)",
                placeholder="e.g. Priya Sharma",
                max_chars=120,
            )

            # Numeric inputs
            st.markdown('<div class="form-section">Demographics</div>', unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)

            age = col_a.number_input(
                "Age",
                min_value  = 1,
                max_value  = 120,
                value      = 30,
                step       = 1,
                help       = "Applicant's current age in years",
            )
            family_members = col_b.number_input(
                "Family Members",
                min_value  = 1,
                max_value  = 30,
                value      = 4,
                step       = 1,
                help       = "Total number of people living in the household",
            )
            family_income = col_c.number_input(
                "Annual Family Income (₹)",
                min_value  = 0.0,
                max_value  = 10_000_000.0,
                value      = 100_000.0,
                step       = 5_000.0,
                format     = "%.0f",
                help       = "Combined gross annual income of all family members in INR",
            )

            # Categorical inputs
            st.markdown('<div class="form-section">Background</div>', unsafe_allow_html=True)
            col_d, col_e = st.columns(2)

            employment_status = col_d.selectbox(
                "Employment Status",
                EMPLOYMENT_OPTIONS,
                help=(
                    "Employed = regular salary  |  Unemployed = no income  |  "
                    "Self-Employed = own business  |  Student = enrolled in education  |  "
                    "Retired = formerly employed"
                ),
            )
            education_level = col_e.selectbox(
                "Education Level",
                EDUCATION_OPTIONS,
                help="Highest qualification attained by the primary applicant",
            )

            # Binary toggle
            st.markdown('<div class="form-section">Disability Status</div>', unsafe_allow_html=True)
            disability_raw = st.radio(
                "Does the applicant have a disability?",
                ["No", "Yes"],
                horizontal = True,
                help       = "Any physical, mental, or cognitive disability that affects daily life",
            )
            disability_status = 1 if disability_raw == "Yes" else 0

            st.markdown("<br/>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔮  Run Prediction",
                use_container_width = True,
            )

        # ── Tip box ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="tip-box">
          <strong style="color:#38bdf8;">How the model decides:</strong>
          The RandomForest queries 200 decision trees. Each tree independently votes
          Eligible or Not Eligible. <code>predict_proba()</code> returns the fraction
          of trees that voted for each class — that fraction is the confidence score.
        </div>
        """, unsafe_allow_html=True)

        # ── Run inference ──────────────────────────────────────────────────────
        if submitted:
            raw_inputs = {
                "age":               int(age),
                "family_income":     float(family_income),
                "family_members":    int(family_members),
                "employment_status": employment_status,
                "education_level":   education_level,
                "disability_status": disability_status,
            }

            with st.spinner("Running inference…"):
                result = run_prediction(raw_inputs, model, encoders, meta)

            # Cache result for save button below the form
            st.session_state["last_result"]     = result
            st.session_state["last_raw_inputs"] = raw_inputs
            st.session_state["last_name"]       = applicant_name

        # ── Display result (persists across reruns via session_state) ──────────
        result     = st.session_state.get("last_result")
        raw_inputs = st.session_state.get("last_raw_inputs")
        saved_name = st.session_state.get("last_name", "")

        if result and raw_inputs:
            eligible  = result["label"] == 1
            certainty = result["certainty_band"]

            st.markdown(
                '<div style="font-size:20px;font-weight:700;color:#e8eaf2;'
                'margin:28px 0 16px;display:flex;align-items:center;gap:10px;">'
                '🎯 Prediction Result'
                '<span style="flex:1;height:1px;background:rgba(255,255,255,0.08);'
                'margin-left:12px;"></span></div>',
                unsafe_allow_html=True,
            )

            # ── Row 1: verdict card + gauge ────────────────────────────────────
            left, right = st.columns([1, 1.3], gap="large")

            with left:
                if eligible:
                    st.markdown(f"""
                    <div class="verdict-eligible">
                      <div style="font-size:64px;line-height:1;margin-bottom:14px;">✅</div>
                      <div style="font-size:28px;font-weight:800;color:#00d4aa;letter-spacing:0.03em;">
                        ELIGIBLE
                      </div>
                      <div style="color:#8892a4;font-size:14px;margin:8px 0 16px;">
                        This applicant qualifies for NGO support
                      </div>
                      {_confidence_bar(result['prob_eligible'], True)}
                      <div style="margin-top:14px;">
                        {_certainty_badge(certainty)}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-not-eligible">
                      <div style="font-size:64px;line-height:1;margin-bottom:14px;">❌</div>
                      <div style="font-size:28px;font-weight:800;color:#f25a7d;letter-spacing:0.03em;">
                        NOT ELIGIBLE
                      </div>
                      <div style="color:#8892a4;font-size:14px;margin:8px 0 16px;">
                        This applicant does not meet the criteria
                      </div>
                      {_confidence_bar(result['prob_not_eligible'], False)}
                      <div style="margin-top:14px;">
                        {_certainty_badge(certainty)}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            with right:
                # Gauge chart
                st.markdown('<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:8px;">', unsafe_allow_html=True)
                st.plotly_chart(
                    plot_prediction_gauge(result["prob_eligible"]),
                    use_container_width = True,
                    config={"displayModeBar": False},
                )

                # Dual probability pills
                p_elig     = result["prob_eligible"]     * 100
                p_notelig  = result["prob_not_eligible"] * 100
                st.markdown(f"""
                <div style="display:flex;gap:10px;justify-content:center;margin-top:4px;">
                  <div style="background:rgba(0,212,170,0.10);border:1px solid rgba(0,212,170,0.25);
                              border-radius:10px;padding:10px 18px;text-align:center;flex:1;">
                    <div style="font-size:11px;color:#8892a4;text-transform:uppercase;letter-spacing:0.06em;">P(Eligible)</div>
                    <div style="font-size:22px;font-weight:800;color:#00d4aa;">{p_elig:.1f}%</div>
                  </div>
                  <div style="background:rgba(242,90,125,0.10);border:1px solid rgba(242,90,125,0.25);
                              border-radius:10px;padding:10px 18px;text-align:center;flex:1;">
                    <div style="font-size:11px;color:#8892a4;text-transform:uppercase;letter-spacing:0.06em;">P(Not Eligible)</div>
                    <div style="font-size:22px;font-weight:800;color:#f25a7d;">{p_notelig:.1f}%</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # ── Row 2: input summary + feature importance ──────────────────────
            row2_l, row2_r = st.columns([1.1, 1], gap="large")

            with row2_l:
                st.markdown(
                    '<div style="font-size:14px;font-weight:600;color:#8892a4;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">'
                    'Input Summary</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_input_summary_card(raw_inputs, saved_name), unsafe_allow_html=True)

                # ── What-if hints ──────────────────────────────────────────────
                hints = []
                if not eligible:
                    if raw_inputs["family_income"] > 180_000:
                        hints.append("Income above ₹1,80,000 — reducing it may change the outcome.")
                    if raw_inputs["family_members"] < 4:
                        hints.append("Larger households (≥ 4) tend to qualify more often.")
                    if raw_inputs["disability_status"] == 0:
                        hints.append("Disability status is a strong eligibility driver.")

                if hints:
                    st.markdown(
                        '<div style="background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);'
                        'border-radius:10px;padding:12px 16px;margin-top:10px;">'
                        '<div style="font-size:12px;font-weight:600;color:#f5a623;margin-bottom:6px;">💡 Factors to Review</div>'
                        + "".join(f'<div style="font-size:13px;color:#8892a4;margin-top:4px;">• {h}</div>' for h in hints)
                        + '</div>',
                        unsafe_allow_html=True,
                    )

            with row2_r:
                fi = result.get("feature_importance", {})
                if fi:
                    st.markdown(
                        '<div style="font-size:14px;font-weight:600;color:#8892a4;'
                        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">'
                        'Feature Influence</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:8px;">', unsafe_allow_html=True)
                    st.plotly_chart(
                        plot_feature_importance(fi),
                        use_container_width = True,
                        config={"displayModeBar": False},
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

            # ── Save to DB ─────────────────────────────────────────────────────
            st.markdown("<br/>", unsafe_allow_html=True)
            scol1, scol2 = st.columns([1, 3])
            with scol1:
                if st.button("💾  Save to Database", key="save_pred_btn"):
                    ok, msg = save_prediction_to_db(
                        raw_inputs,
                        result,
                        applicant_name=saved_name or "Anonymous",
                    )
                    if ok:
                        st.success(f"✅  {msg}")
                    else:
                        st.error(msg)
            with scol2:
                st.markdown(
                    '<div style="padding:10px 0;color:#8892a4;font-size:13px;">'
                    'Saves this prediction to the <code>predictions</code> SQLite table '
                    'for audit and future review.</div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Prediction History
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        st.markdown("""
        <div style="font-size:20px;font-weight:700;color:#e8eaf2;
                    margin:10px 0 16px;display:flex;align-items:center;gap:10px;">
          📜 Prediction History
          <span style="flex:1;height:1px;background:rgba(255,255,255,0.08);margin-left:12px;"></span>
        </div>
        """, unsafe_allow_html=True)

        try:
            hist_df = get_prediction_history(limit=100)
        except Exception as e:
            st.error(f"Could not load history: {e}")
            return

        if hist_df.empty:
            st.info("No predictions saved yet. Run a prediction and click **Save to Database**.")
            return

        # ── Summary cards ──────────────────────────────────────────────────────
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Predictions", len(hist_df))
        h2.metric("Eligible",   int((hist_df["predicted_label"] == 1).sum()))
        h3.metric("Not Eligible", int((hist_df["predicted_label"] == 0).sum()))
        avg_conf = hist_df["confidence_score"].mean()
        h4.metric("Avg Confidence", f"{avg_conf*100:.1f}%" if not pd.isna(avg_conf) else "—")

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Format for display ─────────────────────────────────────────────────
        display = hist_df.copy()
        display["predicted_label"] = display["predicted_label"].map(
            {1: "Eligible", 0: "Not Eligible"}
        )
        display["disability_status"] = display["disability_status"].map(
            {1: "Yes", 0: "No"}
        )
        display["confidence_score"] = display["confidence_score"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )
        display["family_income"] = display["family_income"].apply(
            lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—"
        )
        display.columns = [c.replace("_", " ").title() for c in display.columns]
        st.dataframe(display, use_container_width=True, hide_index=True, height=420)

        # ── Export ─────────────────────────────────────────────────────────────
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Export prediction history as CSV",
            data      = csv,
            file_name = "prediction_history.csv",
            mime      = "text/csv",
        )
