"""
visualization/charts.py
────────────────────────
All Plotly chart functions for the BenefiAI dashboard.
Every function accepts a pandas DataFrame and returns a plotly Figure
ready to be rendered with st.plotly_chart(fig, use_container_width=True).

Colour palette (accessible, works on dark backgrounds):
  Teal    #00d4aa
  Violet  #7c5cbf
  Amber   #f5a623
  Rose    #f25a7d
  Sky     #38bdf8
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Shared theme ───────────────────────────────────────────────────────────────
_BG        = "rgba(0,0,0,0)"          # transparent — lets CSS control bg
_PAPER_BG  = "rgba(15,18,35,0.0)"
_FONT      = dict(family="Inter, sans-serif", color="#c9d1e0")
_GRIDCOLOR = "rgba(255,255,255,0.06)"
_PALETTE   = ["#00d4aa", "#7c5cbf", "#f5a623", "#f25a7d", "#38bdf8", "#a3e635"]

def _base_layout(**kwargs) -> dict:
    """Shared Plotly layout settings for a dark, glassmorphism look."""
    return dict(
        paper_bgcolor = _PAPER_BG,
        plot_bgcolor  = _BG,
        font          = _FONT,
        margin        = dict(l=20, r=20, t=40, b=20),
        legend        = dict(
            bgcolor     = "rgba(255,255,255,0.04)",
            bordercolor = "rgba(255,255,255,0.10)",
            borderwidth = 1,
            font        = dict(color="#c9d1e0"),
        ),
        **kwargs,
    )


# ── 1. Income Distribution ─────────────────────────────────────────────────────

def plot_income_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Overlapping histogram of family income split by eligibility status.
    Shows the income boundary that separates eligible from ineligible applicants.
    """
    df_plot = df.copy()
    df_plot["Eligibility"] = df_plot["eligibility_status"].map(
        {1: "Eligible", 0: "Not Eligible"}
    )

    fig = px.histogram(
        df_plot,
        x        = "family_income",
        color    = "Eligibility",
        nbins    = 40,
        barmode  = "overlay",
        opacity  = 0.78,
        color_discrete_map = {"Eligible": "#00d4aa", "Not Eligible": "#f25a7d"},
        labels   = {"family_income": "Annual Family Income (INR)", "count": "Applicants"},
        title    = "Income Distribution by Eligibility",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        **_base_layout(),
        xaxis = dict(
            gridcolor = _GRIDCOLOR,
            tickprefix= "₹",
            tickformat= ",.0f",
            color     = "#c9d1e0",
        ),
        yaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        bargap = 0.05,
    )
    return fig


# ── 2. Eligibility Distribution ────────────────────────────────────────────────

def plot_eligibility_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart showing the eligible vs not-eligible split.
    """
    counts = df["eligibility_status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    counts["label"] = counts["status"].map({1: "Eligible", 0: "Not Eligible"})

    fig = go.Figure(go.Pie(
        labels       = counts["label"],
        values       = counts["count"],
        hole         = 0.62,
        marker       = dict(
            colors = ["#00d4aa", "#f25a7d"],
            line   = dict(color="rgba(255,255,255,0.08)", width=2),
        ),
        textinfo     = "percent",
        hovertemplate= "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        direction    = "clockwise",
    ))

    total = len(df)
    fig.add_annotation(
        text       = f"<b>{total}</b><br><span style='font-size:11px'>Total</span>",
        x=0.5, y=0.5,
        font       = dict(size=22, color="#ffffff"),
        showarrow  = False,
    )
    fig.update_layout(
        **_base_layout(title="Eligibility Distribution"),
        showlegend = True,
    )
    return fig


# ── 3. Education Level Distribution ───────────────────────────────────────────

def plot_education_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal grouped bar chart showing education level counts
    broken down by eligibility status.
    """
    edu_order = [
        "No Formal", "Primary", "Secondary",
        "Higher Secondary", "Graduate", "Post-Graduate",
    ]
    df_plot = df.copy()
    df_plot["Eligibility"] = df_plot["eligibility_status"].map(
        {1: "Eligible", 0: "Not Eligible"}
    )
    grouped = (
        df_plot.groupby(["education_level", "Eligibility"])
               .size()
               .reset_index(name="count")
    )

    # Keep only levels present in data, maintain natural order
    present = [e for e in edu_order if e in grouped["education_level"].values]

    fig = px.bar(
        grouped,
        x                  = "count",
        y                  = "education_level",
        color              = "Eligibility",
        orientation        = "h",
        barmode            = "group",
        category_orders    = {"education_level": present},
        color_discrete_map = {"Eligible": "#00d4aa", "Not Eligible": "#f25a7d"},
        labels             = {"count": "Applicants", "education_level": "Education Level"},
        title              = "Education Level Distribution",
    )
    fig.update_layout(
        **_base_layout(),
        xaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        yaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        bargap      = 0.25,
        bargroupgap = 0.08,
    )
    return fig


# ── 4. Employment Status Distribution ─────────────────────────────────────────

def plot_employment_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart — employment status vs eligibility count.
    """
    df_plot = df.copy()
    df_plot["Eligibility"] = df_plot["eligibility_status"].map(
        {1: "Eligible", 0: "Not Eligible"}
    )
    grouped = (
        df_plot.groupby(["employment_status", "Eligibility"])
               .size()
               .reset_index(name="count")
    )

    fig = px.bar(
        grouped,
        x                  = "employment_status",
        y                  = "count",
        color              = "Eligibility",
        barmode            = "stack",
        color_discrete_map = {"Eligible": "#00d4aa", "Not Eligible": "#f25a7d"},
        labels             = {"count": "Applicants", "employment_status": "Employment Status"},
        title              = "Employment Status Breakdown",
    )
    fig.update_layout(
        **_base_layout(),
        xaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        yaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        bargap = 0.3,
    )
    return fig


# ── 5. Age Distribution ────────────────────────────────────────────────────────

def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Age distribution histogram with KDE-like smoothing.
    """
    fig = px.histogram(
        df,
        x       = "age",
        nbins   = 30,
        color_discrete_sequence = ["#7c5cbf"],
        labels  = {"age": "Age (years)", "count": "Applicants"},
        title   = "Age Distribution",
    )
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_layout(
        **_base_layout(),
        xaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        yaxis = dict(gridcolor=_GRIDCOLOR, color="#c9d1e0"),
        bargap= 0.05,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ML Charts  (added in Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

# ── 6. Confusion Matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list[list[int]]) -> go.Figure:
    """
    Annotated heatmap confusion matrix.

    Quadrant meaning:
      TN (top-left)  = correctly predicted Not Eligible
      FP (top-right) = predicted Eligible but actually Not Eligible
      FN (bot-left)  = predicted Not Eligible but actually Eligible
      TP (bot-right) = correctly predicted Eligible

    Color scale goes from dark (0) to teal (high) so true positives stand out.
    """
    labels   = ["Not Eligible", "Eligible"]
    tn, fp   = cm[0][0], cm[0][1]
    fn, tp   = cm[1][0], cm[1][1]
    z        = [[tn, fp], [fn, tp]]
    text     = [[f"TN<br><b>{tn}</b>", f"FP<br><b>{fp}</b>"],
                 [f"FN<br><b>{fn}</b>", f"TP<br><b>{tp}</b>"]]

    fig = go.Figure(go.Heatmap(
        z             = z,
        x             = labels,
        y             = labels,
        text          = text,
        texttemplate  = "%{text}",
        textfont      = dict(size=16, color="white"),
        colorscale    = [
            [0.0, "#1a1e2e"],
            [0.5, "#2d5a4f"],
            [1.0, "#00d4aa"],
        ],
        showscale     = False,
        hovertemplate = "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title="Confusion Matrix"),
        xaxis = dict(
            title      = "Predicted Label",
            color      = "#c9d1e0",
            tickfont   = dict(size=13),
        ),
        yaxis = dict(
            title      = "True Label",
            color      = "#c9d1e0",
            tickfont   = dict(size=13),
            autorange  = "reversed",
        ),
    )
    return fig


# ── 7. Feature Importance ──────────────────────────────────────────────────────

def plot_feature_importance(importances: dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart of RandomForest feature importances.

    Importance = mean decrease in impurity across all trees.
    Higher → the feature contributed more to the model's decisions.
    """
    # Sort ascending so the most important bar appears at the top
    items   = sorted(importances.items(), key=lambda kv: kv[1])
    names   = [k.replace("_", " ").title() for k, _ in items]
    values  = [v for _, v in items]
    colors  = [
        "#00d4aa" if v == max(values)
        else "#7c5cbf" if v >= sorted(values)[-2]
        else "#38bdf8"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x             = values,
        y             = names,
        orientation   = "h",
        marker        = dict(
            color     = colors,
            line      = dict(width=0),
        ),
        text          = [f"{v*100:.1f}%" for v in values],
        textposition  = "outside",
        textfont      = dict(color="#c9d1e0", size=12),
        hovertemplate = "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title="Feature Importance (Mean Decrease in Impurity)"),
        xaxis = dict(
            gridcolor  = _GRIDCOLOR,
            color      = "#c9d1e0",
            tickformat = ".0%",
            range      = [0, max(values) * 1.25],
        ),
        yaxis = dict(color="#c9d1e0"),
    )
    return fig


# ── 8. ROC Curve ──────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_test:       list[int],
    y_pred_proba: list[float],
    roc_auc:      float,
) -> go.Figure:
    """
    ROC (Receiver Operating Characteristic) curve.

    The curve plots TPR (recall) vs FPR at every classification threshold.
    AUC = 1.0 → perfect; AUC = 0.5 → random baseline (dashed diagonal).
    """
    from sklearn.metrics import roc_curve
    import numpy as np

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig = go.Figure()

    # ── Random baseline ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = [0, 1], y = [0, 1],
        mode = "lines",
        name = "Random Baseline (AUC = 0.50)",
        line = dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
    ))

    # ── Model ROC curve ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x            = fpr.tolist(),
        y            = tpr.tolist(),
        mode         = "lines",
        name         = f"RandomForest  (AUC = {roc_auc:.3f})",
        line         = dict(color="#00d4aa", width=2.5),
        fill         = "tozeroy",
        fillcolor    = "rgba(0,212,170,0.08)",
        hovertemplate= "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title="ROC Curve"),
        xaxis = dict(
            title      = "False Positive Rate",
            gridcolor  = _GRIDCOLOR,
            color      = "#c9d1e0",
            range      = [0, 1],
        ),
        yaxis = dict(
            title      = "True Positive Rate (Recall)",
            gridcolor  = _GRIDCOLOR,
            color      = "#c9d1e0",
            range      = [0, 1.02],
        ),
        legend = dict(
            bgcolor     = "rgba(255,255,255,0.04)",
            bordercolor = "rgba(255,255,255,0.10)",
            borderwidth = 1,
            x=0.60, y=0.08,
        ),
    )
    return fig


# ── 9. Probability Gauge (single prediction) ───────────────────────────────────

def plot_prediction_gauge(prob_eligible: float) -> go.Figure:
    """
    Gauge chart showing the model's confidence that an applicant is Eligible.
    Used on the Predict Eligibility page for single-applicant inference.
    """
    color = "#00d4aa" if prob_eligible >= 0.5 else "#f25a7d"

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = prob_eligible * 100,
        number = dict(suffix="%", font=dict(size=36, color=color)),
        delta  = dict(reference=50, valueformat=".1f"),
        title  = dict(
            text = "Eligibility Confidence",
            font = dict(size=16, color="#c9d1e0"),
        ),
        gauge  = dict(
            axis      = dict(range=[0, 100], tickcolor="#c9d1e0", tickfont=dict(color="#c9d1e0")),
            bar       = dict(color=color, thickness=0.25),
            bgcolor   = "rgba(255,255,255,0.04)",
            borderwidth= 0,
            steps     = [
                dict(range=[0, 50],  color="rgba(242,90,125,0.12)"),
                dict(range=[50, 100], color="rgba(0,212,170,0.12)"),
            ],
            threshold = dict(
                line  = dict(color="white", width=2),
                thickness=0.75,
                value = 50,
            ),
        ),
    ))

    fig.update_layout(
        **_base_layout(),
        height = 260,
    )
    return fig
