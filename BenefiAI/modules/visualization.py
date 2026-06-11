"""
modules/visualization.py
────────────────────────
Fully-implemented data visualization module for BenefiAI.
All functions accept a Pandas DataFrame and return Plotly Figure objects.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = "#0F172A"
SURFACE   = "#1E293B"
SURFACE2  = "#263348"
BORDER    = "#334155"
TEXT      = "#E2E8F0"
SUBTEXT   = "#94A3B8"
ELIGIBLE  = "#22C55E"
INELIGIBLE= "#EF4444"
ACCENT    = "#38BDF8"
ACCENT2   = "#818CF8"

# Base layout — NO legend key here; each chart sets its own legend config.
LAYOUT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=TEXT, size=13),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=SUBTEXT)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=SUBTEXT)),
)

# Shared legend style reused by bar/scatter charts
_LEGEND = dict(
    bgcolor=SURFACE2,
    bordercolor=BORDER,
    borderwidth=1,
    font=dict(color=TEXT),
)

EDU_ORDER = ["No Formal", "Primary", "Secondary", "Graduate", "Post-Graduate"]


# ── 1. Eligibility Distribution (Donut) ───────────────────────────────────────

def eligibility_pie(df: pd.DataFrame) -> go.Figure:
    """Animated donut chart: Eligible vs Not Eligible."""
    counts = df["eligibility_status"].value_counts().reset_index()
    counts.columns = ["status", "count"]

    fig = go.Figure(go.Pie(
        labels=counts["status"],
        values=counts["count"],
        hole=0.60,
        marker=dict(
            colors=[ELIGIBLE if s == "Eligible" else INELIGIBLE
                    for s in counts["status"]],
            line=dict(color=SURFACE, width=3),
        ),
        textinfo="percent+label",
        textfont=dict(color=TEXT, size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        pull=[0.04, 0],
    ))

    total = len(df)
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px;color:{SUBTEXT}'>Total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=TEXT),
        align="center",
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Eligibility Distribution", font=dict(size=16, color=TEXT), x=0.02),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            bgcolor=SURFACE2, bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT),
        ),
    )
    return fig


# ── 2. Income Distribution (Histogram) ───────────────────────────────────────

def income_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlapping histogram of family income by eligibility status."""
    fig = go.Figure()

    for status, color in [("Eligible", ELIGIBLE), ("Not Eligible", INELIGIBLE)]:
        subset = df[df["eligibility_status"] == status]["family_income"]
        fig.add_trace(go.Histogram(
            x=subset,
            name=status,
            nbinsx=30,
            marker_color=color,
            opacity=0.75,
            hovertemplate=(
                f"<b>{status}</b><br>"
                "Income range: %{x}<br>"
                "Count: %{y}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="overlay",
        legend=_LEGEND,
        title=dict(text="Family Income Distribution", font=dict(size=16, color=TEXT), x=0.02),
        xaxis_title="Annual Family Income (INR)",
        yaxis_title="Number of Applicants",
        xaxis_tickprefix="Rs.",
        xaxis_tickformat=",",
    )
    return fig


# ── 3. Education Level Distribution (Grouped Bar) ────────────────────────────

def education_breakdown(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: education level breakdown by eligibility."""
    grp = (
        df.groupby(["education_level", "eligibility_status"])
          .size()
          .reset_index(name="count")
    )
    grp["education_level"] = pd.Categorical(
        grp["education_level"], categories=EDU_ORDER, ordered=True
    )
    grp = grp.sort_values("education_level")

    fig = go.Figure()
    for status, color in [("Eligible", ELIGIBLE), ("Not Eligible", INELIGIBLE)]:
        sub = grp[grp["eligibility_status"] == status]
        fig.add_trace(go.Bar(
            x=sub["education_level"],
            y=sub["count"],
            name=status,
            marker_color=color,
            marker_line=dict(width=0),
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        legend=_LEGEND,
        title=dict(text="Education Level Distribution", font=dict(size=16, color=TEXT), x=0.02),
        xaxis_title="Education Level",
        yaxis_title="Count",
    )
    return fig


# ── 4. Employment Breakdown (Stacked Bar) ─────────────────────────────────────

def employment_breakdown(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart: employment status × eligibility."""
    grp = (
        df.groupby(["employment_status", "eligibility_status"])
          .size()
          .reset_index(name="count")
    )
    emp_order = ["Unemployed", "Part-time", "Self-employed", "Full-time"]
    grp["employment_status"] = pd.Categorical(
        grp["employment_status"], categories=emp_order, ordered=True
    )
    grp = grp.sort_values("employment_status")

    fig = go.Figure()
    for status, color in [("Eligible", ELIGIBLE), ("Not Eligible", INELIGIBLE)]:
        sub = grp[grp["eligibility_status"] == status]
        fig.add_trace(go.Bar(
            x=sub["employment_status"],
            y=sub["count"],
            name=status,
            marker_color=color,
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="stack",
        legend=_LEGEND,
        title=dict(text="Employment Status Breakdown", font=dict(size=16, color=TEXT), x=0.02),
        xaxis_title="Employment Status",
        yaxis_title="Count",
    )
    return fig


# ── 5. Age Distribution ───────────────────────────────────────────────────────

def age_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of applicants in age brackets."""
    bins   = [18, 25, 35, 45, 55, 65, 71]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    d = df.copy()
    d["age_group"] = pd.cut(d["age"], bins=bins, labels=labels, right=False)
    grp = (
        d.groupby(["age_group", "eligibility_status"], observed=True)
          .size().reset_index(name="count")
    )

    fig = go.Figure()
    for status, color in [("Eligible", ELIGIBLE), ("Not Eligible", INELIGIBLE)]:
        sub = grp[grp["eligibility_status"] == status]
        fig.add_trace(go.Bar(
            x=sub["age_group"].astype(str),
            y=sub["count"],
            name=status,
            marker_color=color,
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="stack",
        legend=_LEGEND,
        title=dict(text="Age Group Distribution", font=dict(size=16, color=TEXT), x=0.02),
        xaxis_title="Age Group",
        yaxis_title="Count",
    )
    return fig


# ── 6. Disability Impact ──────────────────────────────────────────────────────

def disability_impact(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: disability status vs eligibility."""
    grp = (
        df.groupby(["disability_status", "eligibility_status"])
          .size().reset_index(name="count")
    )
    fig = go.Figure()
    for status, color in [("Eligible", ELIGIBLE), ("Not Eligible", INELIGIBLE)]:
        sub = grp[grp["eligibility_status"] == status]
        fig.add_trace(go.Bar(
            x=sub["disability_status"],
            y=sub["count"],
            name=status,
            marker_color=color,
            hovertemplate="<b>Disability: %{x}</b><br>%{fullData.name}: %{y}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        legend=_LEGEND,
        title=dict(text="Disability Status Impact", font=dict(size=16, color=TEXT), x=0.02),
        xaxis_title="Disability Status",
        yaxis_title="Count",
    )
    return fig


# ── 7. Income vs Family Members (Scatter) ────────────────────────────────────

def income_vs_members(df: pd.DataFrame) -> go.Figure:
    """Scatter: family income vs members, coloured by eligibility."""
    color_map = {"Eligible": ELIGIBLE, "Not Eligible": INELIGIBLE}
    fig = px.scatter(
        df,
        x="family_members",
        y="family_income",
        color="eligibility_status",
        color_discrete_map=color_map,
        opacity=0.65,
        hover_data=["applicant_name", "age", "employment_status"],
        labels={
            "family_members": "Family Members",
            "family_income":  "Family Income (INR)",
            "eligibility_status": "Status",
        },
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color=SURFACE)))
    fig.update_layout(
        **LAYOUT_BASE,
        legend=_LEGEND,
        title=dict(text="Income vs Family Members", font=dict(size=16, color=TEXT), x=0.02),
        yaxis_tickprefix="Rs.",
        yaxis_tickformat=",",
    )
    return fig
