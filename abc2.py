import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────

df = pd.read_csv("covid_19_data.csv")
df["ObservationDate"] = pd.to_datetime(df["ObservationDate"])
df["Confirmed"]  = pd.to_numeric(df["Confirmed"],  errors="coerce").fillna(0)
df["Deaths"]     = pd.to_numeric(df["Deaths"],     errors="coerce").fillna(0)
df["Recovered"]  = pd.to_numeric(df["Recovered"],  errors="coerce").fillna(0)

# ── Aggregate per country+date (sum over provinces)
by_cd = (
    df.groupby(["Country/Region", "ObservationDate"])[["Confirmed", "Deaths", "Recovered"]]
    .sum()
    .reset_index()
    .sort_values("ObservationDate")
)

# ── Latest snapshot per country
latest_per_country = (
    by_cd.sort_values("ObservationDate")
    .groupby("Country/Region")
    .last()
    .reset_index()
)

# ── Global time series (sum all countries each day)
global_ts = (
    by_cd.groupby("ObservationDate")[["Confirmed", "Deaths", "Recovered"]]
    .sum()
    .reset_index()
    .sort_values("ObservationDate")
)
global_ts["DailyNew"]    = global_ts["Confirmed"].diff().clip(lower=0)
global_ts["DailyDeaths"] = global_ts["Deaths"].diff().clip(lower=0)
global_ts["Roll7Cases"]  = global_ts["DailyNew"].rolling(7, min_periods=1).mean()
global_ts["Roll7Deaths"] = global_ts["DailyDeaths"].rolling(7, min_periods=1).mean()

# ── KPI totals (last global row)
TOTAL_CASES     = global_ts["Confirmed"].iloc[-1]
TOTAL_DEATHS    = global_ts["Deaths"].iloc[-1]
TOTAL_RECOVERED = global_ts["Recovered"].iloc[-1]
CFR             = (TOTAL_DEATHS / TOTAL_CASES * 100) if TOTAL_CASES else 0
RECOVERY_RATE   = (TOTAL_RECOVERED / TOTAL_CASES * 100) if TOTAL_CASES else 0
N_COUNTRIES     = latest_per_country["Country/Region"].nunique()

# ── Top 15 countries
top15 = latest_per_country.nlargest(15, "Confirmed").copy()
top15["CFR"] = (top15["Deaths"] / top15["Confirmed"] * 100).round(2)

# ── Monthly heatmap data
global_ts["Year"]  = global_ts["ObservationDate"].dt.year
global_ts["Month"] = global_ts["ObservationDate"].dt.month
heatmap_df = (
    global_ts.groupby(["Year", "Month"])["DailyNew"]
    .mean()
    .reset_index()
)
pivot = heatmap_df.pivot(index="Year", columns="Month", values="DailyNew").fillna(0)
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Regional groups (mapped from dataset country names)
REGION_MAP = {
    "Asia-Pacific":  ["China", "Mainland China", "India", "Japan", "South Korea", "Australia",
                      "Philippines", "Indonesia", "Malaysia", "Thailand", "Vietnam", "Singapore",
                      "Pakistan", "Bangladesh", "Nepal", "Sri Lanka", "Myanmar", "Cambodia",
                      "New Zealand", "Taiwan", "Hong Kong", "Macau"],
    "Europe":        ["France", "Germany", "Italy", "Spain", "UK", "Russia", "Turkey", "Poland",
                      "Netherlands", "Belgium", "Sweden", "Switzerland", "Portugal", "Austria",
                      "Greece", "Czech Republic", "Romania", "Hungary", "Denmark", "Finland",
                      "Norway", "Slovakia", "Croatia", "Bulgaria", "Serbia", "Ukraine"],
    "Americas":      ["US", "Brazil", "Mexico", "Colombia", "Argentina", "Peru", "Chile",
                      "Ecuador", "Bolivia", "Paraguay", "Uruguay", "Venezuela", "Cuba",
                      "Dominican Republic", "Guatemala", "Honduras", "Panama", "Canada"],
    "Africa & M.East": ["South Africa", "Nigeria", "Ethiopia", "Egypt", "Kenya", "Morocco",
                        "Tunisia", "Algeria", "Ghana", "Iran", "Iraq", "Saudi Arabia",
                        "UAE", "Israel", "Jordan", "Pakistan", "Lebanon", "Kuwait"],
}


def get_region_ts(region):
    if region == "Global":
        return global_ts
    countries = REGION_MAP.get(region, [])
    sub = by_cd[by_cd["Country/Region"].isin(countries)]
    ts = (
        sub.groupby("ObservationDate")[["Confirmed", "Deaths", "Recovered"]]
        .sum()
        .reset_index()
        .sort_values("ObservationDate")
    )
    ts["DailyNew"]    = ts["Confirmed"].diff().clip(lower=0)
    ts["DailyDeaths"] = ts["Deaths"].diff().clip(lower=0)
    ts["Roll7Cases"]  = ts["DailyNew"].rolling(7, min_periods=1).mean()
    ts["Roll7Deaths"] = ts["DailyDeaths"].rolling(7, min_periods=1).mean()
    return ts


def get_region_kpis(region):
    if region == "Global":
        sub = latest_per_country
    else:
        countries = REGION_MAP.get(region, [])
        sub = latest_per_country[latest_per_country["Country/Region"].isin(countries)]
    c = sub["Confirmed"].sum()
    d = sub["Deaths"].sum()
    r = sub["Recovered"].sum()
    n = sub["Country/Region"].nunique()
    cfr_r = (d / c * 100) if c else 0
    rec_r = (r / c * 100) if c else 0
    return c, d, r, cfr_r, rec_r, n


# ─────────────────────────────────────────
# 2. THEME TOKENS
# ─────────────────────────────────────────

BG      = "#0b0f1a"
SURFACE = "#131828"
SRF2    = "#1a2035"
BORDER  = "rgba(99,179,237,0.12)"
ACCENT  = "#63b3ed"
A2      = "#f6ad55"
A3      = "#68d391"
A4      = "#fc8181"
A5      = "#b794f4"
MUTED   = "#718096"
TEXT    = "#e2e8f0"
MONO    = "Space Mono, monospace"

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=MONO, color=MUTED, size=10),
    colorway=[ACCENT, A4, A3, A2, A5, "#f687b3", "#76e4f7"],
    margin=dict(t=14, r=14, b=40, l=52),
    xaxis=dict(gridcolor="rgba(99,179,237,0.07)", zerolinecolor="rgba(99,179,237,0.1)",
               tickfont=dict(size=9), linecolor="rgba(99,179,237,0.1)"),
    yaxis=dict(gridcolor="rgba(99,179,237,0.07)", zerolinecolor="rgba(99,179,237,0.1)",
               tickfont=dict(size=9), linecolor="rgba(99,179,237,0.1)"),
    showlegend=False,
)

CFG = dict(displayModeBar=False, responsive=True)


def apply_base(fig, **overrides):
    layout = {**LAYOUT_BASE, **overrides}
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────
# 3. CHART BUILDERS
# ─────────────────────────────────────────

def fmt_big(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


def build_timeline(ts, mode="cases"):
    fig = go.Figure()
    if mode in ("cases", "both"):
        fig.add_trace(go.Scatter(
            x=ts["ObservationDate"], y=ts["Roll7Cases"],
            mode="lines", name="7-Day Avg Cases",
            line=dict(color=ACCENT, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(99,179,237,0.08)",
            hovertemplate="%{x|%b %Y}<br>Cases (7d avg): %{y:,.0f}<extra></extra>"
        ))
    if mode in ("deaths", "both"):
        fig.add_trace(go.Scatter(
            x=ts["ObservationDate"], y=ts["Roll7Deaths"],
            mode="lines", name="7-Day Avg Deaths",
            line=dict(color=A4, width=2, shape="spline",
                      dash="dot" if mode == "both" else "solid"),
            fill="tozeroy", fillcolor="rgba(252,129,129,0.07)",
            yaxis="y2" if mode == "both" else "y",
            hovertemplate="%{x|%b %Y}<br>Deaths (7d avg): %{y:,.0f}<extra></extra>"
        ))
    extra = {}
    if mode == "both":
        extra["yaxis2"] = dict(
            overlaying="y", side="right",
            gridcolor="rgba(252,129,129,0.04)",
            tickfont=dict(size=9, color=A4),
            title=dict(text="Deaths (7d avg)", font=dict(size=9, color=A4))
        )
        extra["showlegend"] = True
        extra["legend"] = dict(orientation="h", x=0, y=1.12, font=dict(size=9))
    apply_base(fig,
               yaxis=dict(**LAYOUT_BASE["yaxis"],
                          title=dict(text="Cases (7d avg)" if mode != "deaths" else "Deaths (7d avg)",
                                     font=dict(size=9))),
               **extra)
    return fig


def build_top_cases_bar():
    s = top15.sort_values("Confirmed", ascending=True)
    colors = [A2 if i == len(s)-1 else ACCENT if i == len(s)-2 else A3 if i == len(s)-3 else "#3a5470"
              for i in range(len(s))]
    fig = go.Figure(go.Bar(
        x=s["Confirmed"], y=s["Country/Region"],
        orientation="h",
        marker=dict(color=colors),
        text=[fmt_big(v) for v in s["Confirmed"]],
        textposition="outside", textfont=dict(size=9, color=TEXT),
        hovertemplate="%{y}: %{x:,.0f} cases<extra></extra>"
    ))
    apply_base(fig, margin=dict(t=14, r=70, b=30, l=85),
               xaxis=dict(**LAYOUT_BASE["xaxis"],
                          title=dict(text="Confirmed Cases", font=dict(size=9))))
    return fig


def build_deaths_bar():
    s = top15.sort_values("Deaths", ascending=True)
    fig = go.Figure(go.Bar(
        x=s["Deaths"], y=s["Country/Region"],
        orientation="h",
        marker=dict(color=A4, opacity=0.85),
        text=[fmt_big(v) for v in s["Deaths"]],
        textposition="outside", textfont=dict(size=9, color=TEXT),
        hovertemplate="%{y}: %{x:,.0f} deaths<extra></extra>"
    ))
    apply_base(fig, margin=dict(t=14, r=70, b=30, l=85),
               xaxis=dict(**LAYOUT_BASE["xaxis"],
                          title=dict(text="Confirmed Deaths", font=dict(size=9))))
    return fig


def build_cfr_scatter():
    fig = go.Figure(go.Scatter(
        x=top15["Confirmed"],
        y=top15["CFR"],
        mode="markers+text",
        text=top15["Country/Region"],
        textposition="top center",
        textfont=dict(size=8, color=MUTED),
        marker=dict(
            size=top15["Confirmed"].apply(lambda v: max(8, np.sqrt(v / 10000))),
            color=top15["CFR"].apply(lambda v: A4 if v > 3 else A2 if v > 1.5 else A3),
            opacity=0.85,
            line=dict(color=BG, width=1)
        ),
        hovertemplate="%{text}<br>Cases: %{x:,.0f}<br>CFR: %{y:.2f}%<extra></extra>"
    ))
    apply_base(fig,
               xaxis=dict(**LAYOUT_BASE["xaxis"],
                          title=dict(text="Total Confirmed Cases", font=dict(size=9))),
               yaxis=dict(**LAYOUT_BASE["yaxis"],
                          title=dict(text="Case Fatality Rate (%)", font=dict(size=9))))
    return fig


def build_region_donut():
    region_totals = {}
    for region, countries in REGION_MAP.items():
        sub = latest_per_country[latest_per_country["Country/Region"].isin(countries)]
        region_totals[region] = sub["Confirmed"].sum()
    labels = list(region_totals.keys())
    values = list(region_totals.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        textinfo="none",
        marker=dict(colors=[ACCENT, A5, A2, A3], line=dict(color=BG, width=2)),
        hovertemplate="%{label}<br>%{value:,.0f} cases<extra></extra>"
    ))
    total = sum(values)
    apply_base(fig,
               showlegend=True,
               legend=dict(font=dict(size=9), orientation="v", x=1.02, y=0.5),
               margin=dict(t=14, r=130, b=14, l=14),
               annotations=[dict(text=f"{fmt_big(total)}<br>Total",
                                 showarrow=False,
                                 font=dict(size=10, color=TEXT), x=0.37, y=0.5)])
    return fig


def build_heatmap():
    z_vals = pivot.values
    y_labels = [str(y) for y in pivot.index]
    x_labels = [month_names[m-1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        x=x_labels, y=y_labels, z=z_vals,
        colorscale=[[0, "#0b1525"], [0.15, "#0f2b4a"], [0.35, "#1a5276"],
                    [0.55, "#2874a6"], [0.7, A2], [0.85, "#e74c3c"], [1, A4]],
        hovertemplate="%{y} %{x}<br>Avg Daily New: %{z:,.0f}<extra></extra>",
        colorbar=dict(tickfont=dict(size=8), thickness=10, len=0.8)
    ))
    apply_base(fig, margin=dict(t=14, r=60, b=30, l=45))
    return fig


def build_recovery_bar():
    s = top15.copy()
    s["RecoveryRate"] = (s["Recovered"] / s["Confirmed"] * 100).round(1)
    s = s.sort_values("RecoveryRate", ascending=True)
    fig = go.Figure(go.Bar(
        x=s["RecoveryRate"], y=s["Country/Region"],
        orientation="h",
        marker=dict(color=s["RecoveryRate"].apply(
            lambda v: A3 if v > 80 else ACCENT if v > 50 else A2 if v > 20 else A4)),
        text=[f"{v:.1f}%" for v in s["RecoveryRate"]],
        textposition="outside", textfont=dict(size=9, color=TEXT),
        hovertemplate="%{y}: %{x:.1f}% recovered<extra></extra>"
    ))
    apply_base(fig, margin=dict(t=14, r=55, b=30, l=85),
               xaxis=dict(**LAYOUT_BASE["xaxis"], range=[0, 130],
                          title=dict(text="Recovery Rate (%)", font=dict(size=9))))
    return fig


def build_cfr_bar():
    s = top15.sort_values("CFR", ascending=True)
    fig = go.Figure(go.Bar(
        x=s["CFR"], y=s["Country/Region"],
        orientation="h",
        marker=dict(color=s["CFR"].apply(
            lambda v: A4 if v > 3 else A2 if v > 1.5 else A3)),
        text=[f"{v:.2f}%" for v in s["CFR"]],
        textposition="outside", textfont=dict(size=9, color=TEXT),
        hovertemplate="%{y}: CFR %{x:.2f}%<extra></extra>"
    ))
    apply_base(fig, margin=dict(t=14, r=60, b=30, l=85),
               xaxis=dict(**LAYOUT_BASE["xaxis"],
                          title=dict(text="Case Fatality Rate (%)", font=dict(size=9))))
    return fig


def build_daily_new_bar(ts):
    monthly = ts.resample("M", on="ObservationDate")["DailyNew"].sum().reset_index()
    fig = go.Figure(go.Bar(
        x=monthly["ObservationDate"], y=monthly["DailyNew"],
        marker=dict(color=ACCENT, opacity=0.8),
        hovertemplate="%{x|%b %Y}<br>New Cases: %{y:,.0f}<extra></extra>"
    ))
    apply_base(fig,
               yaxis=dict(**LAYOUT_BASE["yaxis"],
                          title=dict(text="Monthly New Cases", font=dict(size=9))),
               margin=dict(t=14, r=14, b=40, l=60))
    return fig


# ─────────────────────────────────────────
# 4. LAYOUT HELPERS
# ─────────────────────────────────────────

def card(children, style=None):
    base = dict(
        background=SURFACE,
        border=f"1px solid {BORDER}",
        borderRadius="12px",
        padding="1.2rem",
        marginBottom="0",
    )
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_label(text):
    return html.Div(text, style=dict(
        fontFamily=MONO, fontSize="0.65rem", letterSpacing="0.2em",
        color=MUTED, textTransform="uppercase", marginBottom="1rem",
        marginTop="1.5rem", paddingLeft="0.5rem",
        borderLeft=f"2px solid {ACCENT}"
    ))


def kpi_card(label, value_id, sub, color, accent_color):
    return html.Div([
        html.Div(label, style=dict(fontFamily=MONO, fontSize="0.62rem",
                                   letterSpacing="0.12em", color=MUTED,
                                   textTransform="uppercase", marginBottom="0.4rem")),
        html.Div(id=value_id, style=dict(fontSize="1.55rem", fontWeight="700",
                                          color=accent_color, lineHeight="1",
                                          marginBottom="0.3rem")),
        html.Div(sub, style=dict(fontSize="0.68rem", color=MUTED)),
    ], style=dict(
        background=SURFACE, border=f"1px solid {BORDER}",
        borderRadius="10px", padding="1.1rem", position="relative",
        borderRight=f"3px solid {accent_color}",
        flex="1", minWidth="150px"
    ))


def graph_card(title, subtitle, graph_id, height=300, extra_header=None):
    header = [
        html.Div([
            html.Div(title, style=dict(fontSize="0.85rem", fontWeight="600",
                                        color=TEXT, marginBottom="0.15rem")),
            html.Div(subtitle, style=dict(fontSize="0.65rem", color=MUTED,
                                          fontFamily=MONO)),
        ]),
    ]
    if extra_header:
        header.append(extra_header)
    return card([
        html.Div(header, style=dict(display="flex", justifyContent="space-between",
                                     alignItems="flex-start", marginBottom="0.8rem")),
        dcc.Graph(id=graph_id, config=CFG,
                  style=dict(height=f"{height}px")),
    ])


def tab_buttons(options, btn_id_prefix, default):
    return html.Div([
        html.Button(o, id=f"{btn_id_prefix}-{o.lower()}", n_clicks=0,
                    style=dict(
                        background=ACCENT if o.lower() == default else SRF2,
                        border=f"1px solid {BORDER}",
                        borderRadius="5px",
                        color=BG if o.lower() == default else MUTED,
                        fontFamily=MONO, fontSize="0.65rem",
                        padding="3px 10px", cursor="pointer",
                        letterSpacing="0.06em", marginLeft="4px"
                    ))
        for o in options
    ], style=dict(display="flex"))


# ─────────────────────────────────────────
# 5. BUILD APP
# ─────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                           "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap"],
    title="COVID-19 Intelligence Dashboard"
)

app.layout = html.Div([

    # ── HEADER
    html.Header([
        html.Div([
            html.Div(style=dict(
                width="36px", height="36px",
                border=f"2px solid {A2}", borderRadius="50%",
                display="flex", alignItems="center", justifyContent="center",
                background=f"radial-gradient(circle, {A2}33 30%, transparent 70%)"
            )),
            html.Div([
                html.H1("COVID-19 INTELLIGENCE DASHBOARD",
                        style=dict(fontSize="1.05rem", fontWeight="700",
                                   letterSpacing="0.08em", color=TEXT, margin=0)),
                html.P("GLOBAL EPIDEMIOLOGICAL SURVEILLANCE",
                       style=dict(fontSize="0.68rem", color=MUTED,
                                  fontFamily=MONO, letterSpacing="0.1em", margin=0)),
            ])
        ], style=dict(display="flex", alignItems="center", gap="14px")),

        html.Div([
            html.Div([
                html.Span("●", style=dict(color=A3, fontSize="0.7rem",
                                          animation="pulse 1.5s infinite")),
                html.Span(f" DATA: {global_ts['ObservationDate'].min().strftime('%b %Y')} – "
                          f"{global_ts['ObservationDate'].max().strftime('%b %Y')}",
                          style=dict(fontFamily=MONO, fontSize="0.68rem", color=A3,
                                     letterSpacing="0.1em"))
            ], style=dict(display="flex", alignItems="center")),

            dcc.Dropdown(
                id="region-select",
                options=[
                    {"label": "🌍  GLOBAL VIEW",     "value": "Global"},
                    {"label": "🌏  ASIA-PACIFIC",    "value": "Asia-Pacific"},
                    {"label": "🌍  EUROPE",          "value": "Europe"},
                    {"label": "🌎  AMERICAS",        "value": "Americas"},
                    {"label": "🌍  AFRICA & M.EAST", "value": "Africa & M.East"},
                ],
                value="Global",
                clearable=False,
                style=dict(width="200px", fontFamily=MONO, fontSize="0.72rem",
                           background=SRF2, color=TEXT),
            )
        ], style=dict(display="flex", alignItems="center", gap="1.5rem")),

    ], style=dict(
        background=SURFACE, borderBottom=f"1px solid {BORDER}",
        padding="1.1rem 2.5rem",
        display="flex", alignItems="center", justifyContent="space-between",
        position="sticky", top=0, zIndex=100
    )),

    # ── MAIN
    html.Main([

        section_label("KEY PERFORMANCE INDICATORS — CUMULATIVE TOTALS"),

        # KPI row
        html.Div([
            kpi_card("Total Cases",    "kpi-cases",     "Confirmed worldwide", "", ACCENT),
            kpi_card("Total Deaths",   "kpi-deaths",    "Confirmed fatalities", "", A4),
            kpi_card("Recovered",      "kpi-recovered", "Estimated recoveries", "", A3),
            kpi_card("CFR",            "kpi-cfr",       "Case fatality rate", "", A2),
            kpi_card("Countries",      "kpi-countries", "Jurisdictions reporting", "", A5),
        ], style=dict(display="flex", gap="14px", flexWrap="wrap", marginBottom="1.5rem")),

        # Insight strip
        html.Div([
            html.Div([
                html.Div("Peak Daily New Cases", style=dict(fontFamily=MONO, fontSize="0.62rem",
                         letterSpacing="0.1em", color=MUTED, marginBottom="4px")),
                html.Div(id="insight-peak", style=dict(fontSize="1rem", fontWeight="600", color=A4)),
                html.Div(id="insight-peak-date", style=dict(fontSize="0.65rem", color=MUTED, fontFamily=MONO)),
            ]),
            html.Div([
                html.Div("Total Recovered", style=dict(fontFamily=MONO, fontSize="0.62rem",
                         letterSpacing="0.1em", color=MUTED, marginBottom="4px")),
                html.Div(id="insight-recovered", style=dict(fontSize="1rem", fontWeight="600", color=A3)),
                html.Div("Cumulative recoveries", style=dict(fontSize="0.65rem", color=MUTED, fontFamily=MONO)),
            ]),
            html.Div([
                html.Div("Recovery Rate", style=dict(fontFamily=MONO, fontSize="0.62rem",
                         letterSpacing="0.1em", color=MUTED, marginBottom="4px")),
                html.Div(id="insight-recovery-rate", style=dict(fontSize="1rem", fontWeight="600", color=ACCENT)),
                html.Div("Recovered / Confirmed", style=dict(fontSize="0.65rem", color=MUTED, fontFamily=MONO)),
            ]),
            html.Div([
                html.Div("Active Countries", style=dict(fontFamily=MONO, fontSize="0.62rem",
                         letterSpacing="0.1em", color=MUTED, marginBottom="4px")),
                html.Div(id="insight-countries", style=dict(fontSize="1rem", fontWeight="600", color=A5)),
                html.Div("Reporting jurisdictions", style=dict(fontSize="0.65rem", color=MUTED, fontFamily=MONO)),
            ]),
        ], style=dict(
            background=SURFACE, border=f"1px solid {BORDER}",
            borderRadius="10px", padding="1rem 1.5rem",
            display="grid", gridTemplateColumns="repeat(4,1fr)",
            gap="1rem", marginBottom="1.5rem"
        )),

        section_label("TEMPORAL ANALYSIS"),

        # Timeline + Region Donut
        html.Div([
            html.Div([
                graph_card(
                    "Global Daily Cases & Deaths Timeline",
                    "7-DAY ROLLING AVERAGE",
                    "timeline-chart", height=310,
                    extra_header=html.Div([
                        html.Button("CASES",  id="tab-cases",  n_clicks=1,
                                    style=dict(background=ACCENT, border=f"1px solid {BORDER}",
                                               borderRadius="5px", color=BG, fontFamily=MONO,
                                               fontSize="0.65rem", padding="3px 10px",
                                               cursor="pointer", marginLeft="4px")),
                        html.Button("DEATHS", id="tab-deaths", n_clicks=0,
                                    style=dict(background=SRF2, border=f"1px solid {BORDER}",
                                               borderRadius="5px", color=MUTED, fontFamily=MONO,
                                               fontSize="0.65rem", padding="3px 10px",
                                               cursor="pointer", marginLeft="4px")),
                        html.Button("BOTH",   id="tab-both",   n_clicks=0,
                                    style=dict(background=SRF2, border=f"1px solid {BORDER}",
                                               borderRadius="5px", color=MUTED, fontFamily=MONO,
                                               fontSize="0.65rem", padding="3px 10px",
                                               cursor="pointer", marginLeft="4px")),
                    ], style=dict(display="flex")),
                )
            ], style=dict(flex="2", minWidth="0")),

            html.Div([
                graph_card("Regional Case Distribution",
                           "TOTAL CASES BY REGION",
                           "region-donut", height=310)
            ], style=dict(flex="1", minWidth="0")),
        ], style=dict(display="flex", gap="16px", marginBottom="16px")),

        # Monthly bar + Heatmap
        html.Div([
            html.Div([
                graph_card("Monthly New Cases", "TOTAL NEW CASES PER MONTH",
                           "monthly-bar", height=270)
            ], style=dict(flex="1", minWidth="0")),
            html.Div([
                graph_card("Monthly Case Intensity Heatmap",
                           "AVG DAILY NEW CASES × MONTH × YEAR",
                           "heatmap-chart", height=270)
            ], style=dict(flex="1", minWidth="0")),
        ], style=dict(display="flex", gap="16px", marginBottom="16px")),

        section_label("COUNTRY COMPARATIVE ANALYSIS"),

        # Top cases + Deaths
        html.Div([
            html.Div([
                graph_card("Top 15 Countries — Total Cases",
                           "CUMULATIVE CONFIRMED CASES",
                           "top-cases-bar", height=330)
            ], style=dict(flex="1", minWidth="0")),
            html.Div([
                graph_card("Top 15 Countries — Total Deaths",
                           "CUMULATIVE CONFIRMED DEATHS",
                           "deaths-bar", height=330)
            ], style=dict(flex="1", minWidth="0")),
        ], style=dict(display="flex", gap="16px", marginBottom="16px")),

        # CFR scatter + Recovery bar + CFR bar
        html.Div([
            html.Div([
                graph_card("Cases vs Case Fatality Rate",
                           "BUBBLE SIZE = TOTAL CASES",
                           "cfr-scatter", height=300)
            ], style=dict(flex="1", minWidth="0")),
            html.Div([
                graph_card("Recovery Rate by Country",
                           "RECOVERED / CONFIRMED (%)",
                           "recovery-bar", height=300)
            ], style=dict(flex="1", minWidth="0")),
            html.Div([
                graph_card("Case Fatality Rate by Country",
                           "DEATHS / CONFIRMED (%)",
                           "cfr-bar", height=300)
            ], style=dict(flex="1", minWidth="0")),
        ], style=dict(display="flex", gap="16px", marginBottom="16px")),

        section_label("COUNTRY INTELLIGENCE TABLE"),

        # Table
        card([
            html.Div([
                html.Div([
                    html.Div("Top 15 Most Affected Nations — Comprehensive Statistics",
                             style=dict(fontSize="0.85rem", fontWeight="600", color=TEXT)),
                    html.Div("SORTED BY TOTAL CONFIRMED CASES",
                             style=dict(fontSize="0.65rem", color=MUTED, fontFamily=MONO)),
                ]),
                html.Div("SOURCE: COVID-19 DATASET",
                         style=dict(fontFamily=MONO, fontSize="0.62rem", color=MUTED)),
            ], style=dict(display="flex", justifyContent="space-between",
                          alignItems="flex-start", marginBottom="1rem")),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(col, style=dict(fontFamily=MONO, fontSize="0.62rem",
                                           letterSpacing="0.1em", color=MUTED,
                                           padding="0.5rem 0.75rem", textAlign="left",
                                           borderBottom=f"1px solid {BORDER}",
                                           textTransform="uppercase"))
                    for col in ["#", "Country", "Total Cases", "Total Deaths",
                                "Recovered", "CFR", "Severity"]
                ])),
                html.Tbody(id="country-table-body"),
            ], style=dict(width="100%", borderCollapse="collapse", fontSize="0.78rem")),
        ], style=dict(marginBottom="16px")),

    ], style=dict(padding="2rem 2.5rem", maxWidth="1600px", margin="0 auto")),

    # ── FOOTER
    html.Footer(
        f"COVID-19 INTELLIGENCE DASHBOARD  |  DATA: covid_19_data.csv  "
        f"|  RECORDS: {len(df):,}  |  DATE RANGE: "
        f"{global_ts['ObservationDate'].min().strftime('%b %Y')} – "
        f"{global_ts['ObservationDate'].max().strftime('%b %Y')}",
        style=dict(textAlign="center", padding="1.5rem",
                   fontFamily=MONO, fontSize="0.62rem", color=MUTED,
                   letterSpacing="0.08em", borderTop=f"1px solid {BORDER}")
    ),

    # State stores
    dcc.Store(id="timeline-mode", data="cases"),

], style=dict(background=BG, color=TEXT,
              fontFamily="Sora, sans-serif", minHeight="100vh"))


# ─────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────

@app.callback(
    Output("timeline-mode", "data"),
    Output("tab-cases",  "style"),
    Output("tab-deaths", "style"),
    Output("tab-both",   "style"),
    Input("tab-cases",  "n_clicks"),
    Input("tab-deaths", "n_clicks"),
    Input("tab-both",   "n_clicks"),
    prevent_initial_call=True,
)
def update_tab_mode(nc, nd, nb):
    from dash import ctx
    triggered = ctx.triggered_id
    mode = "cases" if triggered == "tab-cases" else "deaths" if triggered == "tab-deaths" else "both"
    active   = dict(background=ACCENT, border=f"1px solid {BORDER}", borderRadius="5px",
                    color=BG, fontFamily=MONO, fontSize="0.65rem",
                    padding="3px 10px", cursor="pointer", marginLeft="4px")
    inactive = dict(background=SRF2, border=f"1px solid {BORDER}", borderRadius="5px",
                    color=MUTED, fontFamily=MONO, fontSize="0.65rem",
                    padding="3px 10px", cursor="pointer", marginLeft="4px")
    styles = {
        "cases":  (active, inactive, inactive),
        "deaths": (inactive, active, inactive),
        "both":   (inactive, inactive, active),
    }
    return (mode,) + styles[mode]


@app.callback(
    Output("kpi-cases",     "children"),
    Output("kpi-deaths",    "children"),
    Output("kpi-recovered", "children"),
    Output("kpi-cfr",       "children"),
    Output("kpi-countries", "children"),
    Output("insight-peak",          "children"),
    Output("insight-peak-date",     "children"),
    Output("insight-recovered",     "children"),
    Output("insight-recovery-rate", "children"),
    Output("insight-countries",     "children"),
    Output("timeline-chart",  "figure"),
    Output("region-donut",    "figure"),
    Output("monthly-bar",     "figure"),
    Output("heatmap-chart",   "figure"),
    Output("top-cases-bar",   "figure"),
    Output("deaths-bar",      "figure"),
    Output("cfr-scatter",     "figure"),
    Output("recovery-bar",    "figure"),
    Output("cfr-bar",         "figure"),
    Output("country-table-body", "children"),
    Input("region-select", "value"),
    Input("timeline-mode", "data"),
)
def update_all(region, mode):
    # ── Region-filtered time series
    ts = get_region_ts(region)
    c, d, r, cfr_r, rec_r, n = get_region_kpis(region)

    # KPIs
    kpi_cases     = fmt_big(c)
    kpi_deaths    = fmt_big(d)
    kpi_recovered = fmt_big(r)
    kpi_cfr       = f"{cfr_r:.2f}%"
    kpi_countries = str(n)

    # Insights
    peak_val  = ts["DailyNew"].max()
    peak_date = ts.loc[ts["DailyNew"].idxmax(), "ObservationDate"].strftime("%b %d, %Y")
    ins_peak  = fmt_big(peak_val)
    ins_rec   = fmt_big(r)
    ins_rr    = f"{rec_r:.1f}%"
    ins_cnt   = str(n)

    # Timeline
    fig_timeline = build_timeline(ts, mode or "cases")

    # Donut (always global — shows regional breakdown regardless)
    fig_donut = build_region_donut()

    # Monthly bar
    fig_monthly = build_daily_new_bar(ts)

    # Heatmap (global only, data doesn't change with region meaningfully here)
    fig_heatmap = build_heatmap()

    # Country charts (top15 is always global top15 for comparison)
    fig_cases    = build_top_cases_bar()
    fig_deaths   = build_deaths_bar()
    fig_scatter  = build_cfr_scatter()
    fig_recovery = build_recovery_bar()
    fig_cfr      = build_cfr_bar()

    # Table rows
    s = top15.sort_values("Confirmed", ascending=False).reset_index(drop=True)
    td_style = dict(padding="0.5rem 0.75rem",
                    borderBottom=f"1px solid rgba(99,179,237,0.05)")
    rank_colors = [
        "rgba(246,173,85,0.15)", "rgba(99,179,237,0.1)",
        "rgba(104,211,145,0.1)"
    ]
    rank_text_colors = [A2, ACCENT, A3]

    table_rows = []
    for i, row in s.iterrows():
        cfr_val = row["CFR"]
        severity_color = A4 if cfr_val > 3 else A2 if cfr_val > 1.5 else A3
        severity_text  = "CRITICAL" if cfr_val > 3 else "HIGH" if cfr_val > 1.5 else "MODERATE"
        rb_bg  = rank_colors[i] if i < 3 else "rgba(113,128,150,0.1)"
        rb_clr = rank_text_colors[i] if i < 3 else MUTED
        rec_val = row["Recovered"]
        table_rows.append(html.Tr([
            html.Td(html.Span(str(i+1), style=dict(
                display="inline-flex", alignItems="center", justifyContent="center",
                width="22px", height="22px", borderRadius="4px",
                fontFamily=MONO, fontSize="0.65rem", fontWeight="700",
                background=rb_bg, color=rb_clr
            )), style=td_style),
            html.Td(row["Country/Region"], style=dict(**td_style, fontWeight="600")),
            html.Td(fmt_big(row["Confirmed"]), style=td_style),
            html.Td(fmt_big(row["Deaths"]),    style=dict(**td_style, color=A4)),
            html.Td(fmt_big(rec_val) if rec_val > 0 else "N/A", style=td_style),
            html.Td(f"{cfr_val:.2f}%", style=dict(**td_style, fontFamily=MONO)),
            html.Td(html.Span(severity_text, style=dict(color=severity_color,
                                                         fontFamily=MONO,
                                                         fontSize="0.68rem")),
                    style=td_style),
        ]))

    return (
        kpi_cases, kpi_deaths, kpi_recovered, kpi_cfr, kpi_countries,
        ins_peak, peak_date, ins_rec, ins_rr, ins_cnt,
        fig_timeline, fig_donut, fig_monthly, fig_heatmap,
        fig_cases, fig_deaths, fig_scatter, fig_recovery, fig_cfr,
        table_rows
    )


# ─────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  COVID-19 Intelligence Dashboard")
    print(f"  Records loaded : {len(df):,}")
    print(f"  Countries      : {N_COUNTRIES}")
    print(f"  Date range     : {global_ts['ObservationDate'].min().strftime('%b %Y')} "
          f"– {global_ts['ObservationDate'].max().strftime('%b %Y')}")
    print("  URL            : http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=False, port=8050)