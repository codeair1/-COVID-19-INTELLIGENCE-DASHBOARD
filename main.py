import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
from functools import lru_cache

# ============== DATA LOADING & CACHING ==============
@lru_cache(maxsize=1)
def load_and_process_data():
    """Load and preprocess data with caching for performance"""
    df = pd.read_csv('covid_19_data.csv')
    
    df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
    
    def parse_mixed_dates(date_str):
        formats = [None, '%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%dT%H:%M:%S']
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt) if fmt else pd.to_datetime(date_str)
            except:
                continue
        return pd.NaT
    
    df['Last Update'] = df['Last Update'].apply(parse_mixed_dates)
    
    country_mapping = {
        'Mainland China': 'China', 'US': 'United States', 'UK': 'United Kingdom',
        'Korea, South': 'South Korea', 'Republic of Korea': 'South Korea',
        'Iran (Islamic Republic of)': 'Iran', 'Hong Kong SAR': 'Hong Kong',
        'Taiwan*': 'Taiwan', 'Macao SAR': 'Macau', 'Russian Federation': 'Russia',
        'Viet Nam': 'Vietnam'
    }
    df['Country/Region'] = df['Country/Region'].replace(country_mapping)
    df['Province/State'] = df['Province/State'].fillna('Unknown')
    
    return df

df = load_and_process_data()

latest_date = df['ObservationDate'].max()
earliest_date = df['ObservationDate'].min()

# ============== GLOBAL SUMMARY ==============
latest_data = df[df['ObservationDate'] == latest_date]
global_summary = latest_data.groupby('Country/Region', as_index=False).agg({
    'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'
})

global_summary['Active'] = global_summary['Confirmed'] - global_summary['Deaths'] - global_summary['Recovered']
global_summary['Mortality Rate (%)'] = (global_summary['Deaths'] / global_summary['Confirmed'] * 100).round(2)
global_summary['Recovery Rate (%)'] = (global_summary['Recovered'] / global_summary['Confirmed'] * 100).round(2)
global_summary = global_summary.replace([np.inf, -np.inf], 0).fillna(0)

total_confirmed = global_summary['Confirmed'].sum()
total_deaths = global_summary['Deaths'].sum()
total_recovered = global_summary['Recovered'].sum()
total_active = total_confirmed - total_deaths - total_recovered

# ============== DAILY GLOBAL TRENDS ==============
daily_global = df.groupby('ObservationDate', as_index=False).agg({
    'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'
})
daily_global['New Cases'] = daily_global['Confirmed'].diff().fillna(0).clip(lower=0)
daily_global['New Deaths'] = daily_global['Deaths'].diff().fillna(0).clip(lower=0)
daily_global['New Recovered'] = daily_global['Recovered'].diff().fillna(0).clip(lower=0)

# ============== REGIONAL MAPPING ==============
region_mapping = {
    'China': 'East Asia', 'Japan': 'East Asia', 'South Korea': 'East Asia', 'Taiwan': 'East Asia',
    'Hong Kong': 'East Asia', 'Macau': 'East Asia',
    'India': 'South Asia', 'Pakistan': 'South Asia', 'Bangladesh': 'South Asia',
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
    'Colombia': 'South America', 'Peru': 'South America', 'Ecuador': 'South America',
    'United Kingdom': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe', 'France': 'Europe',
    'Germany': 'Europe', 'Russia': 'Europe', 'Turkey': 'Europe', 'Netherlands': 'Europe',
    'Belgium': 'Europe', 'Switzerland': 'Europe', 'Sweden': 'Europe', 'Portugal': 'Europe',
    'Austria': 'Europe', 'Poland': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe',
    'Ireland': 'Europe', 'Czech Republic': 'Europe', 'Romania': 'Europe',
    'Iran': 'Middle East', 'Saudi Arabia': 'Middle East', 'United Arab Emirates': 'Middle East',
    'Israel': 'Middle East', 'Qatar': 'Middle East', 'Iraq': 'Middle East',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Algeria': 'Africa', 'Morocco': 'Africa',
    'Nigeria': 'Africa',
    'Thailand': 'Southeast Asia', 'Malaysia': 'Southeast Asia', 'Singapore': 'Southeast Asia',
    'Indonesia': 'Southeast Asia', 'Philippines': 'Southeast Asia', 'Vietnam': 'Southeast Asia'
}

global_summary['Region'] = global_summary['Country/Region'].map(region_mapping).fillna('Other')
regional_summary = global_summary.groupby('Region', as_index=False).agg({
    'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 'Active': 'sum'
})

# ============== TOP COUNTRIES & TIME SERIES ==============
top20_confirmed = global_summary.nlargest(20, 'Confirmed')[
    ['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate (%)', 'Recovery Rate (%)']
]

country_daily = df.groupby(['ObservationDate', 'Country/Region'], as_index=False).agg({
    'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'
})

countries = sorted(df['Country/Region'].unique())

mortality_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0
recovery_rate = (total_recovered / total_confirmed * 100) if total_confirmed > 0 else 0
avg_daily_cases = daily_global['New Cases'].tail(7).mean()
today_new_cases = daily_global['New Cases'].iloc[-1]

# ============== DASHBOARD SETUP ==============
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])
app.title = "COVID-19 Intelligence Dashboard"

colors = {
    'bg': '#0a0e17',
    'card_bg': '#131a2c',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'confirmed': '#ef4444',
    'active': '#f59e0b',
    'deaths': '#b91c1c',
    'recovered': '#10b981',
    'accent': '#3b82f6',
    'border': '#1e293b'
}

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>COVID-19 Dashboard</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
            body { background: linear-gradient(135deg, #0a0e17 0%, #131a2c 100%); min-height: 100vh; }
            .card-stats {
                background: rgba(19, 26, 44, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 24px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: 1px solid #1e293b;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .card-stats:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
                border-color: #334155;
            }
            .dashboard-title {
                font-weight: 800;
                font-size: 2.2rem;
                background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.025em;
            }
            .section-header {
                font-weight: 600;
                letter-spacing: -0.015em;
                border-left: 4px solid #3b82f6;
                padding-left: 16px;
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                line-height: 1.2;
                letter-spacing: -0.025em;
            }
            .stat-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #94a3b8;
                font-weight: 500;
            }
            .stat-trend {
                font-size: 0.8rem;
                margin-top: 8px;
                display: flex;
                align-items: center;
                gap: 4px;
            }
            .trend-up { color: #ef4444; }
            .trend-down { color: #10b981; }
            .dropdown-custom .Select-control {
                background-color: #131a2c !important;
                border: 1px solid #1e293b !important;
                border-radius: 12px !important;
            }
            .dropdown-custom .Select-menu-outer {
                background-color: #131a2c !important;
                border: 1px solid #1e293b !important;
            }
            .dropdown-custom .Select-option {
                background-color: #131a2c !important;
                color: #f1f5f9 !important;
            }
            .dropdown-custom .Select-option:hover {
                background-color: #1e293b !important;
            }
            .dropdown-custom .Select-value-label {
                color: #f1f5f9 !important;
            }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #0a0e17; border-radius: 10px; }
            ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: #475569; }
            .pulse-dot {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #10b981;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                animation: pulse 2s infinite;
                margin-right: 6px;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
                100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }
            .glass-effect {
                background: rgba(19, 26, 44, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(51, 65, 85, 0.5);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# ============== COMPONENT BUILDERS ==============
def create_stat_card(title, value, color, icon, trend=None, trend_up=None):
    trend_html = None
    if trend:
        trend_class = 'trend-up' if trend_up else 'trend-down' if trend_up is False else ''
        trend_html = html.Div([
            html.I(className=f"bi bi-arrow-{'up' if trend_up else 'down'} me-1"),
            trend
        ], className=f"stat-trend {trend_class}")
    
    return dbc.Card([
        html.Div([
            html.Div([
                html.I(className=f"bi bi-{icon} fs-5", style={'color': color, 'opacity': 0.8}),
                html.P(title, className="stat-label mt-2"),
                html.H2(f"{value:,.0f}" if isinstance(value, (int, float)) else value, 
                       className="stat-value", style={'color': '#f1f5f9'}),
                trend_html
            ])
        ])
    ], className="card-stats")

def create_section_header(title, icon, badge=None):
    return html.Div([
        html.Div([
            html.I(className=f"bi bi-{icon} me-2", style={'color': colors['accent'], 'fontSize': '1.3rem'}),
            html.H3(title, className="section-header mb-0", style={'color': colors['text']}),
            html.Span(badge, className="badge ms-3", style={
                'backgroundColor': colors['accent'], 'color': 'white', 
                'padding': '5px 12px', 'borderRadius': '20px', 'fontSize': '0.75rem'
            }) if badge else None
        ], className="d-flex align-items-center")
    ], className="mb-3")

# ============== STAT CARDS ==============
summary_row = dbc.Row([
    dbc.Col(create_stat_card("Total Confirmed", total_confirmed, colors['confirmed'], 
                             "virus", f"+{today_new_cases:,.0f} today", True), width=3),
    dbc.Col(create_stat_card("Active Cases", total_active, colors['active'], "activity"), width=3),
    dbc.Col(create_stat_card("Total Deaths", total_deaths, colors['deaths'], 
                             "heart-pulse", f"{mortality_rate:.2f}% mortality"), width=3),
    dbc.Col(create_stat_card("Total Recovered", total_recovered, colors['recovered'], 
                             "heart-fill", f"{recovery_rate:.2f}% recovery", True), width=3),
], className="g-3 mb-4")

secondary_stats = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([html.I(className="bi bi-globe-americas me-2", style={'color': colors['accent']}),
                     html.Span("Countries", className="text-secondary")], className="mb-2"),
            html.H3(f"{len(countries)}", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats glass-effect"), width=2),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([html.Span(className="pulse-dot"),
                     html.Span("Live Data", className="text-secondary")], className="mb-2"),
            html.H5(latest_date.strftime('%b %d, %Y'), className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats glass-effect"), width=2),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([html.I(className="bi bi-calendar-range me-2", style={'color': colors['accent']}),
                     html.Span("Timeline", className="text-secondary")], className="mb-2"),
            html.H6(f"{earliest_date.strftime('%b %d')} - {latest_date.strftime('%b %d, %Y')}", 
                   className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats glass-effect"), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([html.I(className="bi bi-people-fill me-2", style={'color': colors['accent']}),
                     html.Span("Population Affected", className="text-secondary")], className="mb-2"),
            html.H5(f"{(total_confirmed/7800000000*100):.3f}%", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats glass-effect"), width=2),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([html.I(className="bi bi-graph-up-arrow me-2", style={'color': colors['accent']}),
                     html.Span("7-Day Avg Cases", className="text-secondary")], className="mb-2"),
            html.H5(f"{avg_daily_cases:,.0f}", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats glass-effect"), width=3),
], className="g-3 mb-4")

# ============== APP LAYOUT ==============
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🦠 COVID-19 Intelligence Dashboard", className="dashboard-title"),
                html.P("Real-time global pandemic tracking and predictive analytics", 
                       className="text-secondary", style={'fontSize': '1rem', 'letterSpacing': '0.3px'})
            ], className="py-3 px-2")
        ], width=12)
    ]),
    
    summary_row,
    secondary_stats,
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Global Pandemic Timeline", "graph-up-arrow", "REAL-TIME")),
                dbc.CardBody([dcc.Graph(id='global-timeline', config={'displayModeBar': False, 'responsive': True})])
            ], className="card-stats h-100")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Regional Distribution", "pie-chart-fill", "BY CONTINENT")),
                dbc.CardBody([dcc.Graph(id='regional-pie', config={'displayModeBar': False, 'responsive': True})])
            ], className="card-stats h-100")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Cases by Region", "bar-chart-line", "BREAKDOWN")),
                dbc.CardBody([dcc.Graph(id='regional-bar', config={'displayModeBar': False, 'responsive': True})])
            ], className="card-stats h-100")
        ], width=8),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("New Cases Heatmap", "grid-3x3-gap-fill", "LAST 30 DAYS")),
                dbc.CardBody([dcc.Graph(id='heatmap', config={'displayModeBar': False, 'responsive': True})])
            ], className="card-stats")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Top 20 Most Affected Countries", "trophy-fill", "RANKING")),
                dbc.CardBody([dcc.Graph(id='top20-bar', config={'displayModeBar': False, 'responsive': True})])
            ], className="card-stats")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Country Comparison", "arrow-left-right", "CONFIRMED & DEATHS")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Countries", className="fw-semibold mb-2", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='country-selector',
                                options=[{'label': c, 'value': c} for c in countries],
                                value=['United States', 'Brazil', 'India', 'Russia', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain'],
                                multi=True,
                                className="dropdown-custom mb-3"
                            )
                        ], width=8),
                        dbc.Col([
                            html.Label("Metric", className="fw-semibold mb-2", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='metric-selector',
                                options=[
                                    {'label': '📊 Confirmed', 'value': 'Confirmed'},
                                    {'label': '💀 Deaths', 'value': 'Deaths'}
                                ],
                                value='Confirmed',
                                inline=True,
                                inputStyle={"margin-right": "5px", "margin-left": "15px"},
                                labelStyle={'color': colors['text_secondary'], 'fontWeight': '500'}
                            )
                        ], width=4)
                    ]),
                    dcc.Graph(id='country-comparison', config={'displayModeBar': False, 'responsive': True})
                ])
            ], className="card-stats")
        ], width=12)
    ], className="mb-4"),
    
    html.Footer([
        html.Hr(style={'borderColor': colors['border'], 'opacity': 0.5}),
        dbc.Row([
            dbc.Col(html.P([html.I(className="bi bi-database me-2"), 
                           "Data: Johns Hopkins University CSSE"], 
                          className="text-center", style={'color': colors['text_secondary']}), width=12),
            dbc.Col(html.P(f"Last Updated: {datetime.now().strftime('%B %d, %Y at %H:%M')} | Powered by Dash & Plotly", 
                          className="text-center", style={'color': colors['text_secondary'], 'fontSize': '0.8rem'}), width=12)
        ])
    ], className="mt-4 pb-3")
    
], fluid=True, style={'backgroundColor': 'transparent', 'minHeight': '100vh', 'padding': '24px'})

# ============== CALLBACKS ==============
@app.callback(Output('global-timeline', 'figure'), Input('global-timeline', 'id'))
def update_global_timeline(_):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=daily_global['ObservationDate'], y=daily_global['New Cases'],
        name='Daily New Cases', marker_color=colors['confirmed'], opacity=0.6,
        hovertemplate='%{x|%b %d, %Y}<br>New Cases: %{y:,.0f}<extra></extra>',
        hoverlabel=dict(bgcolor='#1e293b', font_color='#f1f5f9', font_size=14)
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=daily_global['ObservationDate'], y=daily_global['Confirmed'],
        name='Cumulative Confirmed', line=dict(color=colors['confirmed'], width=3), mode='lines',
        hovertemplate='%{x|%b %d, %Y}<br>Confirmed: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=daily_global['ObservationDate'], y=daily_global['Deaths'],
        name='Cumulative Deaths', line=dict(color=colors['deaths'], width=2.5), mode='lines',
        hovertemplate='%{x|%b %d, %Y}<br>Deaths: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=daily_global['ObservationDate'], y=daily_global['Recovered'],
        name='Cumulative Recovered', line=dict(color=colors['recovered'], width=2.5), mode='lines',
        hovertemplate='%{x|%b %d, %Y}<br>Recovered: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text']}, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
        height=400, margin=dict(l=40, r=40, t=20, b=40), dragmode='pan'
    )
    fig.update_xaxes(title_text="", gridcolor=colors['border'], tickformat='%b %d, %Y')
    fig.update_yaxes(title_text="Daily New Cases", secondary_y=False, gridcolor=colors['border'])
    fig.update_yaxes(title_text="Cumulative Cases", secondary_y=True, gridcolor=colors['border'])
    return fig

@app.callback(Output('regional-pie', 'figure'), Input('regional-pie', 'id'))
def update_regional_pie(_):
    fig = px.pie(regional_summary, values='Confirmed', names='Region', hole=0.55,
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition='inside', textinfo='percent', textfont_size=10,
                      hovertemplate='%{label}<br>Cases: %{value:,.0f}<br>%{percent}<extra></extra>')
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text']}, height=320, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5, font=dict(size=9))
    )
    return fig

@app.callback(Output('regional-bar', 'figure'), Input('regional-bar', 'id'))
def update_regional_bar(_):
    fig = go.Figure()
    for col, color, name in [('Active', colors['active'], 'Active'), 
                              ('Recovered', colors['recovered'], 'Recovered'),
                              ('Deaths', colors['deaths'], 'Deaths')]:
        fig.add_trace(go.Bar(
            x=regional_summary['Region'], y=regional_summary[col], name=name, marker_color=color,
            hovertemplate='%{x}<br>' + name + ': %{y:,.0f}<extra></extra>'
        ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text']}, barmode='group', height=380,
        xaxis_title="", yaxis_title="Cases", margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
        xaxis={'gridcolor': colors['border']}, yaxis={'gridcolor': colors['border']}
    )
    return fig

@app.callback(Output('heatmap', 'figure'), Input('heatmap', 'id'))
def update_heatmap(_):
    top_countries = top20_confirmed.head(15)['Country/Region'].tolist()
    recent_data = country_daily[
        (country_daily['ObservationDate'] > latest_date - pd.Timedelta(days=30)) &
        (country_daily['Country/Region'].isin(top_countries))
    ].copy()
    
    recent_data['New Cases'] = recent_data.groupby('Country/Region')['Confirmed'].diff().fillna(0).clip(lower=0)
    pivot_data = recent_data.pivot(index='Country/Region', columns='ObservationDate', values='New Cases').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index,
        colorscale=[[0, '#131a2c'], [0.3, '#f59e0b'], [0.7, '#ef4444'], [1, '#b91c1c']],
        hoverongaps=False, colorbar=dict(title="Daily New Cases", len=0.8),
        hovertemplate='%{y}<br>%{x|%b %d, %Y}<br>New Cases: %{z:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text']}, height=400, xaxis_title="", yaxis_title="",
        margin=dict(l=120, r=40, t=20, b=40)
    )
    return fig

@app.callback(Output('top20-bar', 'figure'), Input('top20-bar', 'id'))
def update_top20_bar(_):
    fig = go.Figure()
    top20_sorted = top20_confirmed.sort_values('Confirmed', ascending=True)
    
    for col, color, name in [('Active', colors['active'], 'Active'), 
                              ('Recovered', colors['recovered'], 'Recovered'),
                              ('Deaths', colors['deaths'], 'Deaths')]:
        fig.add_trace(go.Bar(
            y=top20_sorted['Country/Region'], x=top20_sorted[col], name=name,
            orientation='h', marker_color=color,
            hovertemplate='%{y}<br>' + name + ': %{x:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text']}, barmode='stack', height=500,
        xaxis_title="Number of Cases", yaxis_title="",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=120, r=40, t=20, b=40), xaxis={'gridcolor': colors['border']}
    )
    return fig

@app.callback(
    Output('country-comparison', 'figure'),
    Input('country-selector', 'value'),
    Input('metric-selector', 'value')
)
def update_country_comparison(selected_countries, selected_metric):
    if not selected_countries:
        selected_countries = ['United States']
    
    fig = go.Figure()
    colors_list = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#d946ef']
    
    for i, country in enumerate(selected_countries[:10]):
        country_data = country_daily[country_daily['Country/Region'] == country].copy()
        if len(country_data) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=country_data['ObservationDate'], y=country_data[selected_metric],
            name=country, line=dict(width=2.5, color=colors_list[i % len(colors_list)]),
            hovertemplate='%{x|%b %d, %Y}<br>' + country + ': %{y:,.0f}<extra></extra>'
        ))
    
    metric_labels = {'Confirmed': 'Confirmed Cases', 'Deaths': 'Deaths'}
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text']}, xaxis_title="", yaxis_title=metric_labels[selected_metric],
        height=400, hovermode='x unified', margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='left', x=1.01, bgcolor='rgba(0,0,0,0)'),
        xaxis={'gridcolor': colors['border']}, yaxis={'gridcolor': colors['border']}
    )
    return fig

# ============== RUN ==============
if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8052)