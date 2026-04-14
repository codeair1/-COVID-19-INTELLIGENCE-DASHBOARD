import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime

# Read and preprocess the data
df = pd.read_csv('covid_19_data.csv')

# Convert date columns to datetime with flexible parsing
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

def parse_mixed_dates(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        try:
            return pd.to_datetime(date_str, dayfirst=False)
        except:
            try:
                return pd.to_datetime(date_str, format='%m/%d/%y %H:%M')
            except:
                try:
                    return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')
                except:
                    try:
                        return pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%S')
                    except:
                        return pd.NaT

df['Last Update'] = df['Last Update'].apply(parse_mixed_dates)

# Clean country names
df['Country/Region'] = df['Country/Region'].replace({
    'Mainland China': 'China',
    'US': 'United States',
    'UK': 'United Kingdom',
    'Korea, South': 'South Korea',
    'Republic of Korea': 'South Korea',
    'Iran (Islamic Republic of)': 'Iran',
    'Hong Kong SAR': 'Hong Kong',
    'Taiwan*': 'Taiwan',
    'Macao SAR': 'Macau',
    'Russian Federation': 'Russia',
    'Viet Nam': 'Vietnam'
})

df['Province/State'] = df['Province/State'].fillna('Unknown')

# Get the latest date
latest_date = df['ObservationDate'].max()
earliest_date = df['ObservationDate'].min()

# ============== GLOBAL SUMMARY ==============
latest_data = df[df['ObservationDate'] == latest_date]
global_summary = latest_data.groupby('Country/Region').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum'
}).reset_index()

global_summary['Active'] = global_summary['Confirmed'] - global_summary['Deaths'] - global_summary['Recovered']
global_summary['Mortality Rate (%)'] = (global_summary['Deaths'] / global_summary['Confirmed'] * 100).round(2)
global_summary['Recovery Rate (%)'] = (global_summary['Recovered'] / global_summary['Confirmed'] * 100).round(2)

global_summary = global_summary.replace([np.inf, -np.inf], 0).fillna(0)

total_confirmed = global_summary['Confirmed'].sum()
total_deaths = global_summary['Deaths'].sum()
total_recovered = global_summary['Recovered'].sum()
total_active = total_confirmed - total_deaths - total_recovered

# ============== DAILY GLOBAL TRENDS ==============
daily_global = df.groupby('ObservationDate').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum'
}).reset_index()

daily_global['New Cases'] = daily_global['Confirmed'].diff().fillna(0).clip(lower=0)
daily_global['New Deaths'] = daily_global['Deaths'].diff().fillna(0).clip(lower=0)
daily_global['New Recovered'] = daily_global['Recovered'].diff().fillna(0).clip(lower=0)

# ============== REGIONAL DISTRIBUTION ==============
region_mapping = {
    'China': 'East Asia', 'Japan': 'East Asia', 'South Korea': 'East Asia', 'Taiwan': 'East Asia',
    'Hong Kong': 'East Asia', 'Macau': 'East Asia', 'Mongolia': 'East Asia',
    'India': 'South Asia', 'Pakistan': 'South Asia', 'Bangladesh': 'South Asia',
    'Sri Lanka': 'South Asia', 'Nepal': 'South Asia', 'Afghanistan': 'South Asia',
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
    'Colombia': 'South America', 'Peru': 'South America', 'Ecuador': 'South America',
    'United Kingdom': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe', 'France': 'Europe',
    'Germany': 'Europe', 'Russia': 'Europe', 'Turkey': 'Europe', 'Netherlands': 'Europe',
    'Belgium': 'Europe', 'Switzerland': 'Europe', 'Sweden': 'Europe', 'Portugal': 'Europe',
    'Austria': 'Europe', 'Poland': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe',
    'Ireland': 'Europe', 'Czech Republic': 'Europe', 'Romania': 'Europe',
    'Iran': 'Middle East', 'Saudi Arabia': 'Middle East', 'United Arab Emirates': 'Middle East',
    'Israel': 'Middle East', 'Qatar': 'Middle East', 'Kuwait': 'Middle East', 'Bahrain': 'Middle East',
    'Oman': 'Middle East', 'Iraq': 'Middle East', 'Lebanon': 'Middle East',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Algeria': 'Africa', 'Morocco': 'Africa',
    'Nigeria': 'Africa', 'Senegal': 'Africa', 'Cameroon': 'Africa',
    'Thailand': 'Southeast Asia', 'Malaysia': 'Southeast Asia', 'Singapore': 'Southeast Asia',
    'Indonesia': 'Southeast Asia', 'Philippines': 'Southeast Asia', 'Vietnam': 'Southeast Asia',
    'Cambodia': 'Southeast Asia', 'Myanmar': 'Southeast Asia', 'Laos': 'Southeast Asia'
}

global_summary['Region'] = global_summary['Country/Region'].map(region_mapping).fillna('Other')
regional_summary = global_summary.groupby('Region').agg({
    'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 'Active': 'sum'
}).reset_index()

# ============== TOP COUNTRIES ==============
top20_confirmed = global_summary.nlargest(20, 'Confirmed')[
    ['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality Rate (%)', 'Recovery Rate (%)']
]

country_daily = df.groupby(['ObservationDate', 'Country/Region']).agg({
    'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'
}).reset_index()

countries = sorted(df['Country/Region'].unique())

# ============== DASHBOARD LAYOUT ==============
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])
app.title = "COVID-19 Intelligence Dashboard"

# Modern Color Scheme
colors = {
    'bg': '#0f172a',
    'card_bg': '#1e293b',
    'header_bg': '#0f172a',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'confirmed': '#ef4444',
    'active': '#f59e0b',
    'deaths': '#b91c1c',
    'recovered': '#10b981',
    'accent': '#3b82f6',
    'border': '#334155'
}

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>COVID-19 Dashboard</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Inter', sans-serif; }
            body { background-color: #0f172a; }
            .card-stats {
                border-radius: 16px;
                padding: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
                border: 1px solid #334155;
            }
            .card-stats:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .dashboard-title {
                font-weight: 700;
                background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .section-header {
                font-weight: 600;
                letter-spacing: -0.025em;
                border-left: 4px solid #3b82f6;
                padding-left: 16px;
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                line-height: 1.2;
            }
            .stat-label {
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #94a3b8;
            }
            .stat-change {
                font-size: 0.75rem;
                margin-top: 8px;
            }
            .dropdown-custom .Select-control {
                background-color: #1e293b !important;
                border: 1px solid #334155 !important;
                border-radius: 12px !important;
            }
            .dropdown-custom .Select-menu-outer {
                background-color: #1e293b !important;
                border: 1px solid #334155 !important;
            }
            .dropdown-custom .Select-option {
                background-color: #1e293b !important;
                color: #f1f5f9 !important;
            }
            .dropdown-custom .Select-option:hover {
                background-color: #334155 !important;
            }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #1e293b; border-radius: 10px; }
            ::-webkit-scrollbar-thumb { background: #475569; border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: #64748b; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============== COMPONENTS ==============
def create_stat_card(title, value, color, icon, trend=None):
    return dbc.Card([
        html.Div([
            html.Div([
                html.I(className=f"bi bi-{icon} fs-4", style={'color': color, 'opacity': 0.7}),
                html.P(title, className="stat-label mt-2"),
                html.H2(f"{value:,.0f}" if isinstance(value, (int, float)) else value, 
                       className="stat-value", style={'color': '#f1f5f9'}),
                html.Small(trend, className="stat-change", style={'color': '#10b981'}) if trend else None
            ])
        ], className="p-2")
    ], className="card-stats", style={'backgroundColor': colors['card_bg']})

def create_section_header(title, icon):
    return html.Div([
        html.I(className=f"bi bi-{icon} me-2", style={'color': colors['accent']}),
        html.H3(title, className="section-header", style={'color': colors['text']})
    ], className="d-flex align-items-center mb-3")

# Summary Stats Row
mortality_rate = (total_deaths/total_confirmed*100) if total_confirmed > 0 else 0
recovery_rate = (total_recovered/total_confirmed*100) if total_confirmed > 0 else 0

summary_row = dbc.Row([
    dbc.Col(create_stat_card("Total Confirmed", total_confirmed, colors['confirmed'], "virus", f"+{daily_global['New Cases'].iloc[-1]:,.0f} today"), width=3),
    dbc.Col(create_stat_card("Active Cases", total_active, colors['active'], "activity"), width=3),
    dbc.Col(create_stat_card("Total Deaths", total_deaths, colors['deaths'], "heart-pulse", f"{mortality_rate:.2f}% mortality"), width=3),
    dbc.Col(create_stat_card("Total Recovered", total_recovered, colors['recovered'], "heart-fill", f"{recovery_rate:.2f}% recovery"), width=3),
], className="g-3 mb-4")

# Secondary Stats Row
secondary_stats = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-globe-americas me-2", style={'color': colors['accent']}),
                html.Span("Countries Affected", className="text-secondary")
            ], className="mb-2"),
            html.H3(f"{len(countries)}", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats", style={'backgroundColor': colors['card_bg']}), width=2),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-calendar-check me-2", style={'color': colors['accent']}),
                html.Span("Last Updated", className="text-secondary")
            ], className="mb-2"),
            html.H5(f"{latest_date.strftime('%b %d, %Y')}", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats", style={'backgroundColor': colors['card_bg']}), width=2),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-calendar-range me-2", style={'color': colors['accent']}),
                html.Span("Data Range", className="text-secondary")
            ], className="mb-2"),
            html.H6(f"{earliest_date.strftime('%b %d')} - {latest_date.strftime('%b %d, %Y')}", 
                   className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats", style={'backgroundColor': colors['card_bg']}), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-people-fill me-2", style={'color': colors['accent']}),
                html.Span("Global Population Affected", className="text-secondary")
            ], className="mb-2"),
            html.H5(f"{(total_confirmed/7800000000*100):.3f}%", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats", style={'backgroundColor': colors['card_bg']}), width=3),
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-graph-up me-2", style={'color': colors['accent']}),
                html.Span("Avg Daily New Cases", className="text-secondary")
            ], className="mb-2"),
            html.H5(f"{daily_global['New Cases'].tail(7).mean():,.0f}", className="mb-0", style={'color': colors['text']})
        ])
    ], className="card-stats", style={'backgroundColor': colors['card_bg']}), width=2),
], className="g-3 mb-4")

# App Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("COVID-19 Intelligence Dashboard", className="dashboard-title display-6 fw-bold"),
                html.P("Real-time tracking and analysis of global pandemic data", 
                       className="text-secondary", style={'fontSize': '1rem'})
            ], className="py-3")
        ], width=12)
    ]),
    
    summary_row,
    secondary_stats,
    
    # Main Charts Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Global Pandemic Timeline", "graph-up")),
                dbc.CardBody([dcc.Graph(id='global-timeline', config={'displayModeBar': False})])
            ], className="card-stats h-100", style={'backgroundColor': colors['card_bg']})
        ], width=12)
    ], className="mb-4"),
    
    # Regional Analysis Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Regional Distribution", "pie-chart")),
                dbc.CardBody([dcc.Graph(id='regional-pie', config={'displayModeBar': False})])
            ], className="card-stats h-100", style={'backgroundColor': colors['card_bg']})
        ], width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Cases by Region", "bar-chart")),
                dbc.CardBody([dcc.Graph(id='regional-bar', config={'displayModeBar': False})])
            ], className="card-stats h-100", style={'backgroundColor': colors['card_bg']})
        ], width=7),
    ], className="mb-4"),
    
    # Heatmap Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("New Cases Heatmap (Last 30 Days)", "grid-3x3")),
                dbc.CardBody([dcc.Graph(id='heatmap', config={'displayModeBar': False})])
            ], className="card-stats", style={'backgroundColor': colors['card_bg']})
        ], width=12)
    ], className="mb-4"),
    
    # Top Countries Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Top 20 Most Affected Countries", "trophy")),
                dbc.CardBody([dcc.Graph(id='top20-bar', config={'displayModeBar': False})])
            ], className="card-stats", style={'backgroundColor': colors['card_bg']})
        ], width=12)
    ], className="mb-4"),
    
    # Country Comparison Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(create_section_header("Country Comparison Analysis", "arrow-left-right")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Countries", className="fw-semibold mb-2", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='country-selector',
                                options=[{'label': c, 'value': c} for c in countries],
                                value=['United States', 'Brazil', 'India', 'Russia', 'United Kingdom'],
                                multi=True,
                                className="dropdown-custom mb-3"
                            )
                        ], width=8),
                        dbc.Col([
                            html.Label("Metric", className="fw-semibold mb-2", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='metric-selector',
                                options=[
                                    {'label': 'Confirmed', 'value': 'Confirmed'},
                                    {'label': 'Deaths', 'value': 'Deaths'},
                                    {'label': 'Recovered', 'value': 'Recovered'}
                                ],
                                value='Confirmed',
                                inline=True,
                                className="text-white",
                                inputStyle={"margin-right": "5px", "margin-left": "15px"},
                                labelStyle={'color': colors['text_secondary']}
                            )
                        ], width=4)
                    ]),
                    dcc.Graph(id='country-comparison', config={'displayModeBar': False})
                ])
            ], className="card-stats", style={'backgroundColor': colors['card_bg']})
        ], width=12)
    ], className="mb-4"),
    
    # Footer
    html.Footer([
        html.Hr(style={'borderColor': colors['border']}),
        dbc.Row([
            dbc.Col(html.P("📊 Data Source: Johns Hopkins University CSSE • Updated Daily", 
                          className="text-center", style={'color': colors['text_secondary']}), width=12),
            dbc.Col(html.P(f"Dashboard Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
                          className="text-center", style={'color': colors['text_secondary'], 'fontSize': '0.85rem'}), width=12)
        ])
    ], className="mt-4 pb-3")
    
], fluid=True, style={'backgroundColor': colors['bg'], 'minHeight': '100vh', 'padding': '24px'})

# ============== CALLBACKS ==============
@app.callback(
    Output('global-timeline', 'figure'),
    Input('global-timeline', 'id')
)
def update_global_timeline(_):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=daily_global['ObservationDate'], y=daily_global['New Cases'],
        name='Daily New Cases', marker_color=colors['confirmed'], opacity=0.6
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=daily_global['ObservationDate'], y=daily_global['Confirmed'],
        name='Cumulative Confirmed', line=dict(color=colors['confirmed'], width=3), mode='lines'
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=daily_global['ObservationDate'], y=daily_global['Deaths'],
        name='Cumulative Deaths', line=dict(color=colors['deaths'], width=2.5), mode='lines'
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=daily_global['ObservationDate'], y=daily_global['Recovered'],
        name='Cumulative Recovered', line=dict(color=colors['recovered'], width=2.5), mode='lines'
    ), secondary_y=True)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=colors['card_bg'],
        plot_bgcolor=colors['card_bg'],
        font={'color': colors['text']},
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
        height=400, margin=dict(l=40, r=40, t=20, b=40),
        dragmode='pan'
    )
    
    fig.update_xaxes(title_text="", gridcolor=colors['border'])
    fig.update_yaxes(title_text="Daily New Cases", secondary_y=False, gridcolor=colors['border'])
    fig.update_yaxes(title_text="Cumulative Cases", secondary_y=True, gridcolor=colors['border'])
    
    return fig

@app.callback(Output('regional-pie', 'figure'), Input('regional-pie', 'id'))
def update_regional_pie(_):
    fig = px.pie(regional_summary, values='Confirmed', names='Region', hole=0.5,
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=colors['card_bg'], plot_bgcolor=colors['card_bg'],
        font={'color': colors['text']}, height=350, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
    return fig

@app.callback(Output('regional-bar', 'figure'), Input('regional-bar', 'id'))
def update_regional_bar(_):
    fig = go.Figure()
    for col, color, name in [('Active', colors['active'], 'Active'), 
                              ('Recovered', colors['recovered'], 'Recovered'),
                              ('Deaths', colors['deaths'], 'Deaths')]:
        fig.add_trace(go.Bar(x=regional_summary['Region'], y=regional_summary[col], name=name, marker_color=color))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=colors['card_bg'], plot_bgcolor=colors['card_bg'],
        font={'color': colors['text']}, barmode='group', height=350,
        xaxis_title="", yaxis_title="Number of Cases", margin=dict(l=40, r=20, t=30, b=40),
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
        colorscale=[[0, '#1e293b'], [0.5, '#f59e0b'], [1, '#ef4444']],
        hoverongaps=False, colorbar=dict(title="Daily New Cases", len=0.8)
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=colors['card_bg'], plot_bgcolor=colors['card_bg'],
        font={'color': colors['text']}, height=400, xaxis_title="", yaxis_title="",
        margin=dict(l=120, r=40, t=20, b=40)
    )
    return fig

@app.callback(Output('top20-bar', 'figure'), Input('top20-bar', 'id'))
def update_top20_bar(_):
    fig = go.Figure()
    top20_sorted = top20_confirmed.sort_values('Confirmed', ascending=True)
    
    fig.add_trace(go.Bar(y=top20_sorted['Country/Region'], x=top20_sorted['Active'], 
                         name='Active', orientation='h', marker_color=colors['active']))
    fig.add_trace(go.Bar(y=top20_sorted['Country/Region'], x=top20_sorted['Recovered'], 
                         name='Recovered', orientation='h', marker_color=colors['recovered']))
    fig.add_trace(go.Bar(y=top20_sorted['Country/Region'], x=top20_sorted['Deaths'], 
                         name='Deaths', orientation='h', marker_color=colors['deaths']))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=colors['card_bg'], plot_bgcolor=colors['card_bg'],
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
    colors_list = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
    
    for i, country in enumerate(selected_countries):
        country_data = country_daily[country_daily['Country/Region'] == country].copy()
        if len(country_data) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=country_data['ObservationDate'], y=country_data[selected_metric],
            name=country, line=dict(width=2.5, color=colors_list[i % len(colors_list)]), mode='lines'
        ))
    
    metric_labels = {'Confirmed': 'Confirmed Cases', 'Deaths': 'Deaths', 'Recovered': 'Recovered Cases'}
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=colors['card_bg'], plot_bgcolor=colors['card_bg'],
        font={'color': colors['text']}, xaxis_title="", yaxis_title=metric_labels[selected_metric],
        height=400, hovermode='x unified', margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='left', x=1.01, bgcolor='rgba(0,0,0,0)'),
        xaxis={'gridcolor': colors['border']}, yaxis={'gridcolor': colors['border']}
    )
    return fig

# ============== RUN ==============
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port=8053)