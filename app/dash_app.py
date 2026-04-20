"""KhazinaSmart — Grocery Intelligence Dashboard"""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dash import Dash, dcc, html, Input, Output, State, ctx, ALL, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64, io, warnings
warnings.filterwarnings("ignore")

from src.data_adapter import detect_columns, standardize, build_model_features, FEATURE_COLS
from src.universal_model import UniversalForecastModel
from src.alerts import generate_alerts_dataframe, estimate_financial_impact
from src.chatbot import get_starter_questions, answer_inventory_question

PALETTE = ["#6c63ff","#e94560","#10b981","#f59e0b","#3b82f6","#ec4899","#8b5cf6","#14b8a6"]

def _tmpl(theme):
    return "plotly_dark" if theme == "dark" else "plotly_white"

def _base_layout(theme, title="", h=300):
    return dict(
        template=_tmpl(theme),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        height=h,
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=12, r=12, t=44, b=12),
    )

def _sc(df):
    return "revenue" if "revenue" in df.columns else "sales"

def _empty_fig(theme, msg="Upload a CSV file to get started"):
    fig = go.Figure()
    fig.update_layout(
        **_base_layout(theme, h=260),
        annotations=[dict(text=msg, showarrow=False, x=.5, y=.5,
                          xref="paper", yref="paper",
                          font=dict(color="#8b8db8", size=14))],
    )
    return fig

def _upload_prompt(msg="Upload your CSV using the button above"):
    return html.Div([
        html.Div("📂", style={"fontSize":"40px","marginBottom":"12px"}),
        html.Div(msg, style={"color":"var(--c-text2)","fontSize":"15px"}),
    ], style={"textAlign":"center","padding":"60px 20px",
              "background":"var(--c-card)","borderRadius":"14px",
              "border":"1.5px dashed var(--c-border2)"})

def _read_df(json_str):
    if not json_str:
        return None
    try:
        df = pd.read_json(io.StringIO(json_str), orient="split")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return None

# ─── CHART FUNCTIONS ──────────────────────────────────────────────────────────

def chart_trend(df, theme):
    sc = _sc(df)
    agg = df.copy()
    agg["date"] = pd.to_datetime(agg["date"])
    agg = agg.groupby("date")[sc].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg[sc], mode="lines",
        name="Revenue", line=dict(color="#6c63ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.07)"))
    if "is_promoted" in df.columns:
        p = df[df["is_promoted"]==1].copy()
        p["date"] = pd.to_datetime(p["date"])
        p = p.groupby("date")[sc].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=p["date"], y=p[sc], mode="markers", name="Promo weeks",
            marker=dict(color="#e94560", size=6, symbol="diamond")))
    fig.update_layout(**_base_layout(theme, "Total Weekly Revenue Trend", 290),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig

def chart_category(df, theme):
    sc = _sc(df)
    agg = df.groupby("category")[sc].sum().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=agg.values, y=agg.index, orientation="h",
        marker=dict(color=agg.values,
                    colorscale=[[0,"#1a1a32"],[.5,"#6c63ff"],[1,"#e94560"]])))
    fig.update_layout(**_base_layout(theme, "Revenue by Category", 270))
    return fig

def chart_store(df, theme):
    sc = _sc(df)
    if "store_id" not in df.columns:
        return _empty_fig(theme)
    agg = df.groupby("store_id")[sc].sum().reset_index().sort_values(sc, ascending=False)
    fig = go.Figure(go.Bar(
        x=agg["store_id"], y=agg[sc],
        marker=dict(color=PALETTE[:len(agg)]),
        text=agg[sc].apply(lambda v: f"{v/1000:.0f}K"), textposition="outside"))
    fig.update_layout(**_base_layout(theme, "Revenue by Store", 270))
    return fig

def chart_heatmap(df, theme):
    sc = _sc(df)
    if "category" not in df.columns or "store_id" not in df.columns:
        return _empty_fig(theme)
    pivot = df.pivot_table(index="category", columns="store_id",
                           values=sc, aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, aspect="auto",
                    color_continuous_scale="Purples" if theme=="dark" else "Blues")
    fig.update_layout(**_base_layout(theme, "Heatmap: Category × Store", 300))
    return fig

def chart_seasonal(df, theme):
    sc = _sc(df)
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["date"]).dt.month
    m = df2.groupby("month")[sc].mean().reset_index()
    mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Bar(
        x=[mn[i-1] for i in m["month"]], y=m[sc],
        marker=dict(color=m[sc],
                    colorscale=[[0,"#1a1a32"],[.5,"#6c63ff"],[1,"#e94560"]])))
    fig.update_layout(**_base_layout(theme, "Monthly Seasonality", 250))
    return fig

def build_forecast_fig(hist_df, fut_df, store, cat, theme):
    """Build forecast chart — hist_df is standardized, fut_df already filtered."""
    sc  = _sc(hist_df)
    fig = go.Figure()

    # Historical line
    h = hist_df.copy()
    h["date"] = pd.to_datetime(h["date"])
    if store and "store_id" in h.columns:
        h = h[h["store_id"].astype(str) == str(store)]
    if cat and "category" in h.columns:
        h = h[h["category"].astype(str) == str(cat)]
    agg = h.groupby("date")[sc].sum().reset_index().sort_values("date")

    if not agg.empty:
        fig.add_trace(go.Scatter(
            x=agg["date"], y=agg[sc],
            name="Historical", mode="lines",
            line=dict(color="#6c63ff", width=2.5)))

    # Future prediction
    if fut_df is not None and not fut_df.empty:
        fut = fut_df.copy().sort_values("date")
        fut["date"] = pd.to_datetime(fut["date"])

        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(fut["date"]) + list(fut["date"][::-1]),
            y=list(fut["upper"]) + list(fut["lower"][::-1]),
            fill="toself", fillcolor="rgba(233,69,96,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band", showlegend=True))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["predicted"],
            name="AI Forecast", mode="lines+markers",
            line=dict(color="#e94560", width=2.5, dash="dash"),
            marker=dict(size=7, color="#e94560")))

        if not agg.empty:
            max_date = agg["date"].max()
            fig.add_shape(type="line",
                x0=max_date, x1=max_date, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#f59e0b", width=1.5, dash="dot"))
            fig.add_annotation(
                x=max_date, y=0.98, xref="x", yref="paper",
                text="Forecast start", showarrow=False,
                font=dict(color="#f59e0b", size=11),
                xanchor="left", xshift=6)

    title = f"AI Forecast — {store} / {cat}" if store and cat else "AI Demand Forecast"
    fig.update_layout(
        **_base_layout(theme, title, 420),
        legend=dict(orientation="h", y=1.10),
        xaxis_title="Date",
        yaxis_title="Revenue (MAD)")
    return fig

# ─── APP ──────────────────────────────────────────────────────────────────────
app = Dash(__name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="KhazinaSmart")

# ─── LAYOUT HELPERS ───────────────────────────────────────────────────────────
def kpi_card(icon, label, val_id, sub, accent):
    return html.Div([
        html.Div(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div("—", id=val_id, className="kpi-value"),
        html.Div(sub, className="kpi-sub"),
    ], className=f"kpi-card accent-{accent}")

def page_overview():
    return html.Div([
        html.Div([
            kpi_card("📦","Total Revenue",  "kpi-rev",    "All stores",     "purple"),
            kpi_card("🏪","Active Stores",  "kpi-stores", "In dataset",     "green"),
            kpi_card("🗂","Categories",     "kpi-cats",   "Product groups", "yellow"),
            kpi_card("🔴","Overstock Alerts","kpi-over",  "Needs action",   "red"),
        ], className="kpi-grid"),
        html.Div(id="ov-charts"),
        html.Div(id="ov-preview", style={"marginTop":"20px"}),
    ], id="pg-overview", style={"display":"block"})

def page_forecast():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Store"),
                dcc.Dropdown(id="fc-store", placeholder="Select store…",
                             style={"fontSize":"13px","minWidth":"180px"}),
            ], className="fc-field"),
            html.Div([
                html.Label("Category"),
                dcc.Dropdown(id="fc-cat", placeholder="Select category…",
                             style={"fontSize":"13px","minWidth":"180px"}),
            ], className="fc-field"),
            html.Div([
                html.Label("Weeks ahead"),
                dcc.Slider(id="fc-weeks", min=4, max=16, step=4, value=8,
                           marks={4:"4w",8:"8w",12:"12w",16:"16w"}),
            ], className="fc-field", style={"minWidth":"220px"}),
            html.Button("Generate Forecast", id="fc-run", n_clicks=0,
                        className="btn-primary",
                        style={"height":"38px","alignSelf":"flex-end"}),
        ], className="forecast-controls"),

        dcc.Loading(type="circle", color="#6c63ff", children=[
            html.Div([
                html.Div([
                    html.Div([html.Div("AI Demand Forecast", className="chart-title")],
                             className="chart-header"),
                    html.Div(id="fc-graph-wrap",
                             children=dcc.Graph(id="ch-fc",
                                                config={"displayModeBar":False},
                                                figure=_empty_fig("dark","Click Generate Forecast after uploading CSV")),
                             className="chart-body"),
                ], className="chart-card", style={"marginBottom":"16px"}),
            ]),
        ]),
        html.Div(id="fc-table"),
    ], id="pg-forecast", style={"display":"none"})

def page_alerts():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Status"),
                dcc.Dropdown(id="al-status",
                    options=[{"label":l,"value":v} for l,v in [
                        ("All","All"),("🔴 Overstock","Overstock"),
                        ("🟡 Stockout Risk","Stockout Risk"),("🟢 Healthy","Healthy")]],
                    value="All", clearable=False,
                    style={"fontSize":"13px","minWidth":"170px"}),
            ], className="fc-field"),
            html.Div([
                html.Label("Min risk score"),
                dcc.Slider(id="al-risk", min=0, max=100, step=10, value=0,
                           marks={0:"0",50:"50",100:"100"}),
            ], className="fc-field", style={"minWidth":"200px"}),
            html.Div(id="al-kpis",
                     style={"display":"flex","gap":"10px","alignItems":"center","flexWrap":"wrap"}),
        ], className="filter-bar"),
        html.Div(id="al-table"),
        html.Div(id="al-chart", style={"marginTop":"14px"}),
    ], id="pg-alerts", style={"display":"none"})

# ─── FLOATING CHATBOT ─────────────────────────────────────────────────────────
def floating_chatbot():
    return html.Div([
        # Toggle button
        html.Button("💬", id="float-btn", n_clicks=0, className="float-chat-btn",
                    title="KhazBot — Inventory Assistant"),

        # Chat panel (hidden by default)
        html.Div([
            # Panel header
            html.Div([
                html.Div([
                    html.Span("📦", style={"fontSize":"18px"}),
                    html.Div([
                        html.Div("KhazBot", style={"fontWeight":"700","fontSize":"14px","color":"var(--c-text)"}),
                        html.Div("AI Inventory Assistant", style={"fontSize":"11px","color":"var(--c-text2)"}),
                    ]),
                ], style={"display":"flex","alignItems":"center","gap":"10px"}),
                html.Button("✕", id="float-close", n_clicks=0, className="float-close-btn"),
            ], className="float-panel-header"),

            # Starter questions
            html.Div([
                html.Button(q.split(" 🔴")[0].split(" 📦")[0].split(" 💰")[0].split(" 🏪")[0].split(" 📊")[0],
                            id={"type":"fsq","index":i}, n_clicks=0, className="sq-btn")
                for i, q in enumerate(get_starter_questions())
            ], className="float-starters"),

            # Messages
            html.Div([
                html.Div([
                    html.Div("📦", className="chat-avatar bot-av"),
                    html.Div([
                        html.Strong("KhazBot"),
                        html.Br(),
                        html.Span("Bonjour! Ask about your inventory in French or English."),
                    ], className="chat-bubble bubble-bot"),
                ], className="chat-msg"),
            ], id="float-msgs", className="float-messages"),

            # Input
            html.Div([
                dcc.Input(id="float-in", placeholder="Ask about inventory…",
                          type="text", className="chat-input", debounce=False,
                          style={"flex":"1"}),
                html.Button("Send", id="float-go", n_clicks=0, className="chat-send"),
            ], className="chat-input-row"),
        ], id="float-panel", className="float-panel", style={"display":"none"}),
    ], id="float-chat-wrap")

# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="st-theme",  data="dark",     storage_type="session"),
    dcc.Store(id="st-data",   data=None,       storage_type="memory"),
    dcc.Store(id="st-preds",  data=None,       storage_type="memory"),
    dcc.Store(id="st-alerts", data=None,       storage_type="memory"),
    dcc.Store(id="st-chat",   data=[],         storage_type="memory"),
    dcc.Store(id="st-tab",    data="overview", storage_type="memory"),

    html.Div([
        # HEADER
        html.Header([
            html.Div([
                html.Span("📦", className="header-brand-icon"),
                html.Span("KhazinaSmart", className="header-brand-name"),
                html.Span("BETA", className="header-brand-tag"),
            ], className="header-brand"),
            html.Nav([
                html.Button("Overview", id="tb-ov", n_clicks=0, className="nav-tab active"),
                html.Button("Forecast", id="tb-fc", n_clicks=0, className="nav-tab"),
                html.Button("Alerts",   id="tb-al", n_clicks=0, className="nav-tab"),
            ], className="header-nav"),
            html.Div([
                dcc.Upload(id="up-csv",
                    children=html.Button("⬆ Upload CSV", className="btn-upload"),
                    accept=".csv", multiple=False),
                html.Button("🌙", id="btn-theme", n_clicks=0, className="btn-theme",
                            title="Toggle dark/light mode"),
            ], className="header-actions"),
        ], className="site-header"),

        html.Div([
            dcc.Loading(type="dot", color="#6c63ff", children=[
                html.Div(id="status-bar", children=html.Div([
                    html.Div("📂", style={"fontSize":"18px"}),
                    html.Span("No data loaded — upload your CSV to get started",
                              style={"fontWeight":600,"color":"var(--c-text)"}),
                    html.Span("Accepted: any CSV with date + sales + category columns",
                              className="metric-chip"),
                    html.Span("Test file: data/sample_grocery.csv",
                              className="status-badge badge-transfer"),
                ], className="status-bar")),
            ]),

            page_overview(),
            page_forecast(),
            page_alerts(),

            html.Div("KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business",
                     className="site-footer"),
        ], className="page-body"),

        # FLOATING CHATBOT — always visible on all pages
        floating_chatbot(),

    ], id="app-root", className="theme-dark"),
])

# ─── CALLBACKS ────────────────────────────────────────────────────────────────

# 1. THEME
@app.callback(
    Output("app-root",  "className"),
    Output("btn-theme", "children"),
    Output("st-theme",  "data"),
    Input("btn-theme",  "n_clicks"),
    State("st-theme",   "data"),
)
def toggle_theme(n, current):
    theme = current or "dark"
    if n:
        theme = "light" if theme == "dark" else "dark"
    return f"theme-{theme}", ("☀️" if theme=="light" else "🌙"), theme


# 2. TAB NAV
@app.callback(
    Output("st-tab", "data"),
    Output("tb-ov",  "className"),
    Output("tb-fc",  "className"),
    Output("tb-al",  "className"),
    Input("tb-ov",   "n_clicks"),
    Input("tb-fc",   "n_clicks"),
    Input("tb-al",   "n_clicks"),
    State("st-tab",  "data"),
    prevent_initial_call=True,
)
def switch_tab(n1, n2, n3, current):
    tid = ctx.triggered_id
    tab = {"tb-ov":"overview","tb-fc":"forecast","tb-al":"alerts"}.get(tid, current)
    def c(t): return "nav-tab active" if tab==t else "nav-tab"
    return tab, c("overview"), c("forecast"), c("alerts")


# 3. TAB VISIBILITY
@app.callback(
    Output("pg-overview", "style"),
    Output("pg-forecast", "style"),
    Output("pg-alerts",   "style"),
    Input("st-tab",       "data"),
)
def show_tab(tab):
    s, h = {"display":"block"}, {"display":"none"}
    tab = tab or "overview"
    return (s if tab=="overview" else h,
            s if tab=="forecast" else h,
            s if tab=="alerts"   else h)


# 4. FLOATING CHAT PANEL TOGGLE
@app.callback(
    Output("float-panel", "style"),
    Input("float-btn",   "n_clicks"),
    Input("float-close", "n_clicks"),
    State("float-panel", "style"),
    prevent_initial_call=True,
)
def toggle_float(n_open, n_close, current):
    tid = ctx.triggered_id
    if tid == "float-close":
        return {"display":"none"}
    visible = current.get("display","none") == "flex" if current else False
    return {"display":"none" if visible else "flex"}


# 5. UPLOAD — parse CSV + train model + store predictions
@app.callback(
    Output("st-data",    "data"),
    Output("st-preds",   "data"),
    Output("st-alerts",  "data"),
    Output("status-bar", "children"),
    Input("up-csv",      "contents"),
    State("up-csv",      "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        raise PreventUpdate

    try:
        _, b64 = contents.split(",", 1)
        raw = pd.read_csv(io.StringIO(base64.b64decode(b64).decode("utf-8")))
    except Exception as e:
        return no_update, no_update, no_update, html.Div(
            f"Error reading file: {e}", style={"color":"#e94560","padding":"10px"})

    try:
        col_map = detect_columns(raw)
        std = standardize(raw, col_map)
    except Exception as e:
        return no_update, no_update, no_update, html.Div(
            f"Cannot detect columns: {e}", style={"color":"#e94560","padding":"10px"})

    data_json = std.to_json(orient="split", date_format="iso")
    n_rows    = len(std)
    n_stores  = std["store_id"].nunique() if "store_id" in std.columns else "—"
    n_cats    = std["category"].nunique() if "category" in std.columns else "—"

    preds_json  = None
    alerts_json = None
    model_info  = "Model training…"

    try:
        model   = UniversalForecastModel()
        metrics = model.fit(std, n_transfer_trees=60)
        future  = model.predict_future(std, weeks=12)
        preds_json = future.to_json(orient="split", date_format="iso")

        # Build alerts
        test_preds = model.get_predictions()
        if not test_preds.empty and "predicted" in test_preds.columns:
            ap = test_preds.rename(columns={
                "sales":"Weekly_Sales", "predicted":"predicted_demand",
                "store_id":"Store", "category":"Dept", "date":"Date"})
            al = generate_alerts_dataframe(ap)
            alerts_json = al.to_json(orient="split", date_format="iso")

        tl   = metrics.get("transfer_learning", False)
        mape = metrics.get("mape", 0)
        r2   = metrics.get("r2", 0)
        n_fc = len(future) if future is not None else 0
        model_info = (f"MAPE={mape:.1f}%  R²={r2:.3f}  "
                      f"{'Transfer Learning ✓' if tl else 'Fresh model'}  "
                      f"| {n_fc} forecast rows ready")
    except Exception as e:
        model_info = f"Model warning: {e}"

    bar = html.Div([
        html.Div(className="status-dot"),
        html.Span(f"✓  {filename}", style={"fontWeight":600,"color":"var(--c-text)"}),
        html.Span(f"{n_rows:,} rows",                         className="status-badge badge-rows"),
        html.Span(f"{n_stores} stores · {n_cats} categories", className="metric-chip"),
        html.Span(model_info,                                 className="status-badge badge-transfer"),
    ], className="status-bar")

    return data_json, preds_json, alerts_json, bar


# 6. KPIs
@app.callback(
    Output("kpi-rev",    "children"),
    Output("kpi-stores", "children"),
    Output("kpi-cats",   "children"),
    Output("kpi-over",   "children"),
    Input("st-data",     "data"),
    Input("st-alerts",   "data"),
)
def update_kpis(data_json, alerts_json):
    df = _read_df(data_json)
    if df is None:
        return "—", "—", "—", "—"
    sc     = _sc(df)
    total  = df[sc].sum()
    stores = df["store_id"].nunique() if "store_id" in df.columns else "—"
    cats   = df["category"].nunique() if "category" in df.columns else "—"
    rev    = f"{total/1e6:.1f}M" if total>=1e6 else f"{total/1000:.0f}K"
    ov     = "—"
    al     = _read_df(alerts_json)
    if al is not None and "status" in al.columns:
        ov = str(int((al["status"]=="Overstock").sum()))
    return rev, str(stores), str(cats), ov


# 7. OVERVIEW CHARTS + DATA PREVIEW
@app.callback(
    Output("ov-charts",  "children"),
    Output("ov-preview", "children"),
    Input("st-data",     "data"),
    Input("st-theme",    "data"),
)
def update_overview(data_json, theme):
    t  = theme or "dark"
    df = _read_df(data_json)
    if df is None:
        return _upload_prompt(), None

    charts = html.Div([
        html.Div([
            html.Div([
                html.Div([html.Div("Weekly Revenue Trend", className="chart-title")], className="chart-header"),
                html.Div(dcc.Graph(figure=chart_trend(df, t), config={"displayModeBar":False}), className="chart-body"),
            ], className="chart-card"),
        ], className="chart-grid-1"),
        html.Div([
            html.Div([
                html.Div([html.Div("Revenue by Category", className="chart-title")], className="chart-header"),
                html.Div(dcc.Graph(figure=chart_category(df, t), config={"displayModeBar":False}), className="chart-body"),
            ], className="chart-card"),
            html.Div([
                html.Div([html.Div("Revenue by Store", className="chart-title")], className="chart-header"),
                html.Div(dcc.Graph(figure=chart_store(df, t), config={"displayModeBar":False}), className="chart-body"),
            ], className="chart-card"),
        ], className="chart-grid-2"),
        html.Div([
            html.Div([
                html.Div([html.Div("Category × Store Heatmap", className="chart-title")], className="chart-header"),
                html.Div(dcc.Graph(figure=chart_heatmap(df, t), config={"displayModeBar":False}), className="chart-body"),
            ], className="chart-card"),
            html.Div([
                html.Div([html.Div("Monthly Seasonality", className="chart-title")], className="chart-header"),
                html.Div(dcc.Graph(figure=chart_seasonal(df, t), config={"displayModeBar":False}), className="chart-body"),
            ], className="chart-card"),
        ], className="chart-grid-2"),
    ])

    preview_cols = [c for c in df.columns if c not in ("store_code","category_code")][:7]
    preview_df   = df[preview_cols].head(10)
    preview = html.Div([
        html.Div([
            html.Div("Data Preview — first 10 rows", className="chart-title"),
            html.Span(f"{len(df):,} total rows · {len(df.columns)} columns", className="chart-sub"),
        ], className="chart-header"),
        html.Div([
            html.Table([
                html.Thead(html.Tr([html.Th(c.replace("_"," ").title()) for c in preview_cols])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            row[c].strftime("%Y-%m-%d") if hasattr(row[c], "strftime")
                            else str(row[c])[:30]
                        ) for c in preview_cols
                    ])
                    for _, row in preview_df.iterrows()
                ]),
            ], className="data-table"),
        ], style={"overflowX":"auto","padding":"0 0 12px 0"}),
    ], className="chart-card", style={"marginTop":"4px"})

    return charts, preview


# 8. FORECAST DROPDOWNS
@app.callback(
    Output("fc-store", "options"), Output("fc-store", "value"),
    Output("fc-cat",   "options"), Output("fc-cat",   "value"),
    Input("st-data",   "data"),
)
def fc_dropdowns(data_json):
    df = _read_df(data_json)
    if df is None:
        return [], None, [], None
    stores = sorted(df["store_id"].unique().tolist()) if "store_id" in df.columns else []
    cats   = sorted(df["category"].unique().tolist()) if "category" in df.columns else []
    return ([{"label":s,"value":s} for s in stores], stores[0] if stores else None,
            [{"label":c,"value":c} for c in cats],   cats[0]   if cats   else None)


# 9. FORECAST — reads pre-computed predictions
@app.callback(
    Output("ch-fc",    "figure"),
    Output("fc-table", "children"),
    Input("fc-run",    "n_clicks"),
    State("st-data",   "data"),
    State("st-preds",  "data"),
    State("fc-store",  "value"),
    State("fc-cat",    "value"),
    State("fc-weeks",  "value"),
    State("st-theme",  "data"),
    prevent_initial_call=True,
)
def show_forecast(n, data_json, preds_json, store, cat, weeks, theme):
    if not n:
        raise PreventUpdate
    t = theme or "dark"

    try:
        df  = _read_df(data_json)
        fut = _read_df(preds_json)

        if df is None:
            return _empty_fig(t, "Upload a CSV file first"), _upload_prompt()

        if fut is None or fut.empty:
            return (_empty_fig(t, "No predictions — try re-uploading the CSV"),
                    html.Div("No prediction data.", style={"color":"var(--c-text2)","padding":"12px"}))

        weeks = int(weeks or 8)

        # Filter predictions for selected store + category
        fut_f = fut.copy()
        if store:
            fut_f = fut_f[fut_f["store_id"].astype(str) == str(store)]
        if cat:
            fut_f = fut_f[fut_f["category"].astype(str) == str(cat)]

        # Sort by date and take requested weeks
        fut_f = fut_f.sort_values("date").head(weeks).reset_index(drop=True)

        if fut_f.empty:
            return (_empty_fig(t, f"No forecast data for {store} / {cat}"),
                    html.Div(f"Selection not found in predictions. Try re-uploading.",
                             style={"color":"var(--c-text2)","padding":"12px"}))

        fig   = build_forecast_fig(df, fut_f, store, cat, t)
        fut_f["date"] = pd.to_datetime(fut_f["date"]).dt.strftime("%Y-%m-%d")

        rows = []
        for i, (_, r) in enumerate(fut_f.iterrows()):
            prev = fut_f.iloc[i-1]["predicted"] if i > 0 else r["predicted"]
            trend = "📈" if r["predicted"] >= prev else "📉"
            rows.append(html.Tr([
                html.Td(f"W{i+1}"),
                html.Td(r["date"]),
                html.Td(f"{r['predicted']:,.0f}"),
                html.Td(f"{r['lower']:,.0f}"),
                html.Td(f"{r['upper']:,.0f}"),
                html.Td(trend),
            ]))

        table = html.Div([
            html.Table([
                html.Thead(html.Tr([html.Th(c) for c in
                    ["Week","Date","Predicted","Lower Bound","Upper Bound","Trend"]])),
                html.Tbody(rows),
            ], className="data-table",
               style={"background":"var(--c-card)","borderRadius":"12px","overflow":"hidden"}),
        ], style={"marginTop":"12px"})

        return fig, table

    except Exception as e:
        tb = traceback.format_exc()
        err_fig = _empty_fig(t, f"Error: {str(e)[:80]}")
        err_msg = html.Pre(tb[-500:], style={"color":"#e94560","fontSize":"11px",
                                              "padding":"12px","overflowX":"auto"})
        return err_fig, err_msg


# 10. ALERTS
@app.callback(
    Output("al-table",  "children"),
    Output("al-chart",  "children"),
    Output("al-kpis",   "children"),
    Input("st-alerts",  "data"),
    Input("al-status",  "value"),
    Input("al-risk",    "value"),
    Input("st-theme",   "data"),
)
def update_alerts(alerts_json, status_f, min_risk, theme):
    t  = theme or "dark"
    al = _read_df(alerts_json)
    if al is None or al.empty:
        return _upload_prompt("Upload a CSV to see inventory alerts"), html.Div(), []

    filt = al.copy()
    if status_f and status_f != "All":
        filt = filt[filt["status"] == status_f]
    if min_risk:
        filt = filt[filt["risk_score"] >= min_risk]

    ov     = int((al["status"]=="Overstock").sum())
    st     = int((al["status"]=="Stockout Risk").sum())
    hl     = int((al["status"]=="Healthy").sum())
    impact = estimate_financial_impact(al)
    kpis = [
        html.Span(f"🔴 {ov} Overstock",    className="metric-chip metric-bad"),
        html.Span(f"🟡 {st} Stockout Risk", className="metric-chip metric-warn"),
        html.Span(f"🟢 {hl} Healthy",       className="metric-chip metric-good"),
        html.Span(f"💰 {impact['total_risk_mad']:,.0f} at risk", className="metric-chip"),
    ]

    def pill(s):
        cls = {"Overstock":"pill-overstock","Stockout Risk":"pill-stockout","Healthy":"pill-healthy"}.get(s,"")
        return html.Span(s, className=f"status-pill {cls}")

    def risk_bar(v):
        cls = "risk-high" if v>65 else ("risk-medium" if v>35 else "risk-low")
        return html.Div([
            html.Span(f"{v:.0f}", style={"fontWeight":"600","marginRight":"8px",
                                          "minWidth":"30px","display":"inline-block"}),
            html.Div([html.Div(className=f"risk-fill {cls}",
                               style={"width":f"{min(v,100):.0f}%"})],
                     className="risk-bar"),
        ], style={"display":"flex","alignItems":"center"})

    cols = [c for c in ["Date","Store","Dept","status","risk_score","action_needed"] if c in filt.columns]
    if not cols:
        cols = [c for c in ["date","store_id","category","status","risk_score","action_needed"] if c in filt.columns]

    trows = []
    for _, row in filt.head(100).iterrows():
        cells = []
        for c in cols:
            v = row.get(c, "—")
            if c == "status":
                cells.append(html.Td(pill(str(v))))
            elif c == "risk_score":
                cells.append(html.Td(risk_bar(float(v))))
            else:
                if c == "Date" and hasattr(v, "strftime"):
                    txt = v.strftime("%Y-%m-%d")
                elif isinstance(v, str):
                    txt = v[:55]
                elif isinstance(v, (int, float)):
                    txt = f"{v:,.0f}"
                else:
                    txt = str(v)[:16]
                cells.append(html.Td(txt))
        trows.append(html.Tr(cells))

    table = html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th(c.replace("_"," ").title()) for c in cols])),
            html.Tbody(trows),
        ], className="data-table"),
    ], style={"background":"var(--c-card)","borderRadius":"12px",
              "border":"1px solid var(--c-border)","overflow":"hidden"})

    fig = px.histogram(filt, x="risk_score", color="status", nbins=25,
        color_discrete_map={"Healthy":"#10b981","Overstock":"#e94560","Stockout Risk":"#f59e0b"},
        title="Risk Score Distribution")
    fig.update_layout(**_base_layout(t, h=260))
    chart_sec = html.Div([
        html.Div([
            html.Div([html.Div("Risk Distribution", className="chart-title")], className="chart-header"),
            html.Div(dcc.Graph(figure=fig, config={"displayModeBar":False}), className="chart-body"),
        ], className="chart-card"),
    ])
    return table, chart_sec, kpis


# 11. FLOATING CHATBOT
@app.callback(
    Output("float-msgs", "children"),
    Output("st-chat",    "data"),
    Output("float-in",   "value"),
    Input("float-go",    "n_clicks"),
    Input({"type":"fsq","index":ALL}, "n_clicks"),
    State("float-in",    "value"),
    State("st-chat",     "data"),
    State("st-alerts",   "data"),
    prevent_initial_call=True,
)
def on_chat(n_send, sq_clicks, user_input, history, alerts_json):
    tid = ctx.triggered_id
    question = None

    if tid == "float-go" and user_input and user_input.strip():
        question = user_input.strip()
    elif isinstance(tid, dict) and tid.get("type") == "fsq":
        starters = get_starter_questions()
        idx = tid["index"]
        raw = starters[idx]
        for tail in [" 🔴"," 📦"," 💰"," 🏪"," 📊"," 📈"]:
            raw = raw.split(tail)[0]
        question = raw

    if not question:
        raise PreventUpdate

    al = _read_df(alerts_json)
    if al is None or al.empty:
        response = "Please upload a CSV file first so I can analyze your inventory."
    else:
        response = answer_inventory_question(question, al)

    history = history or []
    history.append({"role":"user",      "content":question})
    history.append({"role":"assistant", "content":response})

    def render(m):
        is_u = m["role"] == "user"
        return html.Div([
            html.Div("👤" if is_u else "📦",
                     className=f"chat-avatar {'user-av' if is_u else 'bot-av'}"),
            html.Div(m["content"],
                     className=f"chat-bubble {'bubble-user' if is_u else 'bubble-bot'}"),
        ], className=f"chat-msg {'user' if is_u else ''}")

    init_msg = html.Div([
        html.Div("📦", className="chat-avatar bot-av"),
        html.Div([html.Strong("KhazBot"), html.Br(),
                  html.Span("Bonjour! Ask about your inventory in French or English.")],
                 className="chat-bubble bubble-bot"),
    ], className="chat-msg")

    return [init_msg] + [render(m) for m in history], history, ""


if __name__ == "__main__":
    app.run(debug=False, port=8502, host="0.0.0.0")
