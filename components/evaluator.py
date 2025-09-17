# components/reports.py
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# -------- Engine (cached) --------
@st.cache_resource
def get_engine():
    cfg = st.secrets["azure_sql"]  # put creds in .streamlit/secrets.toml
    odbc = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={cfg['server']};DATABASE={cfg['database']};"
        f"UID={cfg['username']};PWD={cfg['password']};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=15;"
    )
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}",
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=0,
        pool_recycle=1800,
        fast_executemany=True,
    )

# -------- Helpers --------
def bounds(date_str: str):
    start = datetime.fromisoformat(date_str)
    end = start + timedelta(days=1)
    return start, end

# -------- Data access (cached) --------
@st.cache_data(ttl=600, show_spinner=False)
def get_run_list(date_str: str):
    eng = get_engine()
    start, end = bounds(date_str)
    q = text("""
        SELECT DISTINCT [InsertionTimestamp]
        FROM [DISP].[ForecastGold]
        WHERE [Datetime] >= :start AND [Datetime] < :end
          AND [Profile] = 'Intraday Portfolio'
        ORDER BY [InsertionTimestamp]
    """)
    df = pd.read_sql(q, eng, params={"start": start, "end": end})
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["InsertionTimestamp"]):
        df["InsertionTimestamp"] = pd.to_datetime(df["InsertionTimestamp"])
    return df["InsertionTimestamp"].tolist()

@st.cache_data(ttl=600, show_spinner=False)
def get_portfolio(date_str: str):
    eng = get_engine()
    start, end = bounds(date_str)
    q = text("""
        SELECT [Datetime], [kWh]
        FROM [DISP].[ForecastGold]
        WHERE [Datetime] >= :start AND [Datetime] < :end
          AND [Profile] = 'Portfolio'
        ORDER BY [Datetime]
    """)
    df = pd.read_sql(q, eng, params={"start": start, "end": end})
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_intraday_used(date_str: str):
    """Latest forecast per timestamp (rn=1) computed in SQL for speed."""
    eng = get_engine()
    start, end = bounds(date_str)
    q = text("""
        WITH ranked AS (
          SELECT [Datetime],[kWh],
                 ROW_NUMBER() OVER (PARTITION BY [Datetime]
                                    ORDER BY [InsertionTimestamp] DESC) AS rn
          FROM [DISP].[ForecastGold]
          WHERE [Datetime] >= :start AND [Datetime] < :end
            AND [Profile] = 'Intraday Portfolio'
        )
        SELECT [Datetime],[kWh]
        FROM ranked WHERE rn = 1
        ORDER BY [Datetime]
    """)
    df = pd.read_sql(q, eng, params={"start": start, "end": end})
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_intraday_run(date_str: str, run_ts):
    """Single selected intraday run."""
    if run_ts is None:
        return pd.DataFrame(columns=["Datetime", "kWh"])
    eng = get_engine()
    start, end = bounds(date_str)
    q = text("""
        SELECT [Datetime],[kWh]
        FROM [DISP].[ForecastGold]
        WHERE [Datetime] >= :start AND [Datetime] < :end
          AND [Profile] = 'Intraday Portfolio'
          AND [InsertionTimestamp] = :run
        ORDER BY [Datetime]
    """)
    df = pd.read_sql(q, eng, params={"start": start, "end": end, "run": run_ts})
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_intraday_coverage(date_str: str):
    """Start/End/Points per intraday run for the expander table."""
    eng = get_engine()
    start, end = bounds(date_str)
    q = text("""
        SELECT [InsertionTimestamp] AS Run,
               MIN([Datetime]) AS Start,
               MAX([Datetime]) AS [End],
               COUNT(*) AS Points
        FROM [DISP].[ForecastGold]
        WHERE [Datetime] >= :start AND [Datetime] < :end
          AND [Profile] = 'Intraday Portfolio'
        GROUP BY [InsertionTimestamp]
        ORDER BY [InsertionTimestamp]
    """)
    df = pd.read_sql(q, eng, params={"start": start, "end": end})
    if not df.empty:
        df["Run"] = pd.to_datetime(df["Run"])
        df["Start"] = pd.to_datetime(df["Start"])
        df["End"] = pd.to_datetime(df["End"])
    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_net(date_str: str):
    """Actuals; resampled to 15-minute for plotting."""
    eng = get_engine()
    start, end = bounds(date_str)
    q = text("""
        SELECT [Date],[Time], SUM([Net]) AS [Net]
        FROM [DISP].[P4ReadingView]
        WHERE [Date] >= :start AND [Date] < :end
        GROUP BY [Date],[Time]
    """)
    df = pd.read_sql(q, eng, params={"start": start, "end": end})
    if df.empty:
        return df
    df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    df = df.sort_values("Datetime", kind="mergesort").set_index("Datetime").resample("15T").sum().reset_index()
    return df[["Datetime", "Net"]]


@st.cache_data(ttl=600, show_spinner=False)
def get_entsoe_instances():
    """List available InstanceCode values to pick the bidding zone/currency."""
    eng = get_engine()
    q = text("""
        SELECT DISTINCT [InstanceCode]
        FROM [ENTSOE].[DayAheadSpotPrice]
        ORDER BY [InstanceCode]
    """)
    df = pd.read_sql(q, eng)
    return df["InstanceCode"].tolist()



# -------- Page entrypoint --------
def render():
    st.header("Forecasting models evaluator (Day Ahead, Intraday, Actuals)")

    # --- UI ---
    date_str = st.date_input("Select date", value=date.today()).strftime("%Y-%m-%d")
    show_net = st.toggle("Show actual net consumption and generation Profile", value=False)

    # Step 1: tiny call to get available runs (fast & cached)
    t0 = time.perf_counter()
    runs = get_run_list(date_str)
    t1 = time.perf_counter()

    sel_run = st.select_slider(
        "Select Intraday run (by forecast start time)",
        options=runs,
        value=runs[-1] if runs else None,
        format_func=lambda ts: ts.strftime("%H:%M") if ts else "—",
        disabled=not runs,
        help="Ticks show the run insertion time’s first forecast timestamp (HH:MM).",
    )

    # Step 2: fetch only what we need now (each cached by date/run)
    df_port = get_portfolio(date_str); t2 = time.perf_counter()
    df_used = get_intraday_used(date_str); t3 = time.perf_counter()
    df_run  = get_intraday_run(date_str, sel_run); t4 = time.perf_counter()
    df_cov  = get_intraday_coverage(date_str); t5 = time.perf_counter()
    df_net  = get_net(date_str) if show_net else pd.DataFrame(columns=["Datetime","Net"]); t6 = time.perf_counter()

    # Early exits
    if df_port.empty and df_used.empty and df_run.empty:
        st.warning("No forecast data found for the selected date.")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=110)
    if not df_port.empty:
        ax.plot(df_port["Datetime"], df_port["kWh"], label="Day ahead forecast", linewidth=2)
    if not df_used.empty:
        ax.plot(df_used["Datetime"], df_used["kWh"], label="Intraday (used: latest by time)",
                linewidth=2.0, linestyle="--", alpha=0.9)
    if not df_run.empty:
        # Grab first timestamp of the selected run for label
        first_ts = df_run["Datetime"].iloc[0] if not df_run.empty else None
        hhmm = first_ts.strftime("%H:%M") if first_ts is not None else "?"
        ax.plot(df_run["Datetime"], df_run["kWh"], label=f"Intraday (selected @{hhmm})",
                linewidth=1.5, alpha=0.9)
    if show_net and not df_net.empty:
        ax.plot(df_net["Datetime"], df_net["Net"], label="Net (actuals)", linewidth=1.1, alpha=0.9)

    ax.set_title(f"Forecasts{' vs Net' if show_net else ''} for {date_str}")
    ax.set_xlabel("Datetime"); ax.set_ylabel("kWh")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Coverage table
    with st.expander("Show intraday run coverage"):
        if df_cov.empty:
            st.write("No intraday runs for this date.")
        else:
            st.dataframe(df_cov, use_container_width=True)

    # Timing (helps you see cache hits: repeat runs should show ~0–5 ms)
    st.caption(
        f"Runs: {(t1-t0)*1000:.0f} ms • Portfolio: {(t2-t1)*1000:.0f} ms • "
        f"Used: {(t3-t2)*1000:.0f} ms • Selected: {(t4-t3)*1000:.0f} ms • "
        f"Coverage: {(t5-t4)*1000:.0f} ms • Net: {(t6-t5)*1000:.0f} ms"
    )
