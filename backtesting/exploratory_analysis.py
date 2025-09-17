# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:01:59 2025

@author: waaqa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
ENERGY_COL = "Total"
TZ = "Europe/Amsterdam"  # used only for DST annotations; your data stays as-is

# ---------------------------
# 0) PREP & BASIC CHECKS
# ---------------------------
def basic_checks(energy_df: pd.DataFrame, weather_df: pd.DataFrame, energy_col: str = ENERGY_COL):
    print("=== BASIC SHAPE & ALIGNMENT ===")
    for name, df in [("energy", energy_df), ("weather", weather_df)]:
        print(f"\n{name}_data:")
        print(f"  Start: {df.index.min()} | End: {df.index.max()} | Len: {len(df):,}")
        print(f"  Is monotonic increasing index? {df.index.is_monotonic_increasing}")
        print(f"  Duplicated timestamps: {df.index.duplicated().sum()}")
        print(f"  NaNs per column:\n{df.isna().sum()}")

    # frequency check (mode of diffs)
    def freq_report(df, label):
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            print(f"\nNo diffs for {label} (single row?).")
            return
        mode = diffs.mode().iloc[0]
        bad_steps = (diffs != mode).sum()
        print(f"\n{label} frequency:")
        print(f"  Mode step: {mode} | Mismatched steps: {bad_steps} / {len(diffs)}")
        # expected 15 minutes
        expected = pd.Timedelta(minutes=15)
        if abs(mode - expected) > pd.Timedelta(seconds=0):
            print(f"  WARNING: Mode step != 15 minutes (is {mode}).")
    freq_report(energy_df, "energy")
    freq_report(weather_df, "weather")

    # alignment window overlap
    overlap_start = max(energy_df.index.min(), weather_df.index.min())
    overlap_end = min(energy_df.index.max(), weather_df.index.max())
    print(f"\nOverlap window: {overlap_start} → {overlap_end}")

    # quick range diagnostic for energy
    en = energy_df[energy_col]
    print("\nEnergy range & zeros:")
    print(f"  min={en.min()}, max={en.max()}, mean={en.mean():.2f}, std={en.std():.2f}")
    print(f"  zeros={int((en==0).sum())}")

# ---------------------------
# 1) MONTHLY SUMMARY TABLES
# ---------------------------
def monthly_summaries(energy_df: pd.DataFrame, weather_df: pd.DataFrame, energy_col: str = ENERGY_COL):
    df = energy_df.join(weather_df, how="inner")
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Energy monthly totals & stats
    agg_map = {
        energy_col: ["sum", "mean", "median", "std", "min", "max", "count"]
    }
    for c in weather_df.columns:
        agg_map[c] = ["mean", "median", "std", "min", "max"]

    monthly = df.groupby(["year","month"]).agg(agg_map)
    # Make column names flat
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly = monthly.sort_index()

    print("\n=== MONTHLY SUMMARY (Energy + Weather) ===")
    return df, monthly

# ---------------------------
# 2) SAME-MONTH (JAN-JAN, ...) OVERLAYS
#    Intraday average profile per month, over years
# ---------------------------
def plot_same_month_overlays(df: pd.DataFrame, value_col: str, title_prefix: str):
    # df: DatetimeIndex, has at least value_col
    if df.empty:
        print(f"[{title_prefix}] No data to plot.")
        return

    # add helpers
    tmp = df.copy()
    tmp["year"] = tmp.index.year
    tmp["month"] = tmp.index.month
    tmp["tod"] = tmp.index.time  # time-of-day

    for m in range(1, 13):
        sub = tmp[tmp["month"] == m]
        if sub.empty:
            continue

        # average 15-min value per time-of-day per year within this month
        prof = sub.groupby(["year","tod"])[value_col].mean().reset_index()

        # sort x-axis by time-of-day
        times = sorted(prof["tod"].unique(), key=lambda t:(t.hour,t.minute,t.second))
        x = pd.to_datetime([f"2000-01-01 {t.strftime('%H:%M:%S')}" for t in times])

        plt.figure(figsize=(8,4))
        for y in sorted(prof["year"].unique()):
            yser = prof[prof["year"]==y].set_index("tod").reindex(times)[value_col]
            plt.plot(x, yser.values, label=str(y))
        month_name = pd.Timestamp(2000, m, 1).strftime("%B")
        plt.title(f"{title_prefix} – Avg 15-min profile in {month_name} by year")
        plt.xlabel("Time of day")
        plt.ylabel(value_col)
        plt.legend(title="Year")
        plt.tight_layout()
        plt.show()

# ---------------------------
# 3) MARCH DEEP DIVE
# ---------------------------
def last_sunday(year, month):
    # Find last Sunday of a given month
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while d.weekday() != 6:  # 6=Sunday
        d -= pd.Timedelta(days=1)
    return d.normalize()

def march_deep_dive(energy_df: pd.DataFrame, weather_df: pd.DataFrame, energy_col: str = ENERGY_COL):
    df = energy_df.join(weather_df, how="inner")
    mar = df[df.index.month==3].copy()
    if mar.empty:
        print("No March in the joined data window.")
        return

    # ---- A) Daily totals for March (helps spot DST day, holidays, outages)
    daily = mar[energy_col].resample("D").sum()
    plt.figure(figsize=(9,3.5))
    daily.plot()
    plt.title("March daily energy totals")
    plt.xlabel("Date")
    plt.ylabel(f"{energy_col} (daily sum)")
    plt.tight_layout()
    plt.show()

    # annotate DST start (last Sunday of March) for each year present
    years = sorted(mar.index.year.unique())
    dst_starts = [last_sunday(y, 3) for y in years]
    print("DST start (last Sunday of March) per year:", [d.date() for d in dst_starts])

    # ---- B) Intraday profiles in March by Day-of-Week
    mar["dow"] = mar.index.dayofweek  # 0=Mon..6=Sun
    mar["tod"] = mar.index.time
    prof = mar.groupby(["dow","tod"])[energy_col].mean().reset_index()
    times = sorted(prof["tod"].unique(), key=lambda t:(t.hour,t.minute,t.second))
    x = pd.to_datetime([f"2000-01-01 {t.strftime('%H:%M:%S')}" for t in times])

    plt.figure(figsize=(9,4))
    for d in range(7):
        dser = prof[prof["dow"]==d].set_index("tod").reindex(times)[energy_col]
        plt.plot(x, dser.values, label=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d])
    plt.title("March – Avg 15-min intraday profile by day-of-week")
    plt.xlabel("Time of day")
    plt.ylabel(energy_col)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- C) Energy vs Irradiance scatter (March)
    if "direct_normal_irradiance_instant" in weather_df.columns:
        plt.figure(figsize=(5.5,4.5))
        plt.scatter(mar["direct_normal_irradiance_instant"].values, mar[energy_col].values, s=4, alpha=0.5)
        plt.title("March scatter: Energy vs DNI")
        plt.xlabel("direct_normal_irradiance_instant")
        plt.ylabel(energy_col)
        plt.tight_layout()
        plt.show()

    # ---- D) Simple weather baseline: fit on non-March to predict Energy; look at March residuals
    non_mar = df[df.index.month != 3].dropna()
    mar_only = df[df.index.month == 3].dropna()
    features = [c for c in weather_df.columns if c in df.columns]
    if non_mar.empty or mar_only.empty or len(features)==0:
        print("Skipping baseline residuals (insufficient data or no weather features).")
        return

    X_train = non_mar[features].values
    y_train = non_mar[energy_col].astype(float).values
    # add intercept
    X_design = np.column_stack([np.ones(len(X_train)), X_train])
    coef, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)

    # predict March
    Xm = mar_only[features].values
    Xm_design = np.column_stack([np.ones(len(Xm)), Xm])
    y_pred = Xm_design @ coef
    mar_only = mar_only.copy()
    mar_only["resid"] = mar_only[energy_col].astype(float).values - y_pred

    # daily residuals
    daily_resid = mar_only["resid"].resample("D").sum()
    plt.figure(figsize=(9,3.5))
    daily_resid.plot()
    plt.title("March daily residuals (Energy − simple weather baseline)")
    plt.xlabel("Date")
    plt.ylabel("Residual (sum/day)")
    plt.axhline(0, linestyle="--")
    plt.tight_layout()
    plt.show()

    print("\nWorst March days by residual magnitude (top 10):")
    print(daily_resid.reindex(daily_resid.abs().sort_values(ascending=False).index).head(10))

    # ---- E) Spike/Gap detection in March (rolling-MAD)
    s = mar[energy_col].astype(float)
    med = s.rolling(96*7, min_periods=96).median()  # 7-day window (96=15min/day)
    mad = (s - med).abs().rolling(96*7, min_periods=96).median()
    robust_z = (s - med) / (mad.replace(0, np.nan))
    suspects = robust_z[robust_z.abs() >= 6].dropna()
    print(f"\nPotential anomalies in March (|robust z| ≥ 6): {len(suspects)}")
    print(suspects.sort_values(key=np.abs, ascending=False).head(15))
    
    
    
    
    

def last_sunday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while d.weekday() != 6:  # 6=Sun
        d -= pd.Timedelta(days=1)
    return d

def plot_march_profiles_by_year(energy_df, energy_col=ENERGY_COL, tz=TZ):
    # ensure sorted & unique
    energy_df = energy_df.sort_index()
    energy_df = energy_df[~energy_df.index.duplicated(keep="first")]

    years = sorted(energy_df.index.year.unique())
    march_years = [y for y in years if not energy_df.loc[str(y) + "-03-01": str(y) + "-03-31"].empty]
    if not march_years:
        print("No March data found.")
        return

    plt.figure(figsize=(12,4.5))

    for y in march_years:
        s = energy_df.loc[str(y) + "-03-01": str(y) + "-03-31"][energy_col].copy()

        # put on local wall-clock to visualize the DST gap where it occurs
        idx = s.index
        if idx.tz is None:
            idx_local = idx.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            idx_local = idx.tz_convert(tz)

        # x-axis: day-of-month + fraction of day (so different years overlay)
        x = idx_local.day + (idx_local.hour * 60 + idx_local.minute) / 1440.0

        plt.plot(x, s.values, label=str(y), linewidth=0.9)

        # mark DST start (last Sunday of March in local time)
        dst_date = last_sunday(y, 3)
        plt.axvline(dst_date.day, linestyle="--", alpha=0.5)

    plt.title("Energy — Full March profile (overlay by year, local time)")
    plt.xlabel("Day of March (local, Europe/Amsterdam)")
    plt.ylabel(energy_col)
    plt.xlim(1, 32)
    plt.xticks([1, 8, 15, 22, 29, 31])
    plt.grid(True, alpha=0.3)
    plt.legend(title="Year")
    plt.tight_layout()
    plt.show()

# ---- run it
plot_march_profiles_by_year(energy_data, ENERGY_COL)
    
plt.plot(energy_data)
plt.xlabel("Datum")
plt.ylabel("kWh")
    

imbalance_data = pd.read_excel(
    r"C:\Users\waaqa\Desktop\complate_imbalance_estimation_update.xlsx",
    index_col=0,        # use the first (unnamed) column as index
    parse_dates=[0],    # parse that column as datetimes
)
imbalance_data.index.name = "datetime"
plt.plot(imbalance_data["Prediction [MWh]"])
plt.plot(imbalance_data["Realization [MWh]"])
plt.xlabel("Datum")
plt.ylabel("MWh")

# ---------------------------
# RUN
# ---------------------------
# Ensure sorted, unique index
energy_data = energy_data.sort_index()
weather_data = weather_data.sort_index()
energy_data = energy_data[~energy_data.index.duplicated(keep="first")]
weather_data = weather_data[~weather_data.index.duplicated(keep="first")]

basic_checks(energy_data, weather_data)
joined, monthly = monthly_summaries(energy_data, weather_data, ENERGY_COL)

# Same-month overlays: Energy
plot_same_month_overlays(energy_data, ENERGY_COL, "Energy")

# Same-month overlays: Weather
for wcol in weather_data.columns:
    plot_same_month_overlays(weather_data[[wcol]], wcol, f"Weather: {wcol}")

# March focus
march_deep_dive(energy_data, weather_data, ENERGY_COL)












