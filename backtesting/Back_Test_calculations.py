# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:02:33 2025

@author: user
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from .imbalance_cost_entsoe import imbalance_entsoe
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

def calculate_imbalance_cost(y_true_all, y_pred_all):
    """
    Calculate imbalance costs given predictions and actual values.
    Returns a DataFrame with aligned series, prices, and cost columns.
    Raises ValueError with clear messages if inputs/prices are missing or misaligned.
    """


    # ---------- 0) Validate inputs ----------
    if y_true_all is None or y_pred_all is None:
        raise ValueError("y_true_all or y_pred_all is None")

    # Normalize to 2-column DF: Prediction / Realization
    if isinstance(y_pred_all, pd.Series):
        y_pred = y_pred_all.rename("Prediction [MWh]").to_frame()
    else:
        y_pred = y_pred_all.copy()
        if "y_pred" in y_pred.columns:
            y_pred = y_pred[["y_pred"]].rename(columns={"y_pred": "Prediction [MWh]"})
        else:
            y_pred.columns = ["Prediction [MWh]"]

    if isinstance(y_true_all, pd.Series):
        y_true = y_true_all.rename("Realization [MWh]").to_frame()
    else:
        y_true = y_true_all.copy()
        if "y_true" in y_true.columns:
            y_true = y_true[["y_true"]].rename(columns={"y_true": "Realization [MWh]"})
        else:
            y_true.columns = ["Realization [MWh]"]

    if y_pred.empty or y_true.empty:
        raise ValueError("y_true or y_pred is empty")

    # ---------- 1) Ensure datetime + timezone (assume EU/Amsterdam local, convert to UTC) ----------
    def _to_utc(idx: pd.Index) -> pd.DatetimeIndex:
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx, errors="coerce")
        if idx.tz is None:
            idx = idx.tz_localize("Europe/Amsterdam", ambiguous="infer", nonexistent="shift_forward")
        return idx.tz_convert("UTC")

    y_pred.index = _to_utc(y_pred.index)
    y_true.index = _to_utc(y_true.index)

    # Align on intersection
    df = y_pred.join(y_true, how="inner").sort_index()
    if df.empty:
        raise ValueError("No overlapping timestamps between prediction and realization after alignment")

    # Convert to MWh (keeping your original /1000 behavior)
    df = df / 1000.0
    df["Imbalance [MWh]"] = df["Realization [MWh]"] - df["Prediction [MWh]"]

    # ---------- 2) Fetch imbalance prices with tz-aware bounds ----------
    start_date = df.index.min()   # tz-aware UTC
    end_date   = df.index.max()   # tz-aware UTC

    prices = imbalance_entsoe(start_date, end_date)  # must return a DataFrame
    if prices is None or getattr(prices, "empty", True):
        raise ValueError("Imbalance price fetch returned None/empty. Check source/credentials/time window.")

    # Ensure price index is UTC & sorted
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("Europe/Amsterdam", ambiguous="infer", nonexistent="shift_forward")
    prices.index = prices.index.tz_convert("UTC")
    prices = prices.sort_index()

    # Normalize expected columns
    cols_lower = {c.lower(): c for c in prices.columns}
    long_name  = cols_lower.get("long")  or cols_lower.get("invoeden [eur/mwh]")
    short_name = cols_lower.get("short") or cols_lower.get("afnemen [eur/mwh]")
    if long_name is None or short_name is None:
        raise ValueError(f"Price columns not found. Got {list(prices.columns)}; expected 'Long' and 'Short'.")

    prices = prices.rename(columns={
        long_name:  "Invoeden [EUR/MWh]",
        short_name: "Afnemen [EUR/MWh]"
    })[["Invoeden [EUR/MWh]", "Afnemen [EUR/MWh]"]]

    # ---------- 3) Align prices to df timestamps (DON'T slice by iloc/len) ----------
    # Try direct reindex with forward-fill; fallback to nearest with tolerance
    prices_aligned = prices.reindex(df.index, method="ffill")
    if prices_aligned.isna().any().any():
        prices_aligned = prices.reindex(df.index, method="nearest", tolerance=pd.Timedelta("8min"))
    if prices_aligned.isna().any().any():
        raise ValueError("Could not align prices to the series timestamps (price data too sparse/gappy).")

    final = pd.concat([df, prices_aligned], axis=1)

    # ---------- 4) Cost calculations ----------
    # If Imbalance < 0 => Afnemen (buy), else Invoeden (sell)
    final["Imbalance cost [EUR]"] = np.where(
        final["Imbalance [MWh]"] < 0,
        final["Afnemen [EUR/MWh]"] * final["Imbalance [MWh]"],
        final["Invoeden [EUR/MWh]"] * final["Imbalance [MWh]"],
    )

    # Nomination: hourly mean of Prediction, right-closed delivery hours
    hourly_means = final["Prediction [MWh]"].resample("h", label="right", closed="right").transform("mean")
    final["Nomination [MWh]"] = hourly_means
    final["Imbalance Nomination Based [MWh]"] = final["Realization [MWh]"] - final["Nomination [MWh]"]

    final["Cost Imbalance Nomination Based [Euro]"] = np.where(
        final["Imbalance Nomination Based [MWh]"] < 0,
        final["Afnemen [EUR/MWh]"] * final["Imbalance Nomination Based [MWh]"],
        final["Invoeden [EUR/MWh]"] * final["Imbalance Nomination Based [MWh]"],
    )

    return final



def calculate_and_plot_imbalance(y_true_all, y_pred_all, target_col):
    """Calculate imbalance costs and plot results."""
    costs = calculate_imbalance_cost(y_true_all, y_pred_all)
    
    # ####Plot results
    plt.figure(figsize=(15, 5))
    plt.plot(y_true_all.values, label="Actual", alpha=0.8)
    plt.plot(y_pred_all.values, label="Predicted", alpha=0.8)
    plt.title(f"Forecast vs Actuals - {target_col}")
    plt.xlabel("Time Steps")
    plt.ylabel("Power Generation")
    plt.legend()
    plt.show()
    
    return costs



def calculate_metrics(total_cost):
    metrics = {}

    # 1. Sum Absolute Imbalance Volume (Overall)
    metrics['Sum Absolute Imbalance Volume'] = total_cost["Imbalance [MWh]"].abs().sum()

    # 2. Ratio Absolute Imbalance to Absolute Volume 
    metrics['Sum Absolute Volume'] = total_cost["Realization [MWh]"].abs().sum()
    metrics['Ratio Absolute Imbalance to Absolute Volume'] = (
        metrics['Sum Absolute Imbalance Volume'] / metrics['Sum Absolute Volume']
    )

    # 3. Sum Imbalance Volume (Overall)
    metrics['Sum Imbalance Volume'] = total_cost["Imbalance [MWh]"].sum()

    # 4. Ratio Imbalance Volume to Volume (Overall)
    metrics['Sum Volume'] = total_cost["Realization [MWh]"].sum()
    metrics['Ratio Imbalance Volume to Volume'] = (
        metrics['Sum Imbalance Volume'] / metrics['Sum Volume']
    )

    # 5. Sum Imbalance Cost (Overall)
    metrics['Sum Imbalance Cost'] = total_cost["Cost Imbalance Nomination Based [Euro]"].sum()

    # 6. Ratio Imbalance Cost to Absolute Volume (Overall)
    metrics['Ratio Imbalance Cost to Absolute Volume'] = (
        metrics['Sum Imbalance Cost'] / metrics['Sum Absolute Volume']
    )

    # 7. Ratio Imbalance Cost to Volume (Overall)
    metrics['Ratio Imbalance Cost to Volume'] = (
        metrics['Sum Imbalance Cost'] / metrics['Sum Volume']
    )

    # Metrics conditioned on net production (Realization > 0)
    net_production = total_cost[total_cost["Realization [MWh]"] > 0]
    sum_abs_imbalance_volume_prod = net_production["Imbalance [MWh]"].abs().sum()
    sum_abs_volume_prod = net_production["Realization [MWh]"].abs().sum()
    sum_imbalance_volume_prod = net_production["Imbalance [MWh]"].sum()
    sum_volume_prod = net_production["Realization [MWh]"].sum()
    sum_imbalance_cost_prod = net_production["Cost Imbalance Nomination Based [Euro]"].sum()

    # Net Production Metrics
    metrics['Sum Absolute Imbalance Volume (Net Production)'] = sum_abs_imbalance_volume_prod
    metrics['Sum Absolute Volume (Net Production)'] = sum_abs_volume_prod
    metrics['Ratio Absolute Imbalance to Absolute Volume (Net Production)'] = (
        sum_abs_imbalance_volume_prod / sum_abs_volume_prod
    )
    metrics['Ratio Imbalance to Volume (Net Production)'] = (
        sum_imbalance_volume_prod / sum_volume_prod
    )
    metrics['Sum Imbalance Cost (Net Production)'] = sum_imbalance_cost_prod
    metrics['Ratio Imbalance Cost to Absolute Volume (Net Production)'] = (
        sum_imbalance_cost_prod / sum_abs_volume_prod
    )
    metrics['Ratio Imbalance Cost to Volume (Net Production)'] = (
        sum_imbalance_cost_prod / sum_volume_prod
    )

    # Metrics conditioned on net consumption (Realization < 0)
    net_consumption = total_cost[total_cost["Realization [MWh]"] < 0]
    sum_abs_imbalance_volume_cons = net_consumption["Imbalance [MWh]"].abs().sum()
    sum_abs_volume_cons = net_consumption["Realization [MWh]"].abs().sum()
    sum_imbalance_volume_cons = net_consumption["Imbalance [MWh]"].sum()
    sum_volume_cons = net_consumption["Realization [MWh]"].sum()
    sum_imbalance_cost_cons = net_consumption["Cost Imbalance Nomination Based [Euro]"].sum()

    # Net Consumption Metrics
    metrics['Sum Absolute Imbalance Volume (Net Consumption)'] = sum_abs_imbalance_volume_cons
    metrics['Sum Absolute Volume (Net Consumption)'] = sum_abs_volume_cons
    metrics['Ratio Absolute Imbalance to Absolute Volume (Net Consumption)'] = (
        sum_abs_imbalance_volume_cons / sum_abs_volume_cons
    )
    metrics['Ratio Imbalance to Volume (Net Consumption)'] = (
        sum_imbalance_volume_cons / sum_volume_cons
    )
    metrics['Sum Imbalance Cost (Net Consumption)'] = sum_imbalance_cost_cons
    metrics['Ratio Imbalance Cost to Absolute Volume (Net Consumption)'] = (
        sum_imbalance_cost_cons / sum_abs_volume_cons
    )
    metrics['Ratio Imbalance Cost to Volume (Net Consumption)'] = (
        sum_imbalance_cost_cons / sum_volume_cons
    )

    # Cumulative imbalance cost
    cumulative_cost = total_cost["Cost Imbalance Nomination Based [Euro]"].cumsum()

    return metrics, cumulative_cost


def plot_cumulative_cost(cumulative_cost):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_cost.index, cumulative_cost.values, label='Cumulative Imbalance Cost', color='blue')
    ax.set_xlabel('DateTime')
    ax.set_ylabel('Cumulative Imbalance Cost [Euro]')
    ax.set_title('Cumulative Imbalance Cost Over Time')

    # Highlight and annotate the maximum peak
    max_cost_idx = cumulative_cost.idxmax()
    max_cost = cumulative_cost.max()
    ax.annotate(f'Peak: {max_cost:.2f}€', xy=(max_cost_idx, max_cost), xytext=(max_cost_idx, max_cost * 0.9),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')

    # Highlight and annotate the minimum peak
    min_cost_idx = cumulative_cost.idxmin()
    min_cost = cumulative_cost.min()
    ax.annotate(f'Low: {min_cost:.2f}€', xy=(min_cost_idx, min_cost), xytext=(min_cost_idx, min_cost * 1.1),
                arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green')

    # Highlight weekends
    for i in range(len(cumulative_cost)):
        if cumulative_cost.index[i].weekday() >= 5:  # Weekend
            ax.axvspan(cumulative_cost.index[i], cumulative_cost.index[i] + pd.Timedelta(hours=1),
                       color='grey', alpha=0.2)

    # Add legend entry for weekends
    weekend_patch = mpatches.Patch(color='grey', alpha=0.2, label='Weekend')
    ax.legend(handles=[ax.get_lines()[0], weekend_patch])

    # Tilt x-axis labels 45 degrees
    plt.xticks(rotation=45)

    ax.grid(True)
    plt.tight_layout()
    return fig


def add_divider(pdf, title=None):
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.axis('off')
    if title:
        ax.text(0.5, 0.5, title, fontsize=16, ha='center', va='center')
    ax.plot([0, 1], [0, 0], color='black', lw=2)  # Add a line divider
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def generate_pdf_report(total_cost, y_true_all, y_pred_all, target_col, output_path):
    metrics, cumulative_cost = calculate_metrics(total_cost)
    with PdfPages(output_path) as pdf:
        # Title Page
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        ax.text(0.5, 0.8, "DynamicEnergyTrading", fontsize=30, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.5, 'Backtest Metrics Client Report', fontsize=24, ha='center', va='center')

        
        pdf.savefig(fig)
        plt.close(fig)

        # General Metrics (Overall Net)
        add_divider(pdf, "General Metrics (Overall Net)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        table_data = [
            ["1. Sum Absolute Imbalance Volume", f"{metrics['Sum Absolute Imbalance Volume']:.2f}"],
            ["2. Ratio Absolute Imbalance to Absolute Volume", f"{metrics['Ratio Absolute Imbalance to Absolute Volume']:.2f}"],
            ["3. Sum Imbalance Volume", f"{metrics['Sum Imbalance Volume']:.2f}"],
            ["4. Ratio Imbalance Volume to Volume", f"{metrics['Ratio Imbalance Volume to Volume']:.2f}"],
            ["5. Sum Imbalance Cost", f"{metrics['Sum Imbalance Cost']:.2f}"],
            ["6. Ratio Imbalance Cost to Absolute Volume", f"{metrics['Ratio Imbalance Cost to Absolute Volume']:.2f}"],
            ["7. Ratio Imbalance Cost to Volume", f"{metrics['Ratio Imbalance Cost to Volume']:.2f}"]
        ]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1])
        ax.set_title("Metrics - Generation and Consumption Combined", fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)

        # Net Production Metrics
        add_divider(pdf, "Metrics Conditioned on Net Production (Realization > 0)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        table_data = [
            ["1. Sum Absolute Imbalance Volume (Net Production)", f"{metrics['Sum Absolute Imbalance Volume (Net Production)']:.2f}"],
            ["2. Sum Absolute Volume (Net Production)", f"{metrics['Sum Absolute Volume (Net Production)']:.2f}"],
            ["3. Ratio Absolute Imbalance to Absolute Volume (Net Production)", f"{metrics['Ratio Absolute Imbalance to Absolute Volume (Net Production)']:.2f}"],
            ["4. Ratio Imbalance to Volume (Net Production)", f"{metrics['Ratio Imbalance to Volume (Net Production)']:.2f}"],
            ["5. Sum Imbalance Cost (Net Production)", f"{metrics['Sum Imbalance Cost (Net Production)']:.2f}"],
            ["6. Ratio Imbalance Cost to Absolute Volume (Net Production)", f"{metrics['Ratio Imbalance Cost to Absolute Volume (Net Production)']:.2f}"],
            ["7. Ratio Imbalance Cost to Volume (Net Production)", f"{metrics['Ratio Imbalance Cost to Volume (Net Production)']:.2f}"]
        ]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1])
        ax.set_title("Net Production Metrics", fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)

        # Net Consumption Metrics
        add_divider(pdf, "Metrics Conditioned on Net Consumption (Realization < 0)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        table_data = [
            ["1. Sum Absolute Imbalance Volume (Net Consumption)", f"{metrics['Sum Absolute Imbalance Volume (Net Consumption)']:.2f}"],
            ["2. Sum Absolute Volume (Net Consumption)", f"{metrics['Sum Absolute Volume (Net Consumption)']:.2f}"],
            ["3. Ratio Absolute Imbalance to Absolute Volume (Net Consumption)", f"{metrics['Ratio Absolute Imbalance to Absolute Volume (Net Consumption)']:.2f}"],
            ["4. Ratio Imbalance to Volume (Net Consumption)", f"{metrics['Ratio Imbalance to Volume (Net Consumption)']:.2f}"],
            ["5. Sum Imbalance Cost (Net Consumption)", f"{metrics['Sum Imbalance Cost (Net Consumption)']:.2f}"],
            ["6. Ratio Imbalance Cost to Absolute Volume (Net Consumption)", f"{metrics['Ratio Imbalance Cost to Absolute Volume (Net Consumption)']:.2f}"],
            ["7. Ratio Imbalance Cost to Volume (Net Consumption)", f"{metrics['Ratio Imbalance Cost to Volume (Net Consumption)']:.2f}"]
        ]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1])
        ax.set_title("Net Consumption Metrics", fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)

        # Cumulative Cost Plot
        cumulative_cost_plot = plot_cumulative_cost(cumulative_cost)
        pdf.savefig(cumulative_cost_plot)
        plt.close(cumulative_cost_plot)

    print(f"PDF report successfully generated at: {output_path}")

