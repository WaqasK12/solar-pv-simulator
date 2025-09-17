# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:14:31 2025

@author: waaqa
"""
# components/imbalance.py
import io
import os
import base64
import numpy as np
import pandas as pd
import streamlit as st

# === your modules (as used in your script) ===
from backtesting.weather_data_fetch import get_weather_data
from backtesting.models import rolling_backtest
from backtesting.Back_Test_calculations import generate_pdf_report, calculate_and_plot_imbalance

import holidays

# ---------- Helpers from your script (inlined for convenience) ----------

def load_and_clean_data(file_like_or_path, date_format, target_col):
    """Load and preprocess the uploaded file. Expects a 'Date' column."""
    if isinstance(file_like_or_path, str):
        path = file_like_or_path
        if path.endswith('.xlsx'):
            energy_data = pd.read_excel(path)
        elif path.endswith('.csv'):
            energy_data = pd.read_csv(path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    else:
        # file-like (from st.file_uploader)
        name = file_like_or_path.name.lower()
        if name.endswith('.xlsx'):
            energy_data = pd.read_excel(file_like_or_path)
        elif name.endswith('.csv'):
            energy_data = pd.read_csv(file_like_or_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    energy_data = energy_data.dropna()
    if 'Date' not in energy_data.columns:
        raise ValueError("The input file must contain a 'Date' column.")

    energy_data['Date'] = pd.to_datetime(energy_data['Date'], format=date_format, errors='coerce')
    energy_data.dropna(subset=['Date'], inplace=True)
    energy_data.set_index('Date', inplace=True)

    if target_col not in energy_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in the file. Columns: {list(energy_data.columns)}")

    energy_data = energy_data[[target_col]].dropna()
    energy_data = energy_data.sort_index()
    return energy_data


def fetch_weather_data(power_data, longitude, latitude):
    """Fetch and process historical weather data and align to 15-minute grid."""
    start_date = power_data.index.min().strftime('%Y-%m-%d')
    end_date = power_data.index.max().strftime('%Y-%m-%d')
    weather_data = get_weather_data(longitude, latitude, data_type="historical",
                                    start_date=start_date, end_date=end_date)
    weather_data = weather_data.resample("15min").mean().interpolate(method='linear')
    return weather_data


def generate_time_features(df):
    df['Hour of day'] = df.index.hour
    df['Day of week'] = df.index.weekday
    df['Month'] = df.index.month
    df['year'] = df.index.year
    df['Is Weekend'] = (df['Day of week'] >= 5).astype(int)
    return df


def detect_holidays(index):
    dutch_holidays = holidays.Netherlands()
    return index.map(lambda x: x in dutch_holidays).astype(int)


def process_file(file_like_or_path, date_format, longitude, latitude, target_col):
    energy_data = load_and_clean_data(file_like_or_path, date_format, target_col)
    weather_data = fetch_weather_data(energy_data, longitude, latitude)
    combined_data = energy_data.join(weather_data, how="inner")
    combined_data = generate_time_features(combined_data)
    combined_data['Holiday'] = detect_holidays(combined_data.index)
    combined_data = combined_data.sort_index()
    return weather_data, energy_data, combined_data


def perform_back_testing(combined_data, target_col, train_window=365*96, forecast_horizon=192, data_lag_steps=96):
    backtest_results = rolling_backtest(combined_data, target_col,
                                        train_window=train_window,
                                        forecast_horizon=forecast_horizon,
                                        data_lag_steps=data_lag_steps)
    return backtest_results


def get_results(backtest_results):
    y_true_all = np.concatenate([res['y_true'] for res in backtest_results])
    y_pred_all = np.concatenate([res['y_pred'] for res in backtest_results])
    test_indices = np.concatenate([res['test_indices'] for res in backtest_results])
    y_pred_all = pd.DataFrame(y_pred_all, index=test_indices, columns=['y_pred'])
    y_true_all = pd.DataFrame(y_true_all, index=test_indices, columns=['y_true'])
    return y_true_all, y_pred_all


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode('utf-8')


def _df_to_excel_bytes(df_dict: dict, filename="results.xlsx") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    bio.seek(0)
    return bio.read()


def _embed_pdf(pdf_bytes: bytes, height: int = 800):
    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>
    """
    st.components.v1.html(pdf_display, height=height, scrolling=True)


# ------------------------------ UI Page ------------------------------

def render():
    st.header("Imbalance estimator – Backtesting")

    with st.expander("Inputs", expanded=True):
        uploaded = st.file_uploader("Upload data file (.csv or .xlsx) with a 'Date' column", type=['csv', 'xlsx'])
        cols = st.columns(2)
        with cols[0]:
            date_format = st.text_input("Date format", value="%Y/%m/%d %H:%M",
                                        help="Python strftime format of the 'Date' column.")
            target_col = st.text_input("Target column name", value="Total")
            latitude = st.text_input("Latitude", value="52.09")
        with cols[1]:
            longitude = st.text_input("Longitude", value="5.15")
            train_window = st.number_input("Train window (points)", value=20160, min_value=96, step=96)
            forecast_horizon = st.number_input("Forecast horizon (points)", value=96, min_value=1, step=1)
        data_lag_steps = st.number_input("Training data lag (points)", value=96, min_value=0, step=1)

        run_btn = st.button("Run backtest", type="primary", use_container_width=True, disabled=uploaded is None)

    if not run_btn:
        st.info("Upload a file and click **Run backtest** to start.")
        return

    if uploaded is None:
        st.error("Please upload a .csv or .xlsx file.")
        return

    try:
        with st.status("Processing input and fetching weather…", expanded=False) as status:
            weather_data, energy_data, combined_data = process_file(
                uploaded, date_format, longitude, latitude, target_col
            )
            status.update(label="Running rolling backtest…", state="running")
            backtest_results = perform_back_testing(
                combined_data, target_col,
                train_window=int(train_window),
                forecast_horizon=int(forecast_horizon),
                data_lag_steps=int(data_lag_steps)
            )
            status.update(label="Collecting predictions and metrics…", state="running")
            y_true_all, y_pred_all = get_results(backtest_results)

        # Filter from 2025-01-01 as in your script
        y_true_all = y_true_all[y_true_all.index >= "2025-01-01"]
        y_pred_all = y_pred_all[y_pred_all.index >= "2025-01-01"]

        # Error metrics
        errors = (y_pred_all['y_pred'] - y_true_all['y_true']).to_frame('error')
        abs_errors = errors['error'].abs()
        std_error = errors['error'].std()
        mae = abs_errors.mean()
        rmse = np.sqrt((errors['error']**2).mean())
        # Normalizations (guard against division by zero)
        y_true_abs_mean = np.abs(y_true_all['y_true']).mean()
        y_range = y_true_all['y_true'].max() - y_true_all['y_true'].min()
        normalized_mae = float(mae / y_true_abs_mean) if y_true_abs_mean not in (0, np.nan) else np.nan
        normalized_rmse = float(rmse / y_range) if y_range not in (0, np.nan) else np.nan

        st.success("Backtest complete.")

        st.subheader("Key results")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE", f"{mae:.3f}")
        k2.metric("RMSE", f"{rmse:.3f}")
        k3.metric("N-MAE", f"{normalized_mae:.3f}" if not np.isnan(normalized_mae) else "—")
        k4.metric("Std. error", f"{std_error:.3f}")

        st.subheader("Actual vs Predicted")
        joined = y_true_all.join(y_pred_all, how="inner")
        st.line_chart(joined)

        with st.expander("View tables"):
            st.dataframe(joined, use_container_width=True)
            st.dataframe(errors, use_container_width=True)

        # Imbalance costs + plot (your function handles plotting internally)
        st.subheader("Imbalance cost calculation")
        total_cost = calculate_and_plot_imbalance(y_true_all, y_pred_all, target_col)
        # ensure naive datetime index for saving (like your script)
        try:
            total_cost.index = total_cost.index.tz_localize(None)
        except Exception:
            pass

        st.dataframe(total_cost.tail(200), use_container_width=True)

        # Generate PDF report to bytes
        st.subheader("PDF report")
        pdf_buf = io.BytesIO()
        # many functions write to disk; we allow it to write a temp file and re-read
        tmp_pdf_name = "imbalance_report.pdf"
        generate_pdf_report(total_cost, y_true_all, y_pred_all, target_col, tmp_pdf_name)
        with open(tmp_pdf_name, "rb") as f:
            pdf_bytes = f.read()
        # show inline
        _embed_pdf(pdf_bytes, height=800)

        # ---------- Downloads ----------
        st.subheader("Download results")
        # Predictions CSV
        st.download_button(
            "Download predictions (CSV)",
            data=_df_to_csv_bytes(joined),
            file_name="predictions_actuals.csv",
            mime="text/csv",
            use_container_width=True
        )
        # Errors CSV
        st.download_button(
            "Download errors (CSV)",
            data=_df_to_csv_bytes(errors),
            file_name="errors.csv",
            mime="text/csv",
            use_container_width=True
        )
        # Total cost XLSX
        xlsx_bytes = _df_to_excel_bytes({"total_cost": total_cost})
        st.download_button(
            "Download total_cost (XLSX)",
            data=xlsx_bytes,
            file_name="total_cost.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        # PDF
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="imbalance_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.exception(e)
