import streamlit as st
from datetime import datetime
import io

from pv_simulator import (
    get_weather_data,
    map_azimuth_to_api_angle,
    handle_special_cases,
    calculate_combined_pv_yield
)
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* Reduce left padding to bring content closer to sidebar */
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Optional: make main content area wider */
    .main {
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Match sidebar background */
    section[data-testid="stSidebar"] {
        background-color: white !important;
    }

    .logout-button button {
        width: 100%;
        background-color: #f44336 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

def render():
    # --- Dashboard Section ---
    st.title(" Solar Dashboard")
    
    projects = st.session_state.get("projects", [])
    total_projects = len(projects)
    total_capacity = sum(p.get('kWp_DC', 0) for p in projects)
    avg_generation = 0  # Placeholder as before
    active_forecasts = len(projects)
    
    with st.container():
        st.markdown(
            """
            <style>
            .metric-box {
                background: #f0f2f6;
                border-radius: 10px;
                padding: 20px 15px;
                text-align: center;
                box-shadow: 0 2px 6px rgb(0 0 0 / 0.1);
                transition: background-color 0.3s ease;
            }
            .metric-box:hover {
                background-color: #e1e5ef;
            }
            .metric-title {
                font-size: 0.9rem;
                color: #555;
                margin-bottom: 5px;
                font-weight: 600;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #111;
            }
            .col-1 {background-color: #ffebe6;}
            .col-2 {background-color: #e6f4ff;}
            .col-3 {background-color: #e6fff7;}
            .col-4 {background-color: #fff9e6;}
            </style>
            """, unsafe_allow_html=True)

        cols = st.columns(4)
        metrics = [
            ("Total Projects", total_projects, "col-1"),
            ("Total Capacity", f"{total_capacity:.1f} kWp", "col-2"),
            ("Avg Generation", f"{avg_generation} kWh", "col-3"),
            ("Active Forecasts", active_forecasts, "col-4"),
        ]
        for col, (title, value, color_class) in zip(cols, metrics):
            col.markdown(f'<div class="metric-box {color_class}">'
                         f'<div class="metric-title">{title}</div>'
                         f'<div class="metric-value">{value}</div>'
                         f'</div>', unsafe_allow_html=True)
    
    
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<hr style="border:1.5px solid #888; border-radius: 2px;">',
        unsafe_allow_html=True
    )


    with st.expander("❓ How to use this dashboard"):
        st.markdown(
            """
            1. Fill in the project details on the left panel:
            
            - Enter the project name, location (latitude & longitude), and operational years of the solar plant.
            
            - Specify the total inverter capacity (AC, kWp) **if you have only one plant**.
            
            - Specify the number of PV plants and their individual configurations if you have **multiple plants**. For each plant, enter:
                - DC capacity (kWp)
                - AC capacity (kWp) — leave blank if you have specified the total inverter capacity above
                - Tilt angle (°)
                - Azimuth angle (°)
                
            - Use these configurations according to your setup:
                - **Single plant with one inverter:** Specify either total inverter capacity or AC capacity in Plant 1.
                - **Multiple panels facing different directions with one total inverter:** Specify total inverter capacity, and provide DC capacities and orientations for each panel.
                - **Multiple plants with multiple inverters:** Specify individual AC capacities for each plant.
                
            2. Click **Run Simulation** to generate historical solar generation data based on weather info.
            
            3. View the generated solar yield chart and recent data on the right panel.
            
            4. Download the results as a CSV file for further analysis.
            
            *Tip: The azimuth angle is the compass direction your panels face (0° = North, 180° = South).*
            """
        )



    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Forecast Section: form on left, results on right ---
    left_col, right_col = st.columns([4, 6])  # Left smaller for form, right wider for results

    with left_col:
        st.header(" Generate Solar Historical Generation")
        
        # ✅ This is now outside the form
        num_plants = st.number_input(
            "Number of PV Plants", value=1, step=1, min_value=1, key="num_plants_selector"
        )

        with st.form("forecast_form"):
            project_name = st.text_input("Project Name", "Example PV Project")
            start_date = st.date_input("Start Date", datetime(2025, 1, 1))
            end_date = st.date_input("End Date", datetime(2025, 1, 10))
            latitude = st.number_input("Latitude", value=52.37)
            longitude = st.number_input("Longitude", value=4.89)
            years_operational = st.number_input("Years Operational", value=1)
            kWp_AC = st.number_input("Total Inverter Capacity (AC, kWp)", value=0)

            dc_list, ac_list, tilt_list, azimuth_list = [], [], [], []

            for i in range(num_plants):
                st.markdown(f"---\n**Plant {i+1} Configuration**")
                dc = st.number_input(f"DC Capacity [kWp] - Plant {i+1}", value=10.0, key=f"dc_{i}_np")
                ac = st.number_input(f"AC Capacity [kW] - Plant {i+1}", value=8.0, key=f"ac_{i}_np")
                tilt = st.number_input(f"Tilt (°) - Plant {i+1}", value=30.0, key=f"tilt_{i}_np")
                az_conv = st.number_input(f"Azimuth (Compass °) - Plant {i+1}", value=180.0, key=f"az_{i}_np")

                az_api = map_azimuth_to_api_angle(az_conv)

                dc_list.append(dc)
                ac_list.append(ac)
                tilt_list.append(tilt)
                azimuth_list.append(az_api)

            submitted = st.form_submit_button("🔍 Run Simulation")

    if submitted:
        tilt_list, azimuth_list = handle_special_cases(tilt_list, azimuth_list)
        dfs = []

        for i in range(num_plants):
            df = get_weather_data(
                latitude, longitude, tilt_list[i], azimuth_list[i],
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            dfs.append(df)

        result_df = calculate_combined_pv_yield(
            dfs, years_operational, dc_list, kWp_AC, kWp_ACs=ac_list
        )

        # Save project to session
        project = {
            "name": project_name,
            "latitude": latitude,
            "longitude": longitude,
            "kWp_DC": sum(dc_list),
            "kWp_AC": kWp_AC,
            "start_date": start_date,
            "end_date": end_date,
            "df": result_df,
        }
        if "projects" not in st.session_state:
            st.session_state["projects"] = []
        st.session_state["projects"].append(project)

        with right_col:
            st.success("✅ Simulation Complete!")
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)  # adds 2 line breaks
            st.line_chart(result_df.filter(like="PV_yield"))
            st.dataframe(result_df.tail(5))

            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=True)
            st.download_button("📥 Download CSV", csv_buffer.getvalue(), "pv_result.csv", "text/csv")
    else:
        with right_col:
            st.markdown("<br><br><br>", unsafe_allow_html=True)  # vertical space before info

            st.info("Fill out the form on the left and run simulation to see results here.")


    st.sidebar.markdown(
        """
        <style>
        .fixed-logout {
            position: fixed;
            left: 0;
            bottom: 20px;
            width: 220px;
            z-index: 9999;
        }
        </style>
        <div class="fixed-logout">l
        """,
        unsafe_allow_html=True
    )
    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.show_login_form = False
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
