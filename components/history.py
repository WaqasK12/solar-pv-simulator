# pages/history.py

import streamlit as st
from auth import check_login

def render():

    st.title("📁 Historical Forecasts")

    if "projects" in st.session_state:
        for i, proj in enumerate(st.session_state["projects"]):
            with st.expander(f"{proj['name']} ({proj['start_date']} - {proj['end_date']})"):
                st.write(f"Location: ({proj['latitude']}, {proj['longitude']})")
                st.write(f"Capacity: {proj['kWp_DC']} kWp DC / {proj['kWp_AC']} kWp AC")
                st.dataframe(proj["df"].tail(5))
                st.line_chart(proj["df"].filter(like="PV_yield"))
    else:
        st.info("No forecasts have been saved yet.")
