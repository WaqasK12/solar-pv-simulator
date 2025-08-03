# app.py

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from auth import check_login
import importlib
import os

# ---- LOGIN ----
check_login()

st.markdown("""
    <style>
        /* Remove padding around main content */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            background-color: white;
        }

        /* Set background for whole page and sidebar */
        body, .main, .css-18e3th9, .css-1d391kg, .stSidebar {
            background-color: white;
        }

        /* Optional: remove borders between sidebar and main content */
        .stSidebar {
            border-right: none;
        }
    </style>
""", unsafe_allow_html=True)


# Only run this part if the user is logged in
if st.session_state.get("logged_in"):

    # # Sidebar logout button
    # if st.sidebar.button("🚪 Logout"):
    #     st.session_state["logged_in"] = False
    #     st.rerun()

    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Dashboard", "Historical Data", "Reports", "Settings"],
            icons=["bar-chart", "plus-circle", "clock-history", "file-earmark-text", "gear"],
            menu_icon="cast",
            default_index=0,
        )

    # Load and display solar panel image at top of every page
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "images", "solar_panels.png")
    image = Image.open(image_path)

    st.image(image, use_container_width=True)  # full width image at top

    # Dynamically import selected page
    page_module_map = {
        "Dashboard": "components.dashboard",
        "Historical Data": "components.history",
        "Reports": "components.reports",
        "Settings": "components.settings",
    }

    page_module = importlib.import_module(page_module_map[selected])
    page_module.render()
