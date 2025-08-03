# pages/settings.py

import streamlit as st
from auth import check_login

def render():
    # 🔐 Enforce login
    check_login()


    # ⚙️ Page content
    st.title("⚙️ Settings")
    st.info("Account and configuration settings will appear here.")
