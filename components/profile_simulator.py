# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:30:50 2025

@author: waaqa
"""

import streamlit as st

def render():
    st.title("📊 Profile Forecast Simulator")
    st.write("This page will allow to simulate and forecast the profile of user who doesnot have any data.")

    # Example placeholder input
    # st.number_input("Simulated capacity (MW)", 1, 100, 10)
    st.button("Run Simulation")
