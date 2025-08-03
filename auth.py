import streamlit as st
import os
import json
import hashlib

# === Setup ===
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(page_title="Solar Simulation Model", layout="centered")


# === Utilities ===
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# === Login Check ===
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        show_login()
        st.stop()

# === Show Branded Landing Page ===
def show_login():

    if st.session_state.get("show_login_form"):
        login_page()  # Show actual login/signup
        return

    # CSS and Hero Section
    st.markdown("""
    <style>
        .main { background-color: #f5fafd; }
        .hero {
            text-align: center;
            margin-top: 4rem;
        }
        .hero h1 {
            font-size: 3rem;
            color: #1c1c1c;
        }
        .hero h1 span {
            color: #28a745;
        }
        .hero p {
            font-size: 1.1rem;
            color: #444;
            margin-top: 0.5rem;
        }
        .card-container {
            display: flex;
            justify-content: center;
            margin-top: 3rem;
            gap: 2rem;
        }
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            width: 260px;
        }
        .card h4 {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .card p {
            font-size: 0.95rem;
            color: #555;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #888;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1>Solar Simulation <span>Model</span></h1>
        <p>Generate accurate historical solar generation profiles for PV plants. <br>
        <br>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Sign In to Continue", use_container_width=True):
            st.session_state.show_login_form = True
            st.rerun()

    st.markdown("""
    <div class="card-container">
        <div class="card">
            <div style="width:30px;height:30px;background:#d6f5e3;border-radius:6px;"></div>
            <h4>Historical Analysis</h4>
            <p>Generate detailed historical solar generation profiles based on meteorological data and PV system parameters.</p>
        </div>
        <div class="card">
            <div style="width:30px;height:30px;background:#d8ebfd;border-radius:6px;"></div>
            <h4>Location Specific</h4>
            <p>Accurate simulation using precise latitude, longitude coordinates and local solar irradiance patterns.</p>
        </div>
        <div class="card">
            <div style="width:30px;height:30px;background:#fff3c1;border-radius:6px;"></div>
            <h4>Export Reports</h4>
            <p>Download comprehensive reports in PDF and CSV formats for client presentations and analysis.</p>
        </div>
    </div>
    <div class="footer">
    </div>
    """, unsafe_allow_html=True)

# === Actual Login/Signup Page ===
def login_page():
    st.title("🔐 Sign In to SolarForecast")

    tab = st.radio("Choose an option:", ["Login", "Sign Up"], horizontal=True)

    if tab == "Login":
        st.subheader("Login to your account")
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            login_clicked = st.button("🔑 Login", use_container_width=True, type="primary")
        
        # Check for Enter key press by monitoring if both fields are filled and there's any interaction
        enter_pressed = (email and password and 
                        (st.session_state.get("login_email", "") != "" and 
                         st.session_state.get("login_password", "") != ""))

        if login_clicked or (enter_pressed and st.session_state.get("last_login_attempt") != f"{email}:{password}"):
            if enter_pressed:
                st.session_state["last_login_attempt"] = f"{email}:{password}"
            
            users = load_users()
            if email in users and users[email] == hash_password(password):
                st.session_state["logged_in"] = True
                st.success("Login successful. Redirecting...")
                st.rerun()
            else:
                st.error("Invalid email or password")

        st.caption("💡 Fill in both fields and press Enter, or click the Login button")

    elif tab == "Sign Up":
        st.subheader("Create a new account")
        
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password") 
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            signup_clicked = st.button("✨ Sign Up", use_container_width=True, type="primary")

        if signup_clicked:
            users = load_users()
            if new_email in users:
                st.error("Email already exists")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.warning("Password must be at least 6 characters long")
            else:
                users[new_email] = hash_password(new_password)
                save_users(users)
                st.success("Account created. Please log in.")
                st.rerun()

        st.caption("💡 Fill in all fields and click Sign Up")
