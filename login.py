# login.py
import streamlit as st
import pyrebase
import firebase_admin
import re
from firebase_config import auth
from firebase_admin import credentials, firestore
from streamlit_js_eval import streamlit_js_eval  # pip install streamlit-js-eval

# Firebase Config
firebaseConfig = {
    "apiKey": "AIzaSyBg8HSYkcrhvA1zOsv7XOcrQlH3uwfw2sc",
    "authDomain": "stockforecasting1107.firebaseapp.com",
    "projectId": "stockforecasting1107",
    "storageBucket": "stockforecasting1107.appspot.com",
    "messagingSenderId": "528666678237",
    "appId": "1:528666678237:web:0c4958b6dacf2898fc5c34",
    "databaseURL": "https://stockforecasting1107-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)

firestore_db = firestore.client()

# Restore session from localStorage
def restore_session():
    js_user = streamlit_js_eval(js_expressions="localStorage.getItem('user')", key="restore_user")
    if js_user and isinstance(js_user, str):
        try:
            user = eval(js_user)
            st.session_state["authenticated"] = True
            st.session_state["user"] = user
        except:
            pass

def show_login():
    restore_session()

    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        .stTabs [role="tab"] {
            background-color: #f4f6f9;
            padding: 8px 20px;
            margin-right: 10px;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #ffffff;
            border-bottom: 2px solid #1a73e8;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center'>🔐 Welcome to Forecast App</h3>", unsafe_allow_html=True)

    if "login_tab" not in st.session_state:
        st.session_state.login_tab = 0

    tabs = st.tabs(["🔐 Login", "✍️ Register"])

    # --------------------- LOGIN ---------------------
    with tabs[0]:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            remember = st.checkbox("Remember Me", key="login_remember")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
                    st.error("❗ Enter a valid email address.")
                elif not password:
                    st.error("❗ Please enter your password.")
                else:
                    try:
                        user = auth.sign_in_with_email_and_password(email, password)
                        uid = user['localId']
                        user_info = firestore_db.collection("users").document(uid).get()
                        if user_info.exists:
                            user_data = user_info.to_dict()
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user_data
                            if remember:
                                streamlit_js_eval(js_expressions=f"localStorage.setItem('user', `{user_data}`)", key="set_user")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Login Failed: {e}")

        with st.expander("Forgot Password?"):
            with st.form("forgot_form"):
                reset_email = st.text_input("Enter your email to reset")
                reset_submit = st.form_submit_button("Send Reset Link")
                if reset_submit:
                    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", reset_email):
                        st.error("❗ Enter a valid email address.")
                    else:
                        try:
                            auth.send_password_reset_email(reset_email)
                            st.success("📧 Reset link sent to your email.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

    # --------------------- REGISTER ---------------------
    with tabs[1]:
        with st.form("register_form"):
            first = st.text_input("First Name")
            last = st.text_input("Last Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            remember = st.checkbox("Remember Me", key="register_remember")
            register_btn = st.form_submit_button("Register")

            if register_btn:
                if not first.strip() or not last.strip():
                    st.error("❗ Please fill in your full name.")
                elif not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
                    st.error("❗ Enter a valid email address.")
                elif len(password) < 6:
                    st.error("❗ Password must be at least 6 characters.")
                elif password != confirm:
                    st.error("❗ Passwords do not match.")
                else:
                    try:
                        user = auth.create_user_with_email_and_password(email, password)
                        uid = user['localId']
                        user_data = {
                            "first_name": first,
                            "last_name": last,
                            "email": email
                        }
                        firestore_db.collection("users").document(uid).set(user_data)
                        st.success("✅ Registered successfully. Redirecting to login...")
                        st.session_state.login_tab = 0
                        st.rerun()
                    except Exception as e:
                        if "EMAIL_EXISTS" in str(e):
                            st.error("❗ This email is already registered. Please login instead.")
                        else:
                            st.error(f"❌ Registration Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🔐 Sign in with Google (Coming Soon)"):
        st.info("⚙️ Stay tuned — Google login is coming soon!")

# Show user info
def show_user_header():
    if st.session_state.get("authenticated"):
        user = st.session_state.get("user", {})
        name = f"{user.get('first_name', '')} {user.get('last_name', '')}"
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(f"<h5 style='margin-bottom:0;'>👤 {name}</h5>", unsafe_allow_html=True)
        with col2:
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                streamlit_js_eval(js_expressions="localStorage.removeItem('user')", key="clear_user")
                st.rerun()
