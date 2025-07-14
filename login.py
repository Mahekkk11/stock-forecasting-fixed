# login.py
import streamlit as st
import re
from streamlit_js_eval import streamlit_js_eval  # pip install streamlit-js-eval

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
                    user_data = {
                        "first_name": "Demo",
                        "last_name": "User",
                        "email": email
                    }
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data
                    if remember:
                        streamlit_js_eval(js_expressions=f"localStorage.setItem('user', `{user_data}`)", key="set_user")
                    st.success("✅ Logged in successfully!")
                    st.rerun()

        with st.expander("Forgot Password?"):
            st.info("🔒 This feature is disabled. Please contact support.")

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
                    user_data = {
                        "first_name": first,
                        "last_name": last,
                        "email": email
                    }
                    st.success("✅ Registered successfully. Redirecting to login...")
                    st.session_state.login_tab = 0
                    st.rerun()

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
