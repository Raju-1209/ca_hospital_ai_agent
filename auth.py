"""
auth.py — Login system with role-based access control
Roles: admin, doctor, analyst
"""

import streamlit as st
import hashlib

# ── User database (in production, use PostgreSQL) ─────────────────────────────
USERS = {
    "admin": {
        "password": hashlib.sha256("admin@123".encode()).hexdigest(),
        "role": "Admin",
        "name": "Administrator",
        "icon": "👑",
        "permissions": ["Dashboard", "Denial Prediction", "Readmission Prediction",
                        "Fraud Detection", "High-Cost Patient", "Length of Stay",
                        "Ask AI Agent", "User Management"],
    },
    "doctor": {
        "password": hashlib.sha256("doctor@123".encode()).hexdigest(),
        "role": "Doctor",
        "name": "Dr. Smith",
        "icon": "👨‍⚕️",
        "permissions": ["Dashboard", "Readmission Prediction", "High-Cost Patient",
                        "Length of Stay", "Ask AI Agent"],
    },
    "analyst": {
        "password": hashlib.sha256("analyst@123".encode()).hexdigest(),
        "role": "Analyst",
        "name": "Data Analyst",
        "icon": "📊",
        "permissions": ["Dashboard", "Denial Prediction", "Fraud Detection",
                        "High-Cost Patient", "Ask AI Agent"],
    },
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str):
    username = username.strip().lower()
    if username in USERS:
        if USERS[username]["password"] == hash_password(password):
            return USERS[username]
    return None


def show_login_page():
    """Render the login page UI."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0a0f1e; }
    [data-testid="stSidebar"] { display: none; }

    .login-container {
        max-width: 440px;
        margin: 60px auto;
        background: linear-gradient(135deg, #0f1e3d, #1a2a4a);
        border: 1px solid #2a4080;
        border-radius: 20px;
        padding: 40px;
    }
    .login-title {
        font-size: 28px;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
        margin-bottom: 4px;
    }
    .login-sub {
        font-size: 14px;
        color: #64748b;
        text-align: center;
        margin-bottom: 32px;
    }
    .role-card {
        background: #0a1628;
        border: 1px solid #1e3a6e;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #94a3b8;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a56db, #0e9de0) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 14px !important;
        width: 100% !important;
        margin-top: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <span style='font-size:64px'>🏥</span>
            <div style='font-size:32px; font-weight:700; color:#60a5fa; margin-top:8px;'>
                CA Hospital
            </div>
            <div style='font-size:16px; color:#64748b; margin-bottom:32px;'>
                AI Agent Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("#### 🔐 Sign In")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            login_btn = st.button("Login →", use_container_width=True)

            if login_btn:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = check_credentials(username, password)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user"]          = user
                        st.session_state["username"]      = username
                        st.success(f"Welcome, {user['name']}! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

        st.markdown("---")
        st.markdown("#### 👥 Demo Credentials")
        st.markdown("""
        <div class="role-card">👑 <b>Admin</b> — username: <code>admin</code> &nbsp;|&nbsp; password: <code>admin@123</code></div>
        <div class="role-card">👨‍⚕️ <b>Doctor</b> — username: <code>doctor</code> &nbsp;|&nbsp; password: <code>doctor@123</code></div>
        <div class="role-card">📊 <b>Analyst</b> — username: <code>analyst</code> &nbsp;|&nbsp; password: <code>analyst@123</code></div>
        """, unsafe_allow_html=True)


def require_login():
    """Call at top of app.py — redirects to login if not authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login_page()
        st.stop()


def get_current_user():
    return st.session_state.get("user", {})


def has_permission(page_name: str) -> bool:
    user = get_current_user()
    return page_name in user.get("permissions", [])


def show_user_badge():
    """Show logged-in user info + logout button in sidebar."""
    user = get_current_user()
    st.markdown(f"""
    <div style='background:#0a1628; border:1px solid #1e3a6e; border-radius:10px;
                padding:12px 16px; margin-bottom:8px;'>
        <div style='font-size:20px'>{user.get('icon','👤')}
            <span style='font-size:14px; font-weight:600; color:#93c5fd; margin-left:6px;'>
                {user.get('name','')}
            </span>
        </div>
        <div style='font-size:12px; color:#64748b; margin-top:4px;'>
            Role: {user.get('role','')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        for key in ["authenticated", "user", "username", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()
