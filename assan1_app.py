# assan1_app.py
# ==============================================
# ASSAN - STREAMLIT PORT OF COMPLETE PRODUCTIVITY APP
# Full Implementation with UI styled like Firebase Studio
# ==============================================

import os
import io
import time
import threading
import queue
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Optional TensorFlow import - safe fallback if missing
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Streamlit imports
import streamlit as st
from streamlit.elements import legacy_data_frame  # no-op, just to help linters

# ---------- FILES ----------
USER_FILE = "username.txt"
DATA_FILE = "tasks.csv"
REMINDER_LOG = "reminders.csv"
HABITS_FILE = "habits.csv"
USERS_FILE = "users.csv"
COMMENTS_FILE = "comments.csv"
MODEL_FILE = "model.h5"
SCALER_FILE = "scaler.pkl"

# ---------- APP STYLING (Firebase-like) ----------
FIREBASE_COLORS = {
    "blue": "#4C6EF5",
    "blue2": "#5C7CFA",
    "bg1": "#0F0F1A",
    "bg2": "#15161F",
    "bg3": "#1E1F2B",
    "muted": "#C9C9D1",
    "bright": "#F4F4F9"
}

st.set_page_config(page_title="ASSAN — Productivity", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS: fonts, background, cards, buttons
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
        color: {FIREBASE_COLORS['bright']};
        background: linear-gradient(135deg, {FIREBASE_COLORS['bg1']} 0%, {FIREBASE_COLORS['bg2']} 40%, {FIREBASE_COLORS['bg3']} 100%);
    }}

    .stApp {{
        background: linear-gradient(135deg, {FIREBASE_COLORS['bg1']} 0%, {FIREBASE_COLORS['bg2']} 40%, {FIREBASE_COLORS['bg3']} 100%);
    }}

    /* Card */
    .card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }}

    /* Heading */
    .app-title {{
        font-size: 28px;
        font-weight: 700;
        color: {FIREBASE_COLORS['bright']};
        margin-bottom: 6px;
        text-shadow: 0 2px 12px rgba(76,110,245,0.08);
    }}

    .app-sub {{
        color: {FIREBASE_COLORS['muted']};
        margin-bottom: 12px;
    }}

    /* Buttons */
    .btn {{
        background: linear-gradient(90deg, {FIREBASE_COLORS['blue']}, {FIREBASE_COLORS['blue2']});
        color: white !important;
        padding: 8px 14px;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(76,110,245,0.14);
    }}

    .btn:active {{ transform: translateY(1px); }}

    /* Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div>div {{
        background: #141421;
        border-radius: 10px;
        color: {FIREBASE_COLORS['bright']};
        border: 1px solid rgba(255,255,255,0.06);
        padding: 10px;
    }}

    /* Table headers */
    .dataframe thead th {{
        background: rgba(255,255,255,0.03);
        color: {FIREBASE_COLORS['bright']};
    }}

    /* Small muted text */
    .muted-small {{ color: {FIREBASE_COLORS['muted']}; font-size:13px; }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- CATEGORIES ----------
CATEGORIES = {
    "Work": {"icon": "🏢", "color": "#4C6EF5"},
    "Personal": {"icon": "🏠", "color": "#34A853"},
    "Learning": {"icon": "🎓", "color": "#7C4DFF"},
    "Health": {"icon": "💪", "color": "#EA4335"},
    "Creative": {"icon": "🎨", "color": "#FF69B4"},
    "Other": {"icon": "📌", "color": "#C0C0C0"},
}

# ---------- GLOBALS ----------
if "reminder_queue" not in st.session_state:
    st.session_state.reminder_queue = queue.Queue()
if "shown_reminders" not in st.session_state:
    st.session_state.shown_reminders = set()
if "reminder_thread" not in st.session_state:
    st.session_state.reminder_thread = None
if "stop_reminder_thread" not in st.session_state:
    st.session_state.stop_reminder_thread = threading.Event()

# ---------- UTILITIES ----------
def now_dt():
    return datetime.now()

def parse_deadline_input(inp):
    s = str(inp).strip()
    if not s:
        return pd.NaT
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%H:%M:%S", "%H:%M"]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if f in ("%H:%M", "%H:%M:%S"):
                today = datetime.now().date()
                dt = datetime.combine(today, dt.time())
            elif f == "%Y-%m-%d":
                dt = datetime.combine(dt.date(), datetime.max.time()).replace(microsecond=0)
            return pd.Timestamp(dt)
        except Exception:
            continue
    parsed = pd.to_datetime(s, errors='coerce')
    return parsed if not pd.isna(parsed) else pd.NaT

def time_left_parts(deadline_ts):
    if pd.isna(deadline_ts):
        return ("No deadline", None)
    now = now_dt()
    dl = deadline_ts.to_pydatetime() if isinstance(deadline_ts, pd.Timestamp) else deadline_ts
    diff = dl - now
    secs = int(diff.total_seconds())
    if secs <= 0:
        return ("OVERDUE", secs)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return (" ".join(parts), secs)

def parse_tags(tag_string):
    if pd.isna(tag_string) or str(tag_string).strip() == "":
        return []
    return [t.strip().lower() for t in str(tag_string).split(",") if t.strip()]

def format_tags(tags_list):
    if not tags_list:
        return ""
    return "[" + ", ".join(tags_list) + "]"

def get_category_display(category):
    if category not in CATEGORIES:
        category = "Other"
    cat_info = CATEGORIES[category]
    return f"{cat_info['icon']} {category}"

# ---------- DATA LOAD / SAVE ----------
@st.cache_data(ttl=600)
def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def load_dataframes():
    # Users
    df_users = safe_read_csv(USERS_FILE)
    if df_users.empty:
        df_users = pd.DataFrame(columns=["username","role","added_at","added_by"])
    else:
        if "added_at" in df_users.columns:
            df_users["added_at"] = pd.to_datetime(df_users["added_at"], errors='coerce')

    # Tasks
    df_tasks = safe_read_csv(DATA_FILE, low_memory=False)
    if df_tasks.empty:
        df_tasks = pd.DataFrame(columns=["id","task","priority","status","created_at","completed_at","deadline","ai_prediction","category","tags","habit_id","recurrence","assigned_to","created_by","shared"])
    else:
        for c in ["created_at","completed_at","deadline"]:
            if c in df_tasks.columns:
                df_tasks[c] = pd.to_datetime(df_tasks[c], errors='coerce')
        # enforce columns
        for col, default in [("category","Other"),("tags",""),("ai_prediction", np.nan),("habit_id", np.nan),("recurrence","none"),("assigned_to", ""),("created_by",""),("shared", False)]:
            if col not in df_tasks.columns:
                df_tasks[col] = default

    # Habits
    df_habits = safe_read_csv(HABITS_FILE)
    if df_habits.empty:
        df_habits = pd.DataFrame(columns=["habit_id","habit_name","recurrence","category","active","created_at","last_completed","total_completions"])
    else:
        for c in ["created_at","last_completed"]:
            if c in df_habits.columns:
                df_habits[c] = pd.to_datetime(df_habits[c], errors='coerce')

    # Comments
    df_comments = safe_read_csv(COMMENTS_FILE)
    if df_comments.empty:
        df_comments = pd.DataFrame(columns=["comment_id","task_id","username","comment","timestamp"])
    else:
        if "timestamp" in df_comments.columns:
            df_comments["timestamp"] = pd.to_datetime(df_comments["timestamp"], errors='coerce')

    # Reminders log
    df_reminders = safe_read_csv(REMINDER_LOG)
    if df_reminders.empty:
        df_reminders = pd.DataFrame(columns=["task_id","task_name","alert_type","timestamp"])
    else:
        if "timestamp" in df_reminders.columns:
            df_reminders["timestamp"] = pd.to_datetime(df_reminders["timestamp"], errors='coerce')

    return df_users, df_tasks, df_habits, df_comments, df_reminders

def save_all(df_tasks, df_habits, df_users, df_comments, df_reminders):
    df_tasks.to_csv(DATA_FILE, index=False)
    df_habits.to_csv(HABITS_FILE, index=False)
    df_users.to_csv(USERS_FILE, index=False)
    df_comments.to_csv(COMMENTS_FILE, index=False)
    df_reminders.to_csv(REMINDER_LOG, index=False)

# Load into session_state if not present
if "df_users" not in st.session_state:
    u,t,h,c,r = load_dataframes()
    st.session_state.df_users = u
    st.session_state.df_tasks = t
    st.session_state.df_habits = h
    st.session_state.df_comments = c
    st.session_state.df_reminders = r
    st.session_state.task_id_counter = int(t["id"].max()) + 1 if (not t.empty and "id" in t.columns and pd.notna(t["id"].max())) else 1
    st.session_state.habit_id_counter = int(h["habit_id"].max()) + 1 if (not h.empty and "habit_id" in h.columns and pd.notna(h["habit_id"].max())) else 1
    st.session_state.comment_id_counter = int(c["comment_id"].max()) + 1 if (not c.empty and "comment_id" in c.columns and pd.notna(c["comment_id"].max())) else 1

# ---------- ML helpers ----------
def train_simple_model(df_tasks):
    d = df_tasks.copy()
    if d.empty or len(d) < 5:
        return None
    d["priority_num"] = d["priority"].map({"Y":1,"N":0}).fillna(0).astype(int)
    d["task_len"] = d["task"].astype(str).apply(len)
    d["completed"] = (d["status"]=="Completed").astype(int)
    X = d[["priority_num","task_len"]]
    y = d["completed"]
    if len(y.unique()) < 2:
        return None
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

def predict_prob_local(row):
    # fallback simple heuristic
    return 0.8 if str(row.get("priority","N")).upper()=="Y" else 0.3

# ---------- REMINDERS THREAD ----------
def log_reminder(task_id, task_name, alert_type):
    dfr = st.session_state.df_reminders
    new = pd.DataFrame([{"task_id": int(task_id), "task_name": task_name, "alert_type": alert_type, "timestamp": pd.Timestamp.now()}])
    st.session_state.df_reminders = pd.concat([dfr, new], ignore_index=True)
    # Persist immediately
    st.session_state.df_reminders.to_csv(REMINDER_LOG, index=False)

def check_reminders_background(stop_event):
    while not stop_event.is_set():
        try:
            df_tasks = st.session_state.df_tasks
            if not df_tasks.empty:
                pending = df_tasks[(df_tasks["status"]=="Pending")]
                for _, task in pending.iterrows():
                    if pd.isna(task["deadline"]):
                        continue
                    task_id = int(task["id"])
                    _, secs = time_left_parts(task["deadline"])
                    if secs is None:
                        continue
                    reminder_key = None
                    alert_type = None
                    if secs <= 0:
                        reminder_key = f"{task_id}_overdue"
                        alert_type = "overdue"
                    elif secs <= 900:
                        reminder_key = f"{task_id}_urgent"
                        alert_type = "urgent"
                    elif secs <= 3600:
                        reminder_key = f"{task_id}_warning"
                        alert_type = "warning"
                    if reminder_key and (reminder_key not in st.session_state.shown_reminders):
                        st.session_state.reminder_queue.put({"task_id": task_id, "task_name": task["task"], "alert_type": alert_type})
                        st.session_state.shown_reminders.add(reminder_key)
                        log_reminder(task_id, task["task"], alert_type)
            time.sleep(30)
        except Exception:
            time.sleep(30)

def start_reminder_system():
    if st.session_state.reminder_thread is None or not st.session_state.reminder_thread.is_alive():
        st.session_state.stop_reminder_thread.clear()
        t = threading.Thread(target=check_reminders_background, args=(st.session_state.stop_reminder_thread,), daemon=True)
        st.session_state.reminder_thread = t
        t.start()

def stop_reminder_system():
    st.session_state.stop_reminder_thread.set()
    if st.session_state.reminder_thread:
        st.session_state.reminder_thread.join(timeout=1)

# Start reminder thread
start_reminder_system()

# ---------- UI: header & sidebar ----------
def header():
    st.markdown(
        """
        <div class="card">
            <div style="display:flex; align-items:center; justify-content:space-between">
                <div>
                    <div class="app-title">ASSAN</div>
                    <div class="app-sub">Personal and Team Productivity — Styled like Firebase Studio</div>
                </div>
                <div style="text-align:right">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:6px'>Welcome to ASSAN</h4>", unsafe_allow_html=True)
    # Username control (persist to file if changed)
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE,"r") as f:
                stored_name = f.read().strip()
        except Exception:
            stored_name = ""
    else:
        stored_name = ""
    if "current_user" not in st.session_state:
        st.session_state.current_user = stored_name or ""
    name = st.text_input("Your name", value=st.session_state.current_user, key="ui_name")
    if name and name != st.session_state.current_user:
        st.session_state.current_user = name
        try:
            with open(USER_FILE, "w") as f:
                f.write(name)
        except Exception:
            pass
    st.markdown(f"<div class='muted-small'>Signed in as <b>{st.session_state.current_user or '—'}</b></div>", unsafe_allow_html=True)

    st.markdown("---")
    menu = st.selectbox("Menu", [
        "Dashboard",
        "My Tasks",
        "Add Task",
        "Habits",
        "Team",
        "Analytics",
        "AI / Train",
        "Exports",
        "Reminders",
        "Settings"
    ])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- CORE FEATURES (UI-backed) ----------
def add_task_ui():
    st.markdown("<div class='card'><h3>Add Task</h3>", unsafe_allow_html=True)
    with st.form("add_task_form", clear_on_submit=True):
        name = st.text_input("Task name")
        pr = st.selectbox("Priority", ["N","Y"], index=0, format_func=lambda x: "High (Y)" if x=="Y" else "Normal (N)")
        dl_date = st.date_input("Deadline date (optional)", value=None)
        dl_time = st.time_input("Deadline time (optional)", value=None)
        cats = list(CATEGORIES.keys())
        cat = st.selectbox("Category", cats, index=cats.index("Other"))
        tags = st.text_input("Tags (comma separated)")
        submit = st.form_submit_button("Add task")
        if submit:
            dl_ts = pd.NaT
            if dl_date and dl_time:
                try:
                    dt = datetime.combine(dl_date, dl_time)
                    dl_ts = pd.Timestamp(dt)
                except Exception:
                    dl_ts = pd.NaT
            # predict prob (heuristic)
            prob = predict_prob_local({"priority":pr, "task":name, "deadline":dl_ts, "category":cat})
            df = st.session_state.df_tasks
            new = pd.DataFrame([{
                "id": int(st.session_state.task_id_counter),
                "task": name,
                "priority": pr,
                "status": "Pending",
                "created_at": pd.Timestamp.now(),
                "completed_at": pd.NaT,
                "deadline": dl_ts,
                "ai_prediction": round(prob*100,2),
                "category": cat,
                "tags": tags,
                "habit_id": np.nan,
                "recurrence": "none",
                "assigned_to": st.session_state.current_user,
                "created_by": st.session_state.current_user,
                "shared": False
            }])
            st.session_state.df_tasks = pd.concat([df, new], ignore_index=True)
            st.session_state.task_id_counter += 1
            st.success(f"Added task: {name}")
            # persist
            st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
    st.markdown("</div>", unsafe_allow_html=True)

def view_tasks_ui():
    st.markdown("<div class='card'><h3>My Tasks</h3>", unsafe_allow_html=True)
    my = st.session_state.df_tasks.copy()
    if my.empty:
        st.info("No tasks yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    # Filter by assigned user
    my = my[my["assigned_to"] == st.session_state.current_user] if st.session_state.current_user else my
    # Show interactive table with action buttons per row
    my_display = my[["id","task","status","priority","category","tags","deadline","ai_prediction"]].copy()
    my_display["deadline"] = my_display["deadline"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
    my_display["AI %"] = my_display["ai_prediction"].round(0).astype(str) + "%"
    my_display = my_display.rename(columns={"id":"ID","task":"Task","status":"Status","priority":"Pri","category":"Category","tags":"Tags","deadline":"Deadline"})
    st.dataframe(my_display.reset_index(drop=True), use_container_width=True)
    # Actions
    st.markdown("---")
    cols = st.columns([1,1,1,2])
    with cols[0]:
        tid = st.number_input("Task ID to complete", min_value=0, step=1, value=0)
        if st.button("Mark Completed"):
            if tid in st.session_state.df_tasks["id"].values:
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"]==tid, "status"] = "Completed"
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"]==tid, "completed_at"] = pd.Timestamp.now()
                # habit handling
                task_row = st.session_state.df_tasks[st.session_state.df_tasks["id"]==tid].iloc[0]
                if not pd.isna(task_row.get("habit_id", np.nan)):
                    hid = int(task_row["habit_id"])
                    st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"]==hid, "total_completions"] = st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"]==hid, "total_completions"].fillna(0) + 1
                    st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"]==hid, "last_completed"] = pd.Timestamp.now()
                st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
                st.success(f"Task #{tid} marked completed.")
            else:
                st.error("Task ID not found.")
    with cols[1]:
        tidr = st.number_input("Task ID to remove", min_value=0, step=1, value=0, key="remove_tid")
        if st.button("Remove Task"):
            if tidr in st.session_state.df_tasks["id"].values:
                st.session_state.df_tasks = st.session_state.df_tasks[st.session_state.df_tasks["id"] != tidr]
                st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
                st.success(f"Removed task #{tidr}")
            else:
                st.error("Task ID not found.")
    with cols[2]:
        tide = st.number_input("Task ID to edit", min_value=0, step=1, value=0, key="edit_tid")
        if st.button("Edit Task"):
            if tide in st.session_state.df_tasks["id"].values:
                row = st.session_state.df_tasks[st.session_state.df_tasks["id"]==tide].iloc[0]
                new_name = st.text_input("New name", value=row["task"])
                new_pr = st.selectbox("Priority", ["N","Y"], index=0, key="edit_prio")
                new_deadline = st.text_input("Deadline (YYYY-MM-DD HH:MM)", value=str(row["deadline"]) if not pd.isna(row["deadline"]) else "")
                if st.button("Save Changes"):
                    if new_name:
                        st.session_state.df_tasks.loc[st.session_state.df_tasks["id"]==tide,"task"] = new_name
                    if new_pr in ["Y","N"]:
                        st.session_state.df_tasks.loc[st.session_state.df_tasks["id"]==tide,"priority"] = new_pr
                    if new_deadline:
                        parsed = parse_deadline_input(new_deadline)
                        if not pd.isna(parsed):
                            st.session_state.df_tasks.loc[st.session_state.df_tasks["id"]==tide,"deadline"] = parsed
                    st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
                    st.success("Updated task.")
            else:
                st.error("Task ID not found.")
    with cols[3]:
        tagsearch = st.text_input("Search tag")
        if st.button("Search Tag"):
            if not tagsearch:
                st.warning("Enter a tag")
            else:
                filtered = st.session_state.df_tasks[st.session_state.df_tasks["tags"].apply(lambda x: tagsearch.lower() in str(x).lower())]
                st.write(filtered[["id","task","tags","status"]])
    st.markdown("</div>", unsafe_allow_html=True)

def habits_ui():
    st.markdown("<div class='card'><h3>Habits</h3>", unsafe_allow_html=True)
    with st.form("create_habit", clear_on_submit=True):
        name = st.text_input("Habit name")
        rec = st.selectbox("Recurrence", ["daily","weekly","monthly"])
        cats = list(CATEGORIES.keys())
        cat = st.selectbox("Category", cats, index=cats.index("Health"))
        submit = st.form_submit_button("Create Habit")
        if submit and name:
            hid = int(st.session_state.habit_id_counter)
            new = pd.DataFrame([{
                "habit_id": hid,
                "habit_name": name,
                "recurrence": rec,
                "category": cat,
                "active": True,
                "created_at": pd.Timestamp.now(),
                "last_completed": pd.NaT,
                "total_completions": 0
            }])
            st.session_state.df_habits = pd.concat([st.session_state.df_habits, new], ignore_index=True)
            # create initial task for habit
            dl = now_dt() + timedelta(days=1)
            prob = predict_prob_local({"priority":"Y","task":name,"deadline":dl,"category":cat})
            t = st.session_state.df_tasks
            newt = pd.DataFrame([{
                "id": st.session_state.task_id_counter,
                "task": name,
                "priority":"Y",
                "status":"Pending",
                "created_at": pd.Timestamp.now(),
                "completed_at": pd.NaT,
                "deadline": dl,
                "ai_prediction": round(prob*100,2),
                "category": cat,
                "tags": f"habit,{rec}",
                "habit_id": hid,
                "recurrence": rec,
                "assigned_to": st.session_state.current_user,
                "created_by": st.session_state.current_user,
                "shared": False
            }])
            st.session_state.df_tasks = pd.concat([t, newt], ignore_index=True)
            st.session_state.habit_id_counter += 1
            st.session_state.task_id_counter += 1
            st.success("Created habit and initial task.")
            # persist
            save_all(st.session_state.df_tasks, st.session_state.df_habits, st.session_state.df_users, st.session_state.df_comments, st.session_state.df_reminders)
    st.markdown("---")
    if st.session_state.df_habits.empty:
        st.info("No habits yet.")
    else:
        for _, h in st.session_state.df_habits.iterrows():
            if not h.get("active", True):
                continue
            hid = int(h["habit_id"])
            total = int(h.get("total_completions",0) if not pd.isna(h.get("total_completions",0)) else 0)
            st.markdown(f"**#{hid}** {h['habit_name']} — {h['recurrence']} — 🔁 {total} completions")
    st.markdown("</div>", unsafe_allow_html=True)

def team_ui():
    st.markdown("<div class='card'><h3>Team</h3>", unsafe_allow_html=True)
    with st.form("add_member", clear_on_submit=True):
        un = st.text_input("Username")
        role_choice = st.selectbox("Role", ["member","manager"])
        submit = st.form_submit_button("Add Member")
        if submit and un:
            if un in st.session_state.df_users["username"].values:
                st.warning("User already exists")
            else:
                new = pd.DataFrame([{"username":un, "role":role_choice, "added_at": pd.Timestamp.now(), "added_by": st.session_state.current_user}])
                st.session_state.df_users = pd.concat([st.session_state.df_users, new], ignore_index=True)
                st.session_state.df_users.to_csv(USERS_FILE, index=False)
                st.success(f"Added {un}")
    st.markdown("---")
    if st.session_state.df_users.empty:
        st.info("No team members yet.")
    else:
        for _, u in st.session_state.df_users.iterrows():
            un = u["username"]
            role = u.get("role","member")
            count = len(st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"]==un]) if "assigned_to" in st.session_state.df_tasks.columns else 0
            st.markdown(f"- **{un}** ({role}) — {count} tasks")
    st.markdown("</div>", unsafe_allow_html=True)

def analytics_ui():
    st.markdown("<div class='card'><h3>Analytics</h3>", unsafe_allow_html=True)
    my = st.session_state.df_tasks.copy()
    if my.empty:
        st.info("No tasks to analyze.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    # Category summary
    summary = my.groupby("category").agg({"status":["count", lambda x: (x=="Completed").sum()]}).reset_index()
    summary.columns = ["Category","Total","Completed"]
    summary["Pending"] = summary["Total"] - summary["Completed"]
    summary["Rate %"] = (summary["Completed"] / summary["Total"] * 100).round(1)
    st.table(summary)
    # Simple chart: completed by day
    comp = my[my["status"]=="Completed"]
    if not comp.empty:
        series = comp.groupby(comp["completed_at"].dt.date).size()
        fig, ax = plt.subplots()
        series.plot(kind="bar", ax=ax)
        ax.set_title("Completed tasks by day")
        st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

def ai_train_ui():
    st.markdown("<div class='card'><h3>AI / Train</h3>", unsafe_allow_html=True)
    if st.button("Train simple logistic regression model"):
        model = train_simple_model(st.session_state.df_tasks)
        if model is None:
            st.warning("Not enough data to train a model (need variety of completed/pending).")
        else:
            with open("simple_model.pkl","wb") as f:
                pickle.dump(model, f)
            st.success("Trained simple model and saved to simple_model.pkl")
    st.markdown("</div>", unsafe_allow_html=True)

def exports_ui():
    st.markdown("<div class='card'><h3>Export</h3>", unsafe_allow_html=True)
    # CSV download for current user's tasks
    my = st.session_state.df_tasks.copy()
    my_user = my[my["assigned_to"]==st.session_state.current_user] if st.session_state.current_user else my
    csv_buf = my_user.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of my tasks", data=csv_buf, file_name=f"assan_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    # Excel export
    try:
        import openpyxl
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            my_user.to_excel(writer, sheet_name="All Tasks", index=False)
            # Summary sheet
            if not my_user.empty:
                summary = my_user.groupby("category").agg({"status":["count", lambda x: (x=="Completed").sum()]})
                summary.columns = ["Total","Completed"]
                summary = summary.reset_index()
                summary.to_excel(writer, sheet_name="Summary", index=False)
        st.download_button("Download Excel", data=output.getvalue(), file_name=f"assan_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.info("Install openpyxl to enable Excel export (add to requirements).")
    # PDF report (reportlab)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors as rl_colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        title = Paragraph("<b>Assan Productivity Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1,12))
        user_para = Paragraph(f"<b>User:</b> {st.session_state.current_user}", styles['Normal'])
        story.append(user_para)
        story.append(Spacer(1,12))
        # Table of tasks
        data = [["ID","Task","Status","Priority"]]
        for _, t in my_user.head(50).iterrows():
            data.append([int(t["id"]), str(t["task"])[:40], str(t["status"]), "⭐" if t["priority"]=="Y" else ""])
        table = Table(data, colWidths=[40,300,80,50])
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0), rl_colors.HexColor('#2C3E50')), ('TEXTCOLOR',(0,0),(-1,0), rl_colors.whitesmoke),
                                   ('ALIGN',(0,0),(-1,-1),'LEFT'), ('GRID',(0,0),(-1,-1),1, rl_colors.grey)]))
        story.append(table)
        doc.build(story)
        st.download_button("Download PDF report", data=buffer.getvalue(), file_name=f"assan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    except Exception:
        st.info("Install reportlab to enable PDF export (add to requirements).")
    st.markdown("</div>", unsafe_allow_html=True)

def reminders_ui():
    st.markdown("<div class='card'><h3>Reminders</h3>", unsafe_allow_html=True)
    q = st.session_state.reminder_queue
    pending_items = []
    while not q.empty():
        pending_items.append(q.get())
    if not pending_items:
        st.success("No active reminders.")
    else:
        for rem in pending_items:
            atype = rem.get("alert_type")
            tid = rem.get("task_id")
            name = rem.get("task_name")
            if atype == "overdue":
                st.error(f"OVERDUE: #{tid} {name}")
            elif atype == "urgent":
                st.warning(f"URGENT: #{tid} {name}")
            else:
                st.info(f"Warning: #{tid} {name}")
            # put back in queue for future viewing
            st.session_state.reminder_queue.put(rem)
    st.markdown("</div>", unsafe_allow_html=True)

def settings_ui():
    st.markdown("<div class='card'><h3>Settings</h3>", unsafe_allow_html=True)
    if st.button("Save all data now"):
        save_all(st.session_state.df_tasks, st.session_state.df_habits, st.session_state.df_users, st.session_state.df_comments, st.session_state.df_reminders)
        st.success("Saved.")
    if st.button("Restart reminder system"):
        stop_reminder_system()
        start_reminder_system()
        st.success("Reminder system restarted.")
    if st.button("Stop reminder system"):
        stop_reminder_system()
        st.success("Stopped reminder system.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Main router ----------
header()
content = menu

if content == "Dashboard":
    st.markdown("<div class='card'><h3>Dashboard</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted-small'>Welcome, <b>{st.session_state.current_user or '—'}</b></div>", unsafe_allow_html=True)
    total = len(st.session_state.df_tasks)
    pending = len(st.session_state.df_tasks[st.session_state.df_tasks["status"]=="Pending"]) if not st.session_state.df_tasks.empty else 0
    completed = len(st.session_state.df_tasks[st.session_state.df_tasks["status"]=="Completed"]) if not st.session_state.df_tasks.empty else 0
    st.markdown(f"<div style='margin-top:10px'><b>Total tasks:</b> {total} &nbsp;&nbsp; <b>Pending:</b> {pending} &nbsp;&nbsp; <b>Completed:</b> {completed}</div>", unsafe_allow_html=True)
    # AI plan
    st.markdown("---")
    st.markdown("<h4>Daily AI Plan</h4>", unsafe_allow_html=True)
    pending_tasks = st.session_state.df_tasks[st.session_state.df_tasks["status"]=="Pending"] if not st.session_state.df_tasks.empty else pd.DataFrame()
    if pending_tasks.empty:
        st.info("No pending tasks")
    else:
        top = pending_tasks.sort_values("ai_prediction", ascending=False).head(5)
        for _, t in top.iterrows():
            tleft, _ = time_left_parts(t["deadline"])
            st.markdown(f"- #{int(t['id'])} {t['task']} | AI:{t['ai_prediction']:.0f}% | {tleft}")
    st.markdown("</div>", unsafe_allow_html=True)

elif content == "My Tasks":
    view_tasks_ui()

elif content == "Add Task":
    add_task_ui()

elif content == "Habits":
    habits_ui()

elif content == "Team":
    team_ui()

elif content == "Analytics":
    analytics_ui()

elif content == "AI / Train":
    ai_train_ui()

elif content == "Exports":
    exports_ui()

elif content == "Reminders":
    reminders_ui()

elif content == "Settings":
    settings_ui()

# Footer / autosave on interactions
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
if st.button("Save & Exit"):
    save_all(st.session_state.df_tasks, st.session_state.df_habits, st.session_state.df_users, st.session_state.df_comments, st.session_state.df_reminders)
    stop_reminder_system()
    st.success(f"Saved. Goodbye {st.session_state.current_user}!")

# ensure data persisted occasionally
if st.button("Save now (manual)"):
    save_all(st.session_state.df_tasks, st.session_state.df_habits, st.session_state.df_users, st.session_state.df_comments, st.session_state.df_reminders)
    st.success("Saved.")

# End of file
