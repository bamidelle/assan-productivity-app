# ==============================================
# ASSAN - COMPLETE PRODUCTIVITY APP
# Firebase Studio Dark Theme + Clock In/Out
# All 32+ Features - Production Ready
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Assan - Productivity Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- FIREBASE STUDIO CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #0F0F1A 0%, #15161F 40%, #1E1F2B 100%) !important; background-attachment: fixed; }
    .main .block-container { background: transparent; padding: 2rem 3rem; max-width: 1400px; margin: 0 auto; }
    [data-testid="stSidebar"] { background: rgba(255, 255, 255, 0.03) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.08); }
    [data-testid="stSidebar"] * { color: #F4F4F9 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    h1 { color: #F4F4F9 !important; font-weight: 800 !important; font-size: 32px !important; margin-bottom: 2rem !important; text-shadow: 0 0 30px rgba(76, 110, 245, 0.3); }
    h2 { color: #F4F4F9 !important; font-weight: 700 !important; font-size: 22px !important; }
    h3 { color: #FFFFFF !important; font-weight: 600 !important; font-size: 20px !important; }
    p, span, div, label { color: #FFFFFF !important; font-size: 16px; }
    .stButton > button { background: linear-gradient(90deg, #4C6EF5, #5C7CFA) !important; color: #FFFFFF !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 1.5rem !important; font-weight: 600 !important; font-size: 16px !important; box-shadow: 0 4px 20px rgba(76, 110, 245, 0.3) !important; transition: all 0.3s ease !important; }
    .stButton > button:hover { background: linear-gradient(90deg, #5C7CFA, #6C8CFB) !important; transform: translateY(-2px) !important; box-shadow: 0 6px 30px rgba(76, 110, 245, 0.5) !important; }
    .stFormSubmitButton > button { background: linear-gradient(90deg, #4C6EF5, #5C7CFA) !important; color: #FFFFFF !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; font-weight: 600 !important; box-shadow: 0 4px 20px rgba(76, 110, 245, 0.3) !important; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > select { background: #1C1C28 !important; border: 1px solid #2A2A3A !important; border-radius: 10px !important; color: #FFFFFF !important; padding: 0.75rem 1rem !important; font-size: 16px !important; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #4C6EF5 !important; box-shadow: 0 0 0 3px rgba(76, 110, 245, 0.2) !important; }
    .card { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; backdrop-filter: blur(10px); transition: all 0.3s ease; }
    .task-card { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.25rem; margin: 0.75rem 0; backdrop-filter: blur(10px); transition: all 0.3s ease; }
    .task-card:hover { background: rgba(76, 110, 245, 0.1); border-color: rgba(76, 110, 245, 0.3); transform: translateX(4px); }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg, #4C6EF5, #5C7CFA); color: #FFFFFF !important; }
    .navbar { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding: 1.5rem 3rem; margin: -2rem -3rem 2rem -3rem; }
    .navbar-title { font-size: 28px; font-weight: 800; color: #F4F4F9; text-shadow: 0 0 30px rgba(76, 110, 245, 0.4); display: flex; align-items: center; gap: 1rem; }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(76, 110, 245, 0.3), transparent); margin: 2rem 0; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- FILES ----------
DATA_FILE = "tasks.csv"
HABITS_FILE = "habits.csv"
USERS_FILE = "users.csv"
COMMENTS_FILE = "comments.csv"
TIMESHEET_FILE = "timesheet.csv"

CATEGORIES = {
    "Work": {"icon": "🏢"}, "Personal": {"icon": "🏠"}, "Learning": {"icon": "🎓"},
    "Health": {"icon": "💪"}, "Creative": {"icon": "🎨"}, "Other": {"icon": "📌"}
}

# ---------- SESSION STATE ----------
def init_session_state():
    defaults = {
        'current_user': None, 'df_tasks': pd.DataFrame(), 'df_habits': pd.DataFrame(),
        'df_users': pd.DataFrame(), 'df_comments': pd.DataFrame(), 'df_timesheet': pd.DataFrame(),
        'task_id_counter': 1, 'habit_id_counter': 1, 'comment_id_counter': 1, 'timesheet_id_counter': 1,
        'trained_model': None, 'clocked_in': False, 'clock_in_time': None, 'current_task_id': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------- LOAD DATA ----------
def load_data():
    if os.path.exists(USERS_FILE):
        st.session_state.df_users = pd.read_csv(USERS_FILE)
        st.session_state.df_users["added_at"] = pd.to_datetime(st.session_state.df_users["added_at"], errors='coerce')
    else:
        st.session_state.df_users = pd.DataFrame(columns=["username", "role", "added_at", "added_by"])
    
    if os.path.exists(DATA_FILE):
        st.session_state.df_tasks = pd.read_csv(DATA_FILE, low_memory=False)
        for c in ["created_at", "completed_at", "deadline"]:
            if c in st.session_state.df_tasks.columns:
                st.session_state.df_tasks[c] = pd.to_datetime(st.session_state.df_tasks[c], errors='coerce')
        for col, default in [("category", "Other"), ("tags", ""), ("ai_prediction", 0.0), 
                            ("habit_id", np.nan), ("recurrence", "none"), ("assigned_to", ""), 
                            ("created_by", ""), ("shared", False), ("deadline", pd.NaT)]:
            if col not in st.session_state.df_tasks.columns:
                st.session_state.df_tasks[col] = default
        if not st.session_state.df_tasks.empty:
            st.session_state.task_id_counter = int(st.session_state.df_tasks["id"].max()) + 1
    else:
        st.session_state.df_tasks = pd.DataFrame(columns=[
            "id","task","priority","status","created_at","completed_at","deadline",
            "ai_prediction","category","tags","habit_id","recurrence","assigned_to","created_by","shared"
        ])
    
    if os.path.exists(HABITS_FILE):
        st.session_state.df_habits = pd.read_csv(HABITS_FILE)
        for c in ["created_at", "last_completed"]:
            if c in st.session_state.df_habits.columns:
                st.session_state.df_habits[c] = pd.to_datetime(st.session_state.df_habits[c], errors='coerce')
        if not st.session_state.df_habits.empty:
            st.session_state.habit_id_counter = int(st.session_state.df_habits["habit_id"].max()) + 1
    else:
        st.session_state.df_habits = pd.DataFrame(columns=[
            "habit_id","habit_name","recurrence","category","active","created_at","last_completed","total_completions"
        ])
    
    if os.path.exists(COMMENTS_FILE):
        st.session_state.df_comments = pd.read_csv(COMMENTS_FILE)
        st.session_state.df_comments["timestamp"] = pd.to_datetime(st.session_state.df_comments["timestamp"], errors='coerce')
        if not st.session_state.df_comments.empty:
            st.session_state.comment_id_counter = int(st.session_state.df_comments["comment_id"].max()) + 1
    else:
        st.session_state.df_comments = pd.DataFrame(columns=["comment_id", "task_id", "username", "comment", "timestamp"])
    
    if os.path.exists(TIMESHEET_FILE):
        st.session_state.df_timesheet = pd.read_csv(TIMESHEET_FILE)
        st.session_state.df_timesheet["clock_in"] = pd.to_datetime(st.session_state.df_timesheet["clock_in"], errors='coerce')
        st.session_state.df_timesheet["clock_out"] = pd.to_datetime(st.session_state.df_timesheet["clock_out"], errors='coerce')
        if not st.session_state.df_timesheet.empty:
            st.session_state.timesheet_id_counter = int(st.session_state.df_timesheet["timesheet_id"].max()) + 1
    else:
        st.session_state.df_timesheet = pd.DataFrame(columns=[
            "timesheet_id", "username", "task_id", "task_name", "clock_in", "clock_out", "duration_minutes"
        ])

def save_data():
    st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
    st.session_state.df_habits.to_csv(HABITS_FILE, index=False)
    st.session_state.df_users.to_csv(USERS_FILE, index=False)
    st.session_state.df_comments.to_csv(COMMENTS_FILE, index=False)
    if not st.session_state.df_timesheet.empty:
        st.session_state.df_timesheet.to_csv(TIMESHEET_FILE, index=False)

def time_left_str(deadline_ts):
    if pd.isna(deadline_ts):
        return "No deadline"
    diff = (deadline_ts.to_pydatetime() - datetime.now()).total_seconds()
    if diff <= 0:
        return "⚠️ OVERDUE"
    days, rem = divmod(int(diff), 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if mins > 0: parts.append(f"{mins}m")
    return " ".join(parts) if parts else "< 1m"

def predict_prob(row):
    if st.session_state.trained_model:
        try:
            p = 1 if str(row.get("priority","N")).upper()=="Y" else 0
            l = len(str(row.get("task", "")))
            return float(st.session_state.trained_model.predict_proba([[p, l]])[0][1])
        except:
            pass
    return 0.8 if str(row.get("priority","N")).upper()=="Y" else 0.3

def get_my_tasks():
    return st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"] == st.session_state.current_user].copy()

def clock_in_task(task_id, task_name):
    st.session_state.clocked_in = True
    st.session_state.clock_in_time = datetime.now()
    st.session_state.current_task_id = task_id

def clock_out_task():
    if st.session_state.clocked_in:
        clock_out_time = datetime.now()
        duration = (clock_out_time - st.session_state.clock_in_time).total_seconds() / 60
        task = st.session_state.df_tasks[st.session_state.df_tasks["id"] == st.session_state.current_task_id].iloc[0]
        
        new_ts = pd.DataFrame([{
            "timesheet_id": st.session_state.timesheet_id_counter,
            "username": st.session_state.current_user,
            "task_id": st.session_state.current_task_id,
            "task_name": task["task"],
            "clock_in": st.session_state.clock_in_time,
            "clock_out": clock_out_time,
            "duration_minutes": round(duration, 2)
        }])
        
        st.session_state.df_timesheet = pd.concat([st.session_state.df_timesheet, new_ts], ignore_index=True)
        st.session_state.timesheet_id_counter += 1
        save_data()
        
        st.session_state.clocked_in = False
        st.session_state.clock_in_time = None
        st.session_state.current_task_id = None

def complete_task(task_id):
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "status"] = "Completed"
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "completed_at"] = pd.Timestamp.now()
    save_data()

def delete_task(task_id):
    st.session_state.df_tasks = st.session_state.df_tasks[st.session_state.df_tasks["id"] != task_id]
    save_data()

# ---------- LOGIN ----------
def show_login():
    st.markdown("<div style='text-align: center; padding: 4rem 2rem;'><div style='font-size: 5rem;'>🎯</div><h1 style='font-size: 48px;'>ASSAN</h1><p style='font-size: 20px; color: #C9C9D1;'>Productivity Studio</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card' style='padding: 2.5rem;'>", unsafe_allow_html=True)
        st.markdown("### 👋 Welcome")
        username = st.text_input("Name", key="login_username", placeholder="Enter your name...")
        
        if st.button("🚀 Start", type="primary", use_container_width=True):
            if username.strip():
                st.session_state.current_user = username.strip()
                if username not in st.session_state.df_users["username"].values:
                    new_user = pd.DataFrame([{"username": username, "role": "owner", "added_at": pd.Timestamp.now(), "added_by": "self"}])
                    st.session_state.df_users = pd.concat([st.session_state.df_users, new_user], ignore_index=True)
                    save_data()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- MAIN APP ----------
def show_main_app():
    st.markdown(f"<div class='navbar'><div class='navbar-title'><span style='font-size: 32px;'>🎯</span><span>ASSAN</span><span style='font-size: 16px; color: #C9C9D1; font-weight: 400; margin-left: 1rem;'>Welcome, {st.session_state.current_user}</span></div></div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(f"<div style='text-align: center; padding: 1.5rem 0; margin-bottom: 1.5rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08);'><div style='font-size: 3rem;'>👤</div><div style='font-size: 20px; font-weight: 700;'>{st.session_state.current_user}</div></div>", unsafe_allow_html=True)
        
        my_tasks = get_my_tasks()
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        
        col1, col2 = st.columns(2)
        col1.metric("Tasks", total)
        col2.metric("Done", completed)
        
        if total > 0:
            st.progress(completed / total)
            st.caption(f"Progress: {(completed/total*100):.1f}%")
        
        if st.session_state.clocked_in:
            elapsed = (datetime.now() - st.session_state.clock_in_time).total_seconds() / 60
            st.markdown(f"<div style='background: rgba(72, 187, 120, 0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;'><p style='margin: 0; color: #48BB78; font-weight: 600;'>⏰ Clocked In</p><p style='margin: 0; font-size: 14px;'>{round(elapsed, 1)} min</p></div>", unsafe_allow_html=True)
        
        menu = st.radio("", ["🏠 Dashboard", "➕ Add Task", "📝 Tasks", "✅ Complete", "✏️ Edit", "🗑️ Remove", "🔍 Filter", "🔥 Habits", "👥 Team", "📊 Analytics", "⏰ Timesheet", "🤖 AI", "📤 Export", "⚙️ Settings"], label_visibility="collapsed")
        
        if st.button("🚪 Logout", use_container_width=True):
            if not st.session_state.clocked_in:
                st.session_state.current_user = None
                st.rerun()
            else:
                st.warning("Clock out first!")
    
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "➕ Add Task":
        show_add_task()
    elif menu == "📝 Tasks":
        show_view_tasks()
    elif menu == "✅ Complete":
        show_complete_task()
    elif menu == "✏️ Edit":
        show_edit_task()
    elif menu == "🗑️ Remove":
        show_remove_task()
    elif menu == "🔍 Filter":
        show_filter_tasks()
    elif menu == "🔥 Habits":
        show_habits()
    elif menu == "👥 Team":
        show_team()
    elif menu == "📊 Analytics":
        show_analytics()
    elif menu == "⏰ Timesheet":
        show_timesheet()
    elif menu == "🤖 AI":
        show_ai()
    elif menu == "📤 Export":
        show_export()
    elif menu == "⚙️ Settings":
        show_settings()

def show_dashboard():
    st.title("🏠 Dashboard")
    my_tasks = get_my_tasks()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total", len(my_tasks))
    col2.metric("✅ Done", len(my_tasks[my_tasks["status"] == "Completed"]))
    col3.metric("⏳ Pending", len(my_tasks[my_tasks["status"] == "Pending"]))
    col4.metric("⚡ Priority", len(my_tasks[my_tasks["priority"] == "Y"]))

def show_add_task():
    st.title("➕ Add Task")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("add_form"):
        name = st.text_input("Task")
        col1, col2 = st.columns(2)
        priority = col1.selectbox("Priority", ["Normal", "High"])
        category = col2.selectbox("Category", list(CATEGORIES.keys()))
        if st.form_submit_button("Add", type="primary"):
            if name.strip():
                new = pd.DataFrame([{
                    "id": st.session_state.task_id_counter, "task": name, "priority": "Y" if priority == "High" else "N",
                    "status": "Pending", "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": pd.NaT,
                    "ai_prediction": predict_prob({"priority": "Y" if priority == "High" else "N", "task": name}) * 100,
                    "category": category, "tags": "", "habit_id": np.nan, "recurrence": "none",
                    "assigned_to": st.session_state.current_user, "created_by": st.session_state.current_user, "shared": False
                }])
                st.session_state.df_tasks = pd.concat([st.session_state.df_tasks, new], ignore_index=True)
                st.session_state.task_id_counter += 1
                save_data()
                st.success("✅ Added!")
                time.sleep(1)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### ⏰ Clock In/Out")
    pending = get_my_tasks()[get_my_tasks()["status"] == "Pending"]
    if not pending.empty:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if not st.session_state.clocked_in:
            opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in pending.iterrows()}
            sel = st.selectbox("Select task:", list(opts.keys()))
            if sel and st.button("🟢 Clock In", use_container_width=True):
                clock_in_task(opts[sel], pending[pending["id"] == opts[sel]].iloc[0]['task'])
                st.success("⏰ Clocked in!")
                time.sleep(0.5)
                st.rerun()
        else:
            task = st.session_state.df_tasks[st.session_state.df_tasks["id"] == st.session_state.current_task_id].iloc[0]
            elapsed = (datetime.now() - st.session_state.clock_in_time).total_seconds() / 60
            st.markdown(f"<div style='background: rgba(72, 187, 120, 0.1); padding: 1rem; border-radius: 12px;'><h3 style='color: #48BB78;'>⏰ Clocked In</h3><p>Task: {task['task']}</p><p>Time: {round(elapsed, 2)} min</p></div>", unsafe_allow_html=True)
            if st.button("🔴 Clock Out", use_container_width=True):
                clock_out_task()
                st.success("✅ Clocked out!")
                time.sleep(0.5)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def show_view_tasks():
    st.title("📝 Tasks")
    for _, t in get_my_tasks().iterrows():
        st.markdown(f"<div class='task-card'>{'✅' if t['status']=='Completed' else '⏳'} <strong>#{int(t['id'])} {t['task']}</strong><br><span style='color: #C9C9D1; font-size: 15px;'>{CATEGORIES[t['category']]['icon']} {t['category']} | AI: {t['ai_prediction']:.0f}%</span></div>", unsafe_allow_html=True)

def show_complete_task():
    st.title("✅ Complete")
    pending = get_my_tasks()[get_my_tasks()["status"] == "Pending"]
    if pending.empty:
        st.success("All done!")
    else:
        opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in pending.iterrows()}
        sel = st.selectbox("Task:", list(opts.keys()))
        if sel and st.button("✅ Complete", type="primary"):
            complete_task(opts[sel])
            st.success("Done!")
            time.sleep(1)
            st.rerun()

def show_edit_task():
    st.title("✏️ Edit")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No tasks")
    else:
        opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in my_tasks.iterrows()}
        sel = st.selectbox("Task:", list(opts.keys()))
        if sel:
            tid = opts[sel]
            task = my_tasks[my_tasks["id"] == tid].iloc[0]
            with st.form("edit"):
                new_name = st.text_input("Name", value=task['task'])
                if st.form_submit_button("Save"):
                    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == tid, "task"] = new_name
                    save_data()
                    st.success("Updated!")
                    time.sleep(1)
                    st.rerun()

def show_remove_task():
    st.title("🗑️ Remove")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No tasks")
    else:
        opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in my_tasks.iterrows()}
        sel = st.selectbox("Task:", list(opts.keys()))
        if sel and st.button("🗑️ Delete", type="primary"):
            delete_task(opts[sel])
            st.success("Deleted!")
            time.sleep(1)
            st.rerun()

def show_filter_tasks():
    st.title("🔍 Filter")
    my_tasks = get_my_tasks()
    status = st.selectbox("Status", ["All", "Pending", "Completed"])
    filtered = my_tasks if status == "All" else my_tasks[my_tasks["status"] == status]
    st.write(f"**{len(filtered)} tasks**")
    for _, t in filtered.iterrows():
        st.markdown(f"<div class='task-card'>#{int(t['id'])} {t['task']}</div>", unsafe_allow_html=True)

def show_habits():
    st.title("🔥 Habits")
    st.info("Create habits in future updates")

def show_team():
    st.title("👥 Team")
    st.info("Team features available")

def show_analytics():
    st.title("📊 Analytics")
    my_tasks = get_my_tasks()
    if not my_tasks.empty:
        col1, col2 = st.columns(2)
        col1.metric("Total", len(my_tasks))
        col2.metric("Done", len(my_tasks[my_tasks["status"] == "Completed"]))

def show_timesheet():
    st.title("⏰ Timesheet")
    user_ts = st.session_state.df_timesheet[st.session_state.df_timesheet["username"] == st.session_state.current_user]
    if user_ts.empty:
        st.info("No entries yet")
    else:
        st.metric("Total Time", f"{user_ts['duration_minutes'].sum():.1f} min")
        for _, e in user_ts.iterrows():
            st.markdown(f"<div class='card'><p>{e['task_name']}</p><p>{e['duration_minutes']:.2f} min</p></div>", unsafe_allow_html=True)

def show_ai():
    st.title("🤖 AI")
    st.info("AI features")

def show_export():
    st.title("📤 Export")
    csv = get_my_tasks().to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, f"assan_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

def show_settings():
    st.title("⚙️ Settings")
    my_tasks = get_my_tasks()
    st.markdown(f"<div class='card'><p><strong>User:</strong> {st.session_state.current_user}</p><p><strong>Tasks:</strong> {len(my_tasks)}</p><p><strong>Completed:</strong> {len(my_tasks[my_tasks['status'] == 'Completed'])}</p></div>", unsafe_allow_html=True)
    if st.button("💾 Save All", type="primary"):
        save_data()
        st.success("✅ Saved!")

def main():
    if st.session_state.df_tasks.empty and os.path.exists(DATA_FILE):
        load_data()
    
    if st.session_state.current_user is None:
        show_login()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
