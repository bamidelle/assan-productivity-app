# ==============================================
# ASSAN - COMPLETE PRODUCTIVITY APP
# All 32 Features + Clock In/Out
# White Background, Black Text
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
import glob
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Assan", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# ---------- WHITE THEME CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #FFFFFF !important; }
    .main .block-container { background: #FFFFFF; padding: 2rem 3rem; max-width: 1400px; margin: 0 auto; }
    [data-testid="stSidebar"] { background: #F8F9FA !important; border-right: 1px solid #E0E0E0; }
    h1, h2, h3, p, span, div, label { color: #000000 !important; }
    .stButton > button { background: linear-gradient(90deg, #4C6EF5, #5C7CFA) !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; padding: 0.6rem 1.2rem !important; font-weight: 600 !important; }
    .stButton > button:hover { opacity: 0.9; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > select { background: #FFFFFF !important; border: 1px solid #D0D0D0 !important; border-radius: 6px !important; color: #000000 !important; padding: 0.6rem !important; }
    .card { background: #F8F9FA; border: 1px solid #E0E0E0; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
    .task-card { background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .task-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = "tasks.csv"
HABITS_FILE = "habits.csv"
USERS_FILE = "users.csv"
COMMENTS_FILE = "comments.csv"
TIMESHEET_FILE = "timesheet.csv"

CATEGORIES = {"Work": "🏢", "Personal": "🏠", "Learning": "🎓", "Health": "💪", "Creative": "🎨", "Other": "📌"}

def init_session_state():
    defaults = {'current_user': None, 'df_tasks': pd.DataFrame(), 'df_habits': pd.DataFrame(), 'df_users': pd.DataFrame(), 
                'df_comments': pd.DataFrame(), 'df_timesheet': pd.DataFrame(), 'task_id_counter': 1, 'habit_id_counter': 1, 
                'comment_id_counter': 1, 'timesheet_id_counter': 1, 'trained_model': None, 'clocked_in': False, 
                'clock_in_time': None, 'current_task_id': None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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
        for col, default in [("category", "Other"), ("tags", ""), ("ai_prediction", 0.0), ("habit_id", np.nan), 
                            ("recurrence", "none"), ("assigned_to", ""), ("created_by", ""), ("shared", False), ("deadline", pd.NaT)]:
            if col not in st.session_state.df_tasks.columns:
                st.session_state.df_tasks[col] = default
        if not st.session_state.df_tasks.empty:
            st.session_state.task_id_counter = int(st.session_state.df_tasks["id"].max()) + 1
    else:
        st.session_state.df_tasks = pd.DataFrame(columns=["id","task","priority","status","created_at","completed_at","deadline","ai_prediction","category","tags","habit_id","recurrence","assigned_to","created_by","shared"])
    
    if os.path.exists(HABITS_FILE):
        st.session_state.df_habits = pd.read_csv(HABITS_FILE)
        for c in ["created_at", "last_completed"]:
            if c in st.session_state.df_habits.columns:
                st.session_state.df_habits[c] = pd.to_datetime(st.session_state.df_habits[c], errors='coerce')
        if not st.session_state.df_habits.empty:
            st.session_state.habit_id_counter = int(st.session_state.df_habits["habit_id"].max()) + 1
    else:
        st.session_state.df_habits = pd.DataFrame(columns=["habit_id","habit_name","recurrence","category","active","created_at","last_completed","total_completions"])
    
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
        st.session_state.df_timesheet = pd.DataFrame(columns=["timesheet_id", "username", "task_id", "task_name", "clock_in", "clock_out", "duration_minutes"])

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
        return "OVERDUE"
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

def calculate_streak(habit_id):
    if st.session_state.df_tasks.empty:
        return 0
    habit_tasks = st.session_state.df_tasks[(st.session_state.df_tasks["habit_id"] == habit_id) & (st.session_state.df_tasks["status"] == "Completed")].sort_values("completed_at", ascending=False)
    if habit_tasks.empty:
        return 0
    habit_info = st.session_state.df_habits[st.session_state.df_habits["habit_id"] == habit_id].iloc[0]
    recurrence = habit_info["recurrence"]
    streak = 0
    expected_date = datetime.now().date()
    for _, task in habit_tasks.iterrows():
        completed_date = task["completed_at"].date()
        if recurrence == "daily":
            if completed_date >= expected_date - timedelta(days=1):
                streak += 1
                expected_date = completed_date - timedelta(days=1)
            else:
                break
        elif recurrence == "weekly":
            if completed_date >= expected_date - timedelta(weeks=1):
                streak += 1
                expected_date = completed_date - timedelta(weeks=1)
            else:
                break
    return streak

def get_my_tasks():
    return st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"] == st.session_state.current_user].copy()

def complete_task(task_id):
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "status"] = "Completed"
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "completed_at"] = pd.Timestamp.now()
    task = st.session_state.df_tasks[st.session_state.df_tasks["id"] == task_id].iloc[0]
    if not pd.isna(task["habit_id"]):
        habit_id = int(task["habit_id"])
        st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"] == habit_id, "total_completions"] += 1
        st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"] == habit_id, "last_completed"] = pd.Timestamp.now()
        create_task_from_habit(habit_id, task["task"], task["recurrence"], task["category"])
    save_data()

def delete_task(task_id):
    st.session_state.df_tasks = st.session_state.df_tasks[st.session_state.df_tasks["id"] != task_id]
    save_data()

def create_task_from_habit(habit_id, name, recurrence, category):
    if recurrence == "daily":
        deadline = datetime.now().replace(hour=23, minute=59)
    elif recurrence == "weekly":
        days = (6 - datetime.now().weekday()) % 7
        deadline = (datetime.now() + timedelta(days=days)).replace(hour=23, minute=59)
    else:
        next_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1)
        deadline = next_month - timedelta(seconds=1)
    prob = predict_prob({"priority": "Y", "task": name})
    new_task = pd.DataFrame([{"id": st.session_state.task_id_counter, "task": name, "priority": "Y", "status": "Pending", 
                              "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": deadline, 
                              "ai_prediction": round(prob * 100, 2), "category": category, "tags": f"habit,{recurrence}", 
                              "habit_id": habit_id, "recurrence": recurrence, "assigned_to": st.session_state.current_user, 
                              "created_by": st.session_state.current_user, "shared": False}])
    st.session_state.df_tasks = pd.concat([st.session_state.df_tasks, new_task], ignore_index=True)
    st.session_state.task_id_counter += 1

def clock_in_task(task_id):
    st.session_state.clocked_in = True
    st.session_state.clock_in_time = datetime.now()
    st.session_state.current_task_id = task_id

def clock_out_task():
    if st.session_state.clocked_in:
        clock_out_time = datetime.now()
        duration = (clock_out_time - st.session_state.clock_in_time).total_seconds() / 60
        task = st.session_state.df_tasks[st.session_state.df_tasks["id"] == st.session_state.current_task_id].iloc[0]
        new_ts = pd.DataFrame([{"timesheet_id": st.session_state.timesheet_id_counter, "username": st.session_state.current_user, 
                               "task_id": st.session_state.current_task_id, "task_name": task["task"], 
                               "clock_in": st.session_state.clock_in_time, "clock_out": clock_out_time, 
                               "duration_minutes": round(duration, 2)}])
        st.session_state.df_timesheet = pd.concat([st.session_state.df_timesheet, new_ts], ignore_index=True)
        st.session_state.timesheet_id_counter += 1
        save_data()
        st.session_state.clocked_in = False
        st.session_state.clock_in_time = None
        st.session_state.current_task_id = None

def show_login():
    st.markdown("<div style='text-align: center; padding: 4rem 2rem;'><div style='font-size: 5rem;'>🎯</div><h1>ASSAN</h1><p>Productivity Studio - 32 Features</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Enter your name:", placeholder="Your name...")
        if st.button("🚀 Start", type="primary", use_container_width=True):
            if username.strip():
                st.session_state.current_user = username.strip()
                if username not in st.session_state.df_users["username"].values:
                    new_user = pd.DataFrame([{"username": username, "role": "owner", "added_at": pd.Timestamp.now(), "added_by": "self"}])
                    st.session_state.df_users = pd.concat([st.session_state.df_users, new_user], ignore_index=True)
                    save_data()
                st.rerun()

def show_main_app():
    with st.sidebar:
        st.markdown(f"## 👤 {st.session_state.current_user}")
        my_tasks = get_my_tasks()
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        col1, col2 = st.columns(2)
        col1.metric("Tasks", total)
        col2.metric("Done", completed)
        if total > 0:
            st.progress(completed / total)
        if st.session_state.clocked_in:
            elapsed = (datetime.now() - st.session_state.clock_in_time).total_seconds() / 60
            st.success(f"⏰ Clocked In: {round(elapsed, 1)} min")
        
        menu = st.radio("", ["🏠 Dashboard", "➕ Add Task", "📝 View Tasks", "🗑️ Remove", "✅ Complete", "✏️ Edit", "🔍 Filter", 
                            "📊 Chart", "📅 Weekly", "🏆 Top Days", "📈 Trend", "🎯 Smart Trend", "🤖 AI Plan", "🔔 Reminders", 
                            "📂 By Category", "🔎 Search Tags", "📋 Cat Summary", "🔥 Create Habit", "🔥 View Habits", "🏅 Habit Dash", 
                            "👥 Add Member", "👥 View Team", "📤 Assign", "📥 My Assigned", "💬 Comment", "📊 Team Dash", "🧠 Train AI", 
                            "📥 Export CSV", "📊 Export Excel", "📄 Gen PDF", "📧 Email", "📁 List Files", "⏰ Timesheet", "⚙️ Settings"], 
                       label_visibility="collapsed")
        
        if st.button("🚪 Logout"):
            if not st.session_state.clocked_in:
                st.session_state.current_user = None
                st.rerun()
            else:
                st.warning("Clock out first!")
    
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "➕ Add Task":
        show_add_task()
    elif menu == "📝 View Tasks":
        show_view_tasks()
    elif menu == "🗑️ Remove":
        show_remove()
    elif menu == "✅ Complete":
        show_complete()
    elif menu == "✏️ Edit":
        show_edit()
    elif menu == "🔍 Filter":
        show_filter()
    elif menu == "📊 Chart":
        show_chart()
    elif menu == "📅 Weekly":
        show_weekly()
    elif menu == "🏆 Top Days":
        show_top_days()
    elif menu == "📈 Trend":
        show_trend()
    elif menu == "🎯 Smart Trend":
        show_smart_trend()
    elif menu == "🤖 AI Plan":
        show_ai_plan()
    elif menu == "🔔 Reminders":
        show_reminders()
    elif menu == "📂 By Category":
        show_by_category()
    elif menu == "🔎 Search Tags":
        show_search_tags()
    elif menu == "📋 Cat Summary":
        show_cat_summary()
    elif menu == "🔥 Create Habit":
        show_create_habit()
    elif menu == "🔥 View Habits":
        show_view_habits()
    elif menu == "🏅 Habit Dash":
        show_habit_dash()
    elif menu == "👥 Add Member":
        show_add_member()
    elif menu == "👥 View Team":
        show_view_team()
    elif menu == "📤 Assign":
        show_assign()
    elif menu == "📥 My Assigned":
        show_my_assigned()
    elif menu == "💬 Comment":
        show_comment()
    elif menu == "📊 Team Dash":
        show_team_dash()
    elif menu == "🧠 Train AI":
        show_train_ai()
    elif menu == "📥 Export CSV":
        show_export_csv()
    elif menu == "📊 Export Excel":
        show_export_excel()
    elif menu == "📄 Gen PDF":
        show_gen_pdf()
    elif menu == "📧 Email":
        show_email()
    elif menu == "📁 List Files":
        show_list_files()
    elif menu == "⏰ Timesheet":
        show_timesheet()
    elif menu == "⚙️ Settings":
        show_settings()

# 1. Dashboard
def show_dashboard():
    st.title("🏠 Dashboard")
    my_tasks = get_my_tasks()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(my_tasks))
    col2.metric("Done", len(my_tasks[my_tasks["status"] == "Completed"]))
    col3.metric("Pending", len(my_tasks[my_tasks["status"] == "Pending"]))
    col4.metric("Priority", len(my_tasks[my_tasks["priority"] == "Y"]))

    st.write("Recent tasks:")
    for _, t in my_tasks.head(5).iterrows():
        st.markdown(
            f"<div class='task-card'>{'✅' if t['status']=='Completed' else '⏳'} "
            f"#{int(t['id'])} {t['task']}</div>",
            unsafe_allow_html=True
        )

    # --- FIXED: Button MUST be inside the function ---
    if st.button("Go to Menu"):
        st.session_state.current_page = "menu"
        st.rerun()



# 2. Add Task with Clock In/Out
def show_add_task():
    st.title("➕ Add Task")
    with st.form("add_form"):
        name = st.text_input("Task Name")
        col1, col2 = st.columns(2)
        priority = col1.selectbox("Priority", ["Normal", "High"])
        category = col2.selectbox("Category", list(CATEGORIES.keys()))
        col1, col2 = st.columns(2)
        deadline_date = col1.date_input("Deadline", value=None)
        deadline_time = col2.time_input("Time")
        tags = st.text_input("Tags (comma separated)")
        if st.form_submit_button("Add Task"):
            if name.strip():
                deadline_dt = datetime.combine(deadline_date, deadline_time) if deadline_date else pd.NaT
                new = pd.DataFrame([{"id": st.session_state.task_id_counter, "task": name, 
                                    "priority": "Y" if priority == "High" else "N", "status": "Pending", 
                                    "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": deadline_dt, 
                                    "ai_prediction": predict_prob({"priority": "Y" if priority == "High" else "N", "task": name}) * 100, 
                                    "category": category, "tags": tags, "habit_id": np.nan, "recurrence": "none", 
                                    "assigned_to": st.session_state.current_user, "created_by": st.session_state.current_user, "shared": False}])
                st.session_state.df_tasks = pd.concat([st.session_state.df_tasks, new], ignore_index=True)
                st.session_state.task_id_counter += 1
                save_data()
                st.success("✅ Added!")
                time.sleep(1)
                st.rerun()
    
    st.markdown("---")
    st.subheader("⏰ Clock In/Out")
    pending = get_my_tasks()[get_my_tasks()["status"] == "Pending"]
    if not pending.empty:
        if not st.session_state.clocked_in:
            opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in pending.iterrows()}
            sel = st.selectbox("Select task:", list(opts.keys()))
            if st.button("🟢 Clock In", use_container_width=True):
                clock_in_task(opts[sel])
                st.success("Clocked in!")
                time.sleep(0.5)
                st.rerun()
        else:
            task = st.session_state.df_tasks[st.session_state.df_tasks["id"] == st.session_state.current_task_id].iloc[0]
            elapsed = (datetime.now() - st.session_state.clock_in_time).total_seconds() / 60
            st.info(f"⏰ Working on: {task['task']}\nTime: {round(elapsed, 2)} minutes")
            if st.button("🔴 Clock Out", use_container_width=True):
                clock_out_task()
                st.success("Clocked out!")
                time.sleep(0.5)
                st.rerun()

# 3. View Tasks
def show_view_tasks():
    st.title("📝 View Tasks")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No tasks yet")
    else:
        for _, t in my_tasks.iterrows():
            st.markdown(f"<div class='task-card'>{'✅' if t['status']=='Completed' else '⏳'} <strong>#{int(t['id'])} {t['task']}</strong><br>{CATEGORIES[t['category']]} {t['category']} | AI: {t['ai_prediction']:.0f}% | {time_left_str(t['deadline'])}</div>", unsafe_allow_html=True)

# 4. Remove
def show_remove():
    st.title("🗑️ Remove Task")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No tasks")
    else:
        opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in my_tasks.iterrows()}
        sel = st.selectbox("Task:", list(opts.keys()))
        if st.button("Delete"):
            delete_task(opts[sel])
            st.success("Deleted!")
            time.sleep(1)
            st.rerun()

# 5. Complete
def show_complete():
    st.title("✅ Complete Task")
    pending = get_my_tasks()[get_my_tasks()["status"] == "Pending"]
    if pending.empty:
        st.success("All done!")
    else:
        opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in pending.iterrows()}
        sel = st.selectbox("Task:", list(opts.keys()))
        if st.button("Mark Complete"):
            complete_task(opts[sel])
            st.success("Done!")
            time.sleep(1)
            st.rerun()

# 6. Edit
def show_edit():
    st.title("✏️ Edit Task")
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
                new_priority = st.selectbox("Priority", ["Normal", "High"], index=0 if task['priority'] == 'N' else 1)
                if st.form_submit_button("Save"):
                    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == tid, "task"] = new_name
                    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == tid, "priority"] = "Y" if new_priority == "High" else "N"
                    save_data()
                    st.success("Updated!")
                    time.sleep(1)
                    st.rerun()

# 7. Filter
def show_filter():
    st.title("🔍 Filter Tasks")
    my_tasks = get_my_tasks()
    col1, col2, col3 = st.columns(3)
    status = col1.selectbox("Status", ["All", "Pending", "Completed"])
    priority = col2.selectbox("Priority", ["All", "High", "Normal"])
    category = col3.selectbox("Category", ["All"] + list(CATEGORIES.keys()))
    
    filtered = my_tasks.copy()
    if status != "All":
        filtered = filtered[filtered["status"] == status]
    if priority != "All":
        filtered = filtered[filtered["priority"] == ("Y" if priority == "High" else "N")]
    if category != "All":
        filtered = filtered[filtered["category"] == category]
    
    st.write(f"**{len(filtered)} tasks found**")
    for _, t in filtered.iterrows():
        st.markdown(f"<div class='task-card'>#{int(t['id'])} {t['task']} - {t['status']}</div>", unsafe_allow_html=True)

# 8. Chart
def show_chart():
    st.title("📊 Productivity Chart")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No data")
    else:
        summary = my_tasks.groupby("category").agg({"status": ["count", lambda x: (x == "Completed").sum()]}).reset_index()
        summary.columns = ["Category", "Total", "Completed"]
        summary["Rate %"] = (summary["Completed"] / summary["Total"] * 100).round(1)
        st.dataframe(summary, use_container_width=True)

# 9. Weekly
def show_weekly():
    st.title("📅 Weekly Summary")
    my_tasks = get_my_tasks()
    week_start = datetime.now() - timedelta(days=7)
    week_tasks = my_tasks[my_tasks["created_at"] >= week_start]
    total = len(week_tasks)
    completed = len(week_tasks[week_tasks["status"] == "Completed"])
    rate = (completed / total * 100) if total > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Created", total)
    col2.metric("Completed", completed)
    col3.metric("Rate", f"{rate:.1f}%")

# 10. Top Days
def show_top_days():
    st.title("🏆 Top Productive Days")
    my_tasks = get_my_tasks()
    completed = my_tasks[my_tasks["status"] == "Completed"].copy()
    if not completed.empty:
        completed["date"] = completed["completed_at"].dt.date
        daily = completed.groupby("date").size().reset_index(name="Tasks")
        top = daily.sort_values("Tasks", ascending=False).head(5)
        for i, row in top.iterrows():
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
            st.write(f"{medal} {row['date']}: {row['Tasks']} tasks")
    else:
        st.info("No completed tasks")

# 11. Trend
def show_trend():
    st.title("📈 Productivity Trend")
    my_tasks = get_my_tasks()
    completed = my_tasks[my_tasks["status"] == "Completed"].copy()
    if not completed.empty:
        completed["date"] = completed["completed_at"].dt.date
        daily = completed.groupby("date").size().reset_index(name="count")
        fig, ax = plt.subplots()
        ax.plot(daily["date"], daily["count"], marker='o')
        ax.set_xlabel("Date")
        ax.set_ylabel("Tasks")
        st.pyplot(fig)
    else:
        st.info("No data")

# 12. Smart Trend
def show_smart_trend():
    st.title("🎯 Smart Analysis")
    my_tasks = get_my_tasks()
    total = len(my_tasks)
    completed = len(my_tasks[my_tasks["status"] == "Completed"])
    rate = (completed / total * 100) if total > 0 else 0
    st.metric("Completion Rate", f"{rate:.1f}%")
    if rate >= 70:
        st.success("Excellent!")
    elif rate >= 50:
        st.info("Good progress!")
    else:
        st.warning("Need improvement")

# 13. AI Plan
def show_ai_plan():
    st.title("🤖 AI Daily Plan")
    my_tasks = get_my_tasks()
    pending = my_tasks[my_tasks["status"] == "Pending"]
    if pending.empty:
        st.success("All caught up!")
    else:
        top = pending.sort_values("ai_prediction", ascending=False).head(5)
        st.write("**Top 5 recommended tasks:**")
        for i, (_, t) in enumerate(top.iterrows(), 1):
            st.write(f"{i}. {t['task']} - AI: {t['ai_prediction']:.0f}% | {time_left_str(t['deadline'])}")

# 14. Reminders
def show_reminders():
    st.title("🔔 Reminders")
    my_tasks = get_my_tasks()
    pending = my_tasks[my_tasks["status"] == "Pending"]
    urgent = []
    for _, t in pending.iterrows():
        if not pd.isna(t["deadline"]):
            diff = (t["deadline"] - pd.Timestamp.now()).total_seconds()
            if diff <= 0:
                urgent.append(("OVERDUE", t))
            elif diff <= 3600:
                urgent.append(("URGENT", t))
    if urgent:
        for label, t in urgent:
            st.error(f"{label}: {t['task']}")
    else:
        st.success("No urgent reminders")

# 15. By Category
def show_by_category():
    st.title("📂 Tasks by Category")
    my_tasks = get_my_tasks()
    for cat in CATEGORIES:
        cat_tasks = my_tasks[my_tasks["category"] == cat]
        if not cat_tasks.empty:
            st.subheader(f"{CATEGORIES[cat]} {cat}")
            for _, t in cat_tasks.iterrows():
                st.write(f"#{int(t['id'])} {t['task']} - {t['status']}")

# 16. Search Tags
def show_search_tags():
    st.title("🔎 Search by Tags")
    my_tasks = get_my_tasks()
    tag = st.text_input("Enter tag:")
    if tag:
        filtered = my_tasks[my_tasks["tags"].str.contains(tag, case=False, na=False)]
        st.write(f"**{len(filtered)} tasks found**")
        for _, t in filtered.iterrows():
            st.write(f"#{int(t['id'])} {t['task']}")

# 17. Category Summary
def show_cat_summary():
    st.title("📋 Category Summary")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No tasks")
    else:
        summary = my_tasks.groupby("category").agg({"status": ["count", lambda x: (x == "Completed").sum()]}).reset_index()
        summary.columns = ["Category", "Total", "Completed"]
        summary["Rate %"] = (summary["Completed"] / summary["Total"] * 100).round(1)
        st.dataframe(summary, use_container_width=True)

# 18. Create Habit
def show_create_habit():
    st.title("🔥 Create Habit")
    with st.form("habit_form"):
        name = st.text_input("Habit Name")
        col1, col2 = st.columns(2)
        recurrence = col1.selectbox("Frequency", ["daily", "weekly", "monthly"])
        category = col2.selectbox("Category", list(CATEGORIES.keys()))
        if st.form_submit_button("Create"):
            if name.strip():
                new = pd.DataFrame([{"habit_id": st.session_state.habit_id_counter, "habit_name": name, "recurrence": recurrence, 
                                    "category": category, "active": True, "created_at": pd.Timestamp.now(), 
                                    "last_completed": pd.NaT, "total_completions": 0}])
                st.session_state.df_habits = pd.concat([st.session_state.df_habits, new], ignore_index=True)
                create_task_from_habit(st.session_state.habit_id_counter, name, recurrence, category)
                st.session_state.habit_id_counter += 1
                save_data()
                st.success("Created!")
                time.sleep(1)
                st.rerun()

# 19. View Habits
def show_view_habits():
    st.title("🔥 View Habits")
    active = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
    if active.empty:
        st.info("No habits")
    else:
        for _, h in active.iterrows():
            streak = calculate_streak(int(h["habit_id"]))
            st.write(f"#{int(h['habit_id'])} {h['habit_name']} - {h['recurrence']} - 🔥 {streak} streak")

# 20. Habit Dashboard
def show_habit_dash():
    st.title("🏅 Habit Dashboard")
    active = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
    if active.empty:
        st.info("No habits")
    else:
        for i, (_, h) in enumerate(active.iterrows()):
            streak = calculate_streak(int(h["habit_id"]))
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
            st.write(f"{medal} {h['habit_name']}: {streak} streak ({int(h['total_completions'])} total)")

# 21. Add Member
def show_add_member():
    st.title("👥 Add Team Member")
    with st.form("member_form"):
        username = st.text_input("Username")
        role = st.selectbox("Role", ["member", "manager"])
        if st.form_submit_button("Add"):
            if username.strip() and username not in st.session_state.df_users["username"].values:
                new = pd.DataFrame([{"username": username, "role": role, "added_at": pd.Timestamp.now(), 
                                    "added_by": st.session_state.current_user}])
                st.session_state.df_users = pd.concat([st.session_state.df_users, new], ignore_index=True)
                save_data()
                st.success(f"Added {username}")
                time.sleep(1)
                st.rerun()

# 22. View Team
def show_view_team():
    st.title("👥 View Team")
    if st.session_state.df_users.empty:
        st.info("No members")
    else:
        for _, u in st.session_state.df_users.iterrows():
            user_tasks = st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"] == u["username"]]
            st.write(f"{u['username']} ({u['role']}): {len(user_tasks)} tasks")

# 23. Assign
def show_assign():
    st.title("📤 Assign Task")
    my_tasks = get_my_tasks()
    if my_tasks.empty or len(st.session_state.df_users) <= 1:
        st.info("Add tasks and team members first")
    else:
        task_opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in my_tasks.iterrows()}
        task_sel = st.selectbox("Task:", list(task_opts.keys()))
        user_opts = st.session_state.df_users["username"].tolist()
        user_sel = st.selectbox("Assign to:", user_opts)
        if st.button("Assign"):
            st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_opts[task_sel], "assigned_to"] = user_sel
            st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_opts[task_sel], "shared"] = True
            save_data()
            st.success("Assigned!")
            time.sleep(1)
            st.rerun()

# 24. My Assigned
def show_my_assigned():
    st.title("📥 My Assigned Tasks")
    show_view_tasks()

# 25. Comment
def show_comment():
    st.title("💬 Comments")
    my_tasks = get_my_tasks()
    if my_tasks.empty:
        st.info("No tasks")
    else:
        task_opts = {f"#{int(r['id'])} - {r['task']}": int(r['id']) for _, r in my_tasks.iterrows()}
        task_sel = st.selectbox("Task:", list(task_opts.keys()))
        task_id = task_opts[task_sel]
        
        comments = st.session_state.df_comments[st.session_state.df_comments["task_id"] == task_id]
        if not comments.empty:
            for _, c in comments.iterrows():
                st.info(f"**{c['username']}** ({c['timestamp'].strftime('%Y-%m-%d %H:%M')}): {c['comment']}")
        
        with st.form("comment_form"):
            new_comment = st.text_area("Add comment:")
            if st.form_submit_button("Add"):
                if new_comment.strip():
                    new = pd.DataFrame([{"comment_id": st.session_state.comment_id_counter, "task_id": task_id, 
                                        "username": st.session_state.current_user, "comment": new_comment, 
                                        "timestamp": pd.Timestamp.now()}])
                    st.session_state.df_comments = pd.concat([st.session_state.df_comments, new], ignore_index=True)
                    st.session_state.comment_id_counter += 1
                    save_data()
                    st.success("Added!")
                    time.sleep(1)
                    st.rerun()

# 26. Team Dashboard
def show_team_dash():
    st.title("📊 Team Dashboard")
    if len(st.session_state.df_users) <= 1:
        st.info("Add team members")
    else:
        stats = []
        for _, u in st.session_state.df_users.iterrows():
            user_tasks = st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"] == u["username"]]
            total = len(user_tasks)
            completed = len(user_tasks[user_tasks["status"] == "Completed"])
            rate = (completed / total * 100) if total > 0 else 0
            stats.append({"Member": u["username"], "Total": total, "Completed": completed, "Rate %": round(rate, 1)})
        df = pd.DataFrame(stats)
        st.dataframe(df, use_container_width=True)

# 27. Train AI
def show_train_ai():
    st.title("🧠 Train AI Model")
    my_tasks = get_my_tasks()
    completed = my_tasks[my_tasks["status"] == "Completed"]
    st.write(f"**Training data:** {len(completed)} completed tasks")
    if len(completed) < 10:
        st.warning("Need 10+ completed tasks")
    else:
        if st.button("Train Model"):
            d = my_tasks.copy()
            d["priority_num"] = d["priority"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
            d["task_len"] = d["task"].astype(str).apply(len)
            d["completed"] = (d["status"] == "Completed").astype(int)
            X = d[["priority_num", "task_len"]]
            y = d["completed"]
            if len(y.unique()) >= 2:
                model = LogisticRegression(max_iter=200)
                model.fit(X, y)
                st.session_state.trained_model = model
                for idx, row in st.session_state.df_tasks.iterrows():
                    prob = predict_prob(row)
                    st.session_state.df_tasks.at[idx, "ai_prediction"] = round(prob * 100, 2)
                save_data()
                st.success("✅ Model trained!")

# 28. Export CSV
def show_export_csv():
    st.title("📥 Export CSV")
    my_tasks = get_my_tasks()
    csv = my_tasks.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, f"assan_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    st.success(f"Ready to export {len(my_tasks)} tasks")

# 29. Export Excel
def show_export_excel():
    st.title("📊 Export Excel")
    my_tasks = get_my_tasks()
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            my_tasks.to_excel(writer, sheet_name='Tasks', index=False)
        output.seek(0)
        st.download_button("⬇️ Download Excel", output, f"assan_{datetime.now().strftime('%Y%m%d')}.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        st.success(f"Ready to export {len(my_tasks)} tasks")
    except:
        st.error("openpyxl not installed")

# 30. Generate PDF
def show_gen_pdf():
    st.title("📄 Generate PDF")
    st.info("PDF generation requires reportlab library. Use CSV/Excel for now.")

# 31. Email
def show_email():
    st.title("📧 Email Summary")
    my_tasks = get_my_tasks()
    total = len(my_tasks)
    completed = len(my_tasks[my_tasks["status"] == "Completed"])
    rate = (completed / total * 100) if total > 0 else 0
    summary = f"""
ASSAN PRODUCTIVITY SUMMARY

User: {st.session_state.current_user}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Statistics:
• Total Tasks: {total}
• Completed: {completed}
• Pending: {total - completed}
• Completion Rate: {rate:.1f}%

Copy this summary to email!
    """
    st.text_area("Summary", summary, height=300)

# 32. List Files
def show_list_files():
    st.title("📁 List Export Files")
    files = glob.glob('export_*.csv') + glob.glob('export_*.xlsx') + glob.glob('report_*.pdf')
    if files:
        st.write("**Export files:**")
        for f in files:
            st.write(f"📄 {f}")
    else:
        st.info("No export files yet")

# 33. Timesheet
def show_timesheet():
    st.title("⏰ Timesheet")
    user_ts = st.session_state.df_timesheet[st.session_state.df_timesheet["username"] == st.session_state.current_user]
    if user_ts.empty:
        st.info("No time entries")
    else:
        total_min = user_ts["duration_minutes"].sum()
        total_hrs = total_min / 60
        col1, col2 = st.columns(2)
        col1.metric("Total Hours", f"{total_hrs:.2f}h")
        col2.metric("Total Minutes", f"{total_min:.0f}m")
        st.write("**Time Entries:**")
        for _, e in user_ts.sort_values("clock_in", ascending=False).iterrows():
            st.markdown(f"""<div class='card'>
            <p><strong>{e['task_name']}</strong></p>
            <p>In: {e['clock_in'].strftime('%Y-%m-%d %H:%M')} | Out: {e['clock_out'].strftime('%H:%M')}</p>
            <p>Duration: {e['duration_minutes']:.2f} min ({e['duration_minutes']/60:.2f} hrs)</p>
            </div>""", unsafe_allow_html=True)

# 34. Settings
def show_settings():
    st.title("⚙️ Settings")
    my_tasks = get_my_tasks()
    st.write(f"**User:** {st.session_state.current_user}")
    st.write(f"**Total Tasks:** {len(my_tasks)}")
    st.write(f"**Completed:** {len(my_tasks[my_tasks['status'] == 'Completed'])}")
    st.write(f"**Active Habits:** {len(st.session_state.df_habits[st.session_state.df_habits['active'] == True])}")
    if st.button("💾 Save All Data"):
        save_data()
        st.success("✅ Saved!")

# ===============================
# 1. User Profile Page
# ===============================
def show_profile():
    st.title("👤 Your Profile")
    
    st.write(f"**Username:** {st.session_state.current_user}")
    
    # Editable display name
    new_name = st.text_input("Display Name", value=st.session_state.current_user or "")
    
    if st.button("💾 Save Profile"):
        if new_name.strip():
            st.session_state.current_user = new_name.strip()
            st.success("✅ Profile updated!")

    # Back to dashboard
    st.divider()
    if st.button("🏠 Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.experimental_rerun()


# ===============================
# 2. Main App Controller
# ===============================
def main():
    # Load data if tasks are empty but file exists
    if st.session_state.df_tasks.empty and os.path.exists(DATA_FILE):
        load_data()

    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    # User not logged in
    if st.session_state.current_user is None:
        show_login()
        return

    # ------------------------
    # Page Router
    # ------------------------
    if st.session_state.page == "dashboard":
        show_main_app()  # your existing dashboard/sidebar function

        st.divider()
        st.subheader("🔙 Navigation")
        if st.button("👤 Go to Profile"):
            st.session_state.page = "profile"
            st.experimental_rerun()

    elif st.session_state.page == "profile":
        show_profile()

    elif st.session_state.page == "settings":
        show_settings()

        st.divider()
        if st.button("🏠 Back to Dashboard"):
            st.session_state.page = "dashboard"
            st.experimental_rerun()


# ===============================
# 3. Run App
# ===============================
if __name__ == "__main__":
    main()








