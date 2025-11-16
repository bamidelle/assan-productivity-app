# ==============================================
# ASSAN - COMPLETE STREAMLIT PRODUCTIVITY APP
# All 32 Features Included
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

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Assan - Productivity App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------- FILES ----------
DATA_FILE = "tasks.csv"
HABITS_FILE = "habits.csv"
USERS_FILE = "users.csv"
COMMENTS_FILE = "comments.csv"
REMINDER_LOG = "reminders.csv"
MODEL_FILE = "model.pkl"

# ---------- CATEGORIES ----------
CATEGORIES = {
    "Work": {"icon": "🏢", "color": "#3498db"},
    "Personal": {"icon": "🏠", "color": "#2ecc71"},
    "Learning": {"icon": "🎓", "color": "#9b59b6"},
    "Health": {"icon": "💪", "color": "#e74c3c"},
    "Creative": {"icon": "🎨", "color": "#e91e63"},
    "Other": {"icon": "📌", "color": "#00bcd4"}
}

# ---------- INITIALIZE SESSION STATE ----------
def init_session_state():
    defaults = {
        'current_user': None,
        'df_tasks': pd.DataFrame(),
        'df_habits': pd.DataFrame(),
        'df_users': pd.DataFrame(),
        'df_comments': pd.DataFrame(),
        'df_reminders': pd.DataFrame(),
        'task_id_counter': 1,
        'habit_id_counter': 1,
        'comment_id_counter': 1,
        'trained_model': None,
        'page': 'dashboard'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------- LOAD DATA ----------
def load_data():
    # Load Users
    if os.path.exists(USERS_FILE):
        st.session_state.df_users = pd.read_csv(USERS_FILE)
        st.session_state.df_users["added_at"] = pd.to_datetime(st.session_state.df_users["added_at"], errors='coerce')
    else:
        st.session_state.df_users = pd.DataFrame(columns=["username", "role", "added_at", "added_by"])
    
    # Load Tasks
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
    
    # Load Habits
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
    
    # Load Comments
    if os.path.exists(COMMENTS_FILE):
        st.session_state.df_comments = pd.read_csv(COMMENTS_FILE)
        st.session_state.df_comments["timestamp"] = pd.to_datetime(st.session_state.df_comments["timestamp"], errors='coerce')
        if not st.session_state.df_comments.empty:
            st.session_state.comment_id_counter = int(st.session_state.df_comments["comment_id"].max()) + 1
    else:
        st.session_state.df_comments = pd.DataFrame(columns=["comment_id", "task_id", "username", "comment", "timestamp"])
    
    # Load Reminders
    if os.path.exists(REMINDER_LOG):
        st.session_state.df_reminders = pd.read_csv(REMINDER_LOG)
        st.session_state.df_reminders["timestamp"] = pd.to_datetime(st.session_state.df_reminders["timestamp"], errors='coerce')
    else:
        st.session_state.df_reminders = pd.DataFrame(columns=["task_id", "task_name", "alert_type", "timestamp"])

# ---------- SAVE DATA ----------
def save_data():
    st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
    st.session_state.df_habits.to_csv(HABITS_FILE, index=False)
    st.session_state.df_users.to_csv(USERS_FILE, index=False)
    st.session_state.df_comments.to_csv(COMMENTS_FILE, index=False)
    if not st.session_state.df_reminders.empty:
        st.session_state.df_reminders.to_csv(REMINDER_LOG, index=False)

# ---------- UTILITY FUNCTIONS ----------
def parse_deadline_input(inp):
    if not inp or pd.isna(inp):
        return pd.NaT
    try:
        return pd.to_datetime(inp)
    except:
        return pd.NaT

def time_left_str(deadline_ts):
    if pd.isna(deadline_ts):
        return "No deadline"
    now = datetime.now()
    dl = deadline_ts.to_pydatetime() if isinstance(deadline_ts, pd.Timestamp) else deadline_ts
    diff = dl - now
    secs = int(diff.total_seconds())
    if secs <= 0:
        return "⚠️ OVERDUE"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "< 1m"

def predict_prob(row):
    """AI prediction"""
    if st.session_state.trained_model is not None:
        try:
            priority_num = 1 if str(row.get("priority","N")).upper()=="Y" else 0
            task_len = len(str(row.get("task", "")))
            X = np.array([[priority_num, task_len]])
            return float(st.session_state.trained_model.predict_proba(X)[0][1])
        except:
            pass
    return 0.8 if str(row.get("priority","N")).upper()=="Y" else 0.3

def calculate_streak(habit_id):
    if st.session_state.df_tasks.empty:
        return 0
    habit_tasks = st.session_state.df_tasks[
        (st.session_state.df_tasks["habit_id"] == habit_id) & 
        (st.session_state.df_tasks["status"] == "Completed")
    ].sort_values("completed_at", ascending=False)
    
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
    return st.session_state.df_tasks[
        st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
    ].copy()

# ---------- TASK OPERATIONS ----------
def complete_task(task_id):
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "status"] = "Completed"
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "completed_at"] = pd.Timestamp.now()
    
    # Check if it's a habit task
    task = st.session_state.df_tasks[st.session_state.df_tasks["id"] == task_id].iloc[0]
    if not pd.isna(task["habit_id"]):
        habit_id = int(task["habit_id"])
        st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"] == habit_id, "total_completions"] += 1
        st.session_state.df_habits.loc[st.session_state.df_habits["habit_id"] == habit_id, "last_completed"] = pd.Timestamp.now()
        # Create next occurrence
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
    
    new_task = pd.DataFrame([{
        "id": st.session_state.task_id_counter,
        "task": name,
        "priority": "Y",
        "status": "Pending",
        "created_at": pd.Timestamp.now(),
        "completed_at": pd.NaT,
        "deadline": deadline,
        "ai_prediction": round(prob * 100, 2),
        "category": category,
        "tags": f"habit,{recurrence}",
        "habit_id": habit_id,
        "recurrence": recurrence,
        "assigned_to": st.session_state.current_user,
        "created_by": st.session_state.current_user,
        "shared": False
    }])
    
    st.session_state.df_tasks = pd.concat([st.session_state.df_tasks, new_task], ignore_index=True)
    st.session_state.task_id_counter += 1

# ---------- LOGIN PAGE ----------
def show_login():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #e91e63; font-size: 4rem;'>🎯 ASSAN</h1>
            <p style='font-size: 1.5rem; color: #666;'>Your Complete Productivity Companion</p>
            <p style='color: #999;'>32 Powerful Features | AI-Powered | Team Collaboration</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 👤 Welcome Back")
        username = st.text_input("Enter your name:", key="login_username", placeholder="Your name...")
        
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            if username.strip():
                st.session_state.current_user = username.strip()
                
                # Add user if new
                if username not in st.session_state.df_users["username"].values:
                    new_user = pd.DataFrame([{
                        "username": username,
                        "role": "owner",
                        "added_at": pd.Timestamp.now(),
                        "added_by": "self"
                    }])
                    st.session_state.df_users = pd.concat([st.session_state.df_users, new_user], ignore_index=True)
                    save_data()
                
                st.rerun()
            else:
                st.error("Please enter your name")

# ---------- MAIN APP ----------
def show_main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.current_user}")
        
        my_tasks = get_my_tasks()
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        
        col1, col2 = st.columns(2)
        col1.metric("📋 Total", total)
        col2.metric("✅ Done", completed)
        
        if total > 0:
            completion_rate = (completed / total * 100)
            st.progress(completion_rate / 100)
            st.caption(f"Completion: {completion_rate:.1f}%")
        
        st.divider()
        
        # Menu with all features
        st.markdown("### 📋 Main Menu")
        menu = st.radio("Navigate:", [
            "🏠 Dashboard",
            "➕ Add Task",
            "📝 View Tasks",
            "🗑️ Remove Task",
            "✅ Complete Task",
            "✏️ Edit Task",
            "🔍 Filter Tasks",
            "🔥 Habits",
            "👥 Team",
            "📊 Analytics",
            "🤖 AI & Training",
            "📤 Export & Reports",
            "⚙️ Settings"
        ], label_visibility="collapsed")
        
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_user = None
            st.rerun()
    
    # Main Content Router
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "➕ Add Task":
        show_add_task()
    elif menu == "📝 View Tasks":
        show_view_tasks()
    elif menu == "🗑️ Remove Task":
        show_remove_task()
    elif menu == "✅ Complete Task":
        show_complete_task()
    elif menu == "✏️ Edit Task":
        show_edit_task()
    elif menu == "🔍 Filter Tasks":
        show_filter_tasks()
    elif menu == "🔥 Habits":
        show_habits()
    elif menu == "👥 Team":
        show_team()
    elif menu == "📊 Analytics":
        show_analytics()
    elif menu == "🤖 AI & Training":
        show_ai_features()
    elif menu == "📤 Export & Reports":
        show_exports()
    elif menu == "⚙️ Settings":
        show_settings()

# ---------- 1. DASHBOARD ----------
def show_dashboard():
    st.title("🏠 Dashboard")
    
    my_tasks = get_my_tasks()
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    total = len(my_tasks)
    completed = len(my_tasks[my_tasks["status"] == "Completed"])
    pending = len(my_tasks[my_tasks["status"] == "Pending"])
    high_priority = len(my_tasks[my_tasks["priority"] == "Y"])
    
    col1.metric("📋 Total Tasks", total)
    col2.metric("✅ Completed", completed)
    col3.metric("⏳ Pending", pending)
    col4.metric("⚡ High Priority", high_priority)
    
    st.divider()
    
    # Reminders / Urgent Tasks
    show_reminders_section()
    
    st.divider()
    
    # Recent Activity & Habits
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Tasks")
        recent = my_tasks.sort_values("created_at", ascending=False).head(5)
        if not recent.empty:
            for _, task in recent.iterrows():
                status_icon = "✅" if task["status"] == "Completed" else "⏳"
                st.write(f"{status_icon} {task['task']}")
        else:
            st.info("No tasks yet")
    
    with col2:
        st.subheader("🔥 Habit Streaks")
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        if not active_habits.empty:
            for _, habit in active_habits.head(5).iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                st.write(f"🔥 {habit['habit_name']}: **{streak}** day streak")
        else:
            st.info("No active habits")

# ---------- 2. ADD TASK ----------
def show_add_task():
    st.title("➕ Add New Task")
    
    with st.form("add_task_form"):
        task_name = st.text_input("📝 Task Name*", placeholder="Enter task description...")
        
        col1, col2 = st.columns(2)
        with col1:
            priority = st.selectbox("⚡ Priority", ["Normal", "High"])
        with col2:
            category = st.selectbox("📂 Category", list(CATEGORIES.keys()))
        
        col1, col2 = st.columns(2)
        with col1:
            deadline_date = st.date_input("📅 Deadline (Optional)", value=None)
        with col2:
            deadline_time = st.time_input("⏰ Time", value=datetime.now().time())
        
        tags = st.text_input("🏷️ Tags (comma separated)", placeholder="urgent, important")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("➕ Add Task", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🔄 Clear", use_container_width=True)
        
        if submitted:
            if task_name.strip():
                # Combine date and time
                if deadline_date:
                    deadline_dt = datetime.combine(deadline_date, deadline_time)
                else:
                    deadline_dt = pd.NaT
                
                priority_val = "Y" if priority == "High" else "N"
                prob = predict_prob({"priority": priority_val, "task": task_name})
                
                new_task = pd.DataFrame([{
                    "id": st.session_state.task_id_counter,
                    "task": task_name,
                    "priority": priority_val,
                    "status": "Pending",
                    "created_at": pd.Timestamp.now(),
                    "completed_at": pd.NaT,
                    "deadline": deadline_dt,
                    "ai_prediction": round(prob * 100, 2),
                    "category": category,
                    "tags": tags,
                    "habit_id": np.nan,
                    "recurrence": "none",
                    "assigned_to": st.session_state.current_user,
                    "created_by": st.session_state.current_user,
                    "shared": False
                }])
                
                st.session_state.df_tasks = pd.concat([st.session_state.df_tasks, new_task], ignore_index=True)
                st.session_state.task_id_counter += 1
                save_data()
                
                st.success(f"✅ Task added! AI Completion Prediction: {round(prob * 100, 2)}%")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter a task name")

# ---------- 3. VIEW TASKS ----------
def show_view_tasks():
    st.title("📝 My Tasks")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("📭 No tasks yet. Create your first task!")
        return
    
    st.write(f"**Total: {len(my_tasks)} tasks**")
    
    # Display tasks in a nice format
    for _, task in my_tasks.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            
            status_icon = "✅" if task["status"] == "Completed" else "⏳"
            priority_icon = "⚡" if task["priority"] == "Y" else ""
            
            with col1:
                st.write(f"**#{int(task['id'])}** {status_icon} {priority_icon} {task['task']}")
                st.caption(f"{CATEGORIES[task['category']]['icon']} {task['category']} | AI: {task['ai_prediction']:.0f}% | Created: {task['created_at'].strftime('%Y-%m-%d')}")
            
            with col2:
                st.write(f"**{time_left_str(task['deadline'])}**")
            
            with col3:
                st.write(f"**{task['status']}**")
            
            st.divider()

# ---------- 4. REMOVE TASK ----------
def show_remove_task():
    st.title("🗑️ Remove Task")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No tasks to remove")
        return
    
    task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in my_tasks.iterrows()}
    
    selected = st.selectbox("Select task to remove:", list(task_options.keys()))
    
    if selected:
        task_id = task_options[selected]
        task = my_tasks[my_tasks["id"] == task_id].iloc[0]
        
        st.warning(f"⚠️ You are about to delete: **{task['task']}**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
                delete_task(task_id)
                st.success("✅ Task deleted!")
                time.sleep(1)
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.info("Deletion cancelled")

# ---------- 5. COMPLETE TASK ----------
def show_complete_task():
    st.title("✅ Complete Task")
    
    my_tasks = get_my_tasks()
    pending = my_tasks[my_tasks["status"] == "Pending"]
    
    if pending.empty:
        st.success("🎉 All tasks completed!")
        return
    
    task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in pending.iterrows()}
    
    selected = st.selectbox("Select task to complete:", list(task_options.keys()))
    
    if selected:
        task_id = task_options[selected]
        task = pending[pending["id"] == task_id].iloc[0]
        
        st.info(f"📝 Task: **{task['task']}**")
        st.write(f"Category: {CATEGORIES[task['category']]['icon']} {task['category']}")
        st.write(f"Priority: {'⚡ High' if task['priority'] == 'Y' else '📍 Normal'}")
        
        if st.button("✅ Mark as Completed", type="primary", use_container_width=True):
            complete_task(task_id)
            
            # Check for habit streak
            if not pd.isna(task["habit_id"]):
                streak = calculate_streak(int(task["habit_id"]))
                st.success(f"🎉 Task completed! 🔥 {streak} day streak!")
            else:
                st.success("🎉 Task completed!")
            
            time.sleep(1)
            st.rerun()

# ---------- 6. EDIT TASK ----------
def show_edit_task():
    st.title("✏️ Edit Task")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No tasks to edit")
        return
    
    task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in my_tasks.iterrows()}
    selected = st.selectbox("Select task to edit:", list(task_options.keys()))
    
    if selected:
        task_id = task_options[selected]
        task = my_tasks[my_tasks["id"] == task_id].iloc[0]
        
        with st.form("edit_task_form"):
            new_name = st.text_input("Task Name", value=task['task'])
            new_priority = st.selectbox("Priority", ["Normal", "High"], 
                                       index=0 if task['priority'] == 'N' else 1)
            new_category = st.selectbox("Category", list(CATEGORIES.keys()),
                                       index=list(CATEGORIES.keys()).index(task['category']))
            
            if not pd.isna(task['deadline']):
                default_date = task['deadline'].date()
                default_time = task['deadline'].time()
            else:
                default_date = None
                default_time = datetime.now().time()
            
            new_deadline_date = st.date_input("Deadline Date", value=default_date)
            new_deadline_time = st.time_input("Deadline Time", value=default_time)
            
            if st.form_submit_button("💾 Save Changes", type="primary"):
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "task"] = new_name
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "priority"] = "Y" if new_priority == "High" else "N"
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "category"] = new_category
                
                if new_deadline_date:
                    new_deadline = datetime.combine(new_deadline_date, new_deadline_time)
                    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "deadline"] = new_deadline
                
                save_data()
                st.success("✅ Task updated!")
                time.sleep(1)
                st.rerun()

# ---------- 7. FILTER TASKS ----------
def show_filter_tasks():
    st.title("🔍 Filter & Search Tasks")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No tasks to filter")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("📊 Status", ["All", "Pending", "Completed"])
    with col2:
        priority_filter = st.selectbox("⚡ Priority", ["All", "High", "Normal"])
    with col3:
        category_filter = st.selectbox("📂 Category", ["All"] + list(CATEGORIES.keys()))
    
    # Tag search
    tag_search = st.text_input("🔎 Search by Tag", placeholder="Enter tag to search...")
    
    # Apply filters
    filtered = my_tasks.copy()
    
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    
    if priority_filter != "All":
        pri_val = "Y" if priority_filter == "High" else "N"
        filtered = filtered[filtered["priority"] == pri_val]
    
    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]
    
    if tag_search:
        filtered = filtered[filtered["tags"].str.contains(tag_search, case=False, na=False)]
    
    st.write(f"**Found {len(filtered)} tasks**")
    st.divider()
    
    # Display filtered results
    for _, task in filtered.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        status_icon = "✅" if task["status"] == "Completed" else "⏳"
        priority_icon = "⚡" if task["priority"] == "Y" else ""
        
        col1.write(f"**#{int(task['id'])}** {status_icon} {priority_icon} {task['task']}")
        col2.write(time_left_str(task['deadline']))
        col3.write(task['status'])
        st.divider()

# ---------- 8. HABITS ----------
def show_habits():
    st.title("🔥 Habits")
    
    tab1, tab2, tab3 = st.tabs(["📋 Active Habits", "➕ Create Habit", "📊 Habit Dashboard"])
    
    # TAB 1: View Habits
    with tab1:
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        
        if active_habits.empty:
            st.info("No habits yet. Create your first habit in the next tab!")
        else:
            for _, habit in active_habits.iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                
                with st.expander(f"🔥 {habit['habit_name']} - {streak} day streak"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📅 Frequency", habit['recurrence'].title())
                    col2.metric("🔥 Streak", f"{streak} days")
                    col3.metric("✅ Total", int(habit['total_completions']))
                    
                    st.write(f"**Category:** {CATEGORIES[habit['category']]['icon']} {habit['category']}")
                    st.write(f"**Created:** {habit['created_at'].strftime('%Y-%m-%d')}")
                    
                    if not pd.isna(habit['last_completed']):
                        st.write(f"**Last Completed:** {habit['last_completed'].strftime('%Y-%m-%d %H:%M')}")
    
    # TAB 2: Create Habit
    with tab2:
        with st.form("create_habit"):
            st.subheader("Create New Habit")
            habit_name = st.text_input("🎯 Habit Name", placeholder="Exercise daily, Read for 30 mins...")
            
            col1, col2 = st.columns(2)
            with col1:
                recurrence = st.selectbox("📅 Frequency", ["daily", "weekly", "monthly"])
            with col2:
                category = st.selectbox("📂 Category", list(CATEGORIES.keys()))
            
            if st.form_submit_button("🔥 Create Habit", type="primary"):
                if habit_name.strip():
                    new_habit = pd.DataFrame([{
                        "habit_id": st.session_state.habit_id_counter,
                        "habit_name": habit_name,
                        "recurrence": recurrence,
                        "category": category,
                        "active": True,
                        "created_at": pd.Timestamp.now(),
                        "last_completed": pd.NaT,
                        "total_completions": 0
                    }])
                    
                    st.session_state.df_habits = pd.concat([st.session_state.df_habits, new_habit], ignore_index=True)
                    
                    # Create first task
                    create_task_from_habit(st.session_state.habit_id_counter, habit_name, recurrence, category)
                    
                    st.session_state.habit_id_counter += 1
                    save_data()
                    
                    st.success(f"✅ Habit '{habit_name}' created!")
                    time.sleep(1)
                    st.rerun()
    
    # TAB 3: Habit Dashboard
    with tab3:
        st.subheader("🏆 Habit Leaderboard")
        
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        
        if not active_habits.empty:
            streaks = []
            for _, habit in active_habits.iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                streaks.append({
                    "Habit": habit["habit_name"],
                    "Streak": streak,
                    "Total": int(habit["total_completions"]),
                    "Category": habit["category"]
                })
            
            df_streaks = pd.DataFrame(streaks).sort_values("Streak", ascending=False)
            
            for i, row in df_streaks.iterrows():
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                st.write(f"{medal} **{row['Habit']}**: {row['Streak']} day streak ({row['Total']} total)")
        else:
            st.info("Create habits to see your leaderboard!")

# ---------- 9. TEAM ----------
def show_team():
    st.title("👥 Team Management")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Members", "➕ Add Member", "📤 Assign Task", "💬 Comments", "📊 Team Dashboard"])
    
    # TAB 1: View Team
    with tab1:
        if st.session_state.df_users.empty:
            st.info("No team members yet")
        else:
            for _, user in st.session_state.df_users.iterrows():
                user_tasks = st.session_state.df_tasks[
                    st.session_state.df_tasks["assigned_to"] == user["username"]
                ]
                total = len(user_tasks)
                completed = len(user_tasks[user_tasks["status"] == "Completed"])
                
                with st.expander(f"👤 {user['username']} ({user['role']})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📋 Tasks", total)
                    col2.metric("✅ Completed", completed)
                    if total > 0:
                        col3.metric("📈 Rate", f"{(completed/total*100):.1f}%")
                    
                    st.write(f"**Added:** {user['added_at'].strftime('%Y-%m-%d')}")
                    st.write(f"**Added by:** {user['added_by']}")
    
    # TAB 2: Add Member
    with tab2:
        with st.form("add_member"):
            st.subheader("Add Team Member")
            username = st.text_input("👤 Username")
            role = st.selectbox("🎭 Role", ["member", "manager"])
            
            if st.form_submit_button("➕ Add Member", type="primary"):
                if username.strip() and username not in st.session_state.df_users["username"].values:
                    new_user = pd.DataFrame([{
                        "username": username,
                        "role": role,
                        "added_at": pd.Timestamp.now(),
                        "added_by": st.session_state.current_user
                    }])
                    
                    st.session_state.df_users = pd.concat([st.session_state.df_users, new_user], ignore_index=True)
                    save_data()
                    st.success(f"✅ Added {username}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or already exists")
    
    # TAB 3: Assign Task
    with tab3:
        st.subheader("📤 Assign Task to Team Member")
        
        my_tasks = get_my_tasks()
        
        if my_tasks.empty:
            st.info("No tasks to assign")
        elif len(st.session_state.df_users) <= 1:
            st.info("Add team members first")
        else:
            task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in my_tasks.iterrows()}
            selected_task = st.selectbox("Select task:", list(task_options.keys()))
            
            user_options = st.session_state.df_users["username"].tolist()
            selected_user = st.selectbox("Assign to:", user_options)
            
            if st.button("📤 Assign Task", type="primary"):
                task_id = task_options[selected_task]
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "assigned_to"] = selected_user
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "shared"] = True
                save_data()
                st.success(f"✅ Task assigned to {selected_user}")
                time.sleep(1)
                st.rerun()
    
    # TAB 4: Comments
    with tab4:
        st.subheader("💬 Task Comments")
        
        my_tasks = get_my_tasks()
        
        if my_tasks.empty:
            st.info("No tasks to comment on")
        else:
            task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in my_tasks.iterrows()}
            selected_task = st.selectbox("Select task:", list(task_options.keys()), key="comment_task")
            
            task_id = task_options[selected_task]
            
            # Show existing comments
            task_comments = st.session_state.df_comments[st.session_state.df_comments["task_id"] == task_id]
            
            if not task_comments.empty:
                st.write("**Existing Comments:**")
                for _, comment in task_comments.iterrows():
                    st.info(f"👤 **{comment['username']}** ({comment['timestamp'].strftime('%Y-%m-%d %H:%M')})\n\n{comment['comment']}")
            
            # Add new comment
            with st.form("add_comment"):
                new_comment = st.text_area("Add comment:", placeholder="Enter your comment...")
                
                if st.form_submit_button("💬 Add Comment"):
                    if new_comment.strip():
                        new_comment_row = pd.DataFrame([{
                            "comment_id": st.session_state.comment_id_counter,
                            "task_id": task_id,
                            "username": st.session_state.current_user,
                            "comment": new_comment,
                            "timestamp": pd.Timestamp.now()
                        }])
                        
                        st.session_state.df_comments = pd.concat([st.session_state.df_comments, new_comment_row], ignore_index=True)
                        st.session_state.comment_id_counter += 1
                        save_data()
                        st.success("✅ Comment added!")
                        time.sleep(1)
                        st.rerun()
    
    # TAB 5: Team Dashboard
    with tab5:
        st.subheader("📊 Team Performance Dashboard")
        
        if len(st.session_state.df_users) <= 1:
            st.info("Add team members to see team dashboard")
        else:
            team_stats = []
            for _, user in st.session_state.df_users.iterrows():
                user_tasks = st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"] == user["username"]]
                total = len(user_tasks)
                completed = len(user_tasks[user_tasks["status"] == "Completed"])
                rate = (completed / total * 100) if total > 0 else 0
                
                team_stats.append({
                    "Member": user["username"],
                    "Total": total,
                    "Completed": completed,
                    "Pending": total - completed,
                    "Rate %": round(rate, 1)
                })
            
            df_team = pd.DataFrame(team_stats).sort_values("Rate %", ascending=False)
            st.dataframe(df_team, use_container_width=True)

# ---------- 10. ANALYTICS ----------
def show_analytics():
    st.title("📊 Analytics & Insights")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No data yet")
        return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📅 Weekly", "🏆 Top Days", "📈 Trends", "🎯 Smart Analysis"])
    
    # TAB 1: Chart/Overview
    with tab1:
        st.subheader("📊 Category Breakdown")
        
        category_stats = my_tasks.groupby("category").agg({
            "status": ["count", lambda x: (x == "Completed").sum()]
        }).reset_index()
        category_stats.columns = ["Category", "Total", "Completed"]
        category_stats["Rate %"] = (category_stats["Completed"] / category_stats["Total"] * 100).round(1)
        
        st.dataframe(category_stats, use_container_width=True)
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(category_stats["Total"], labels=category_stats["Category"], autopct='%1.1f%%', startangle=90)
        ax.set_title("Tasks by Category")
        st.pyplot(fig)
    
    # TAB 2: Weekly Summary
    with tab2:
        st.subheader("📅 Weekly Summary")
        
        week_start = datetime.now() - timedelta(days=7)
        week_tasks = my_tasks[my_tasks["created_at"] >= week_start]
        
        total = len(week_tasks)
        completed = len(week_tasks[week_tasks["status"] == "Completed"])
        rate = (completed / total * 100) if total > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📋 Tasks Created", total)
        col2.metric("✅ Completed", completed)
        col3.metric("📈 Completion Rate", f"{rate:.1f}%")
        
        st.write("**This Week's Activity**")
        if not week_tasks.empty:
            for _, task in week_tasks.head(10).iterrows():
                status_icon = "✅" if task["status"] == "Completed" else "⏳"
                st.write(f"{status_icon} {task['task']}")
    
    # TAB 3: Top Days
    with tab3:
        st.subheader("🏆 Most Productive Days")
        
        completed_tasks = my_tasks[my_tasks["status"] == "Completed"].copy()
        
        if not completed_tasks.empty:
            completed_tasks["date"] = completed_tasks["completed_at"].dt.date
            daily_counts = completed_tasks.groupby("date").size().reset_index(name="Tasks Completed")
            top_days = daily_counts.sort_values("Tasks Completed", ascending=False).head(5)
            
            for i, row in top_days.iterrows():
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                st.write(f"{medal} **{row['date']}**: {row['Tasks Completed']} tasks")
        else:
            st.info("No completed tasks yet")
    
    # TAB 4: Productivity Trend
    with tab4:
        st.subheader("📈 Daily Productivity Trend")
        
        completed_tasks = my_tasks[my_tasks["status"] == "Completed"].copy()
        
        if not completed_tasks.empty:
            completed_tasks["date"] = completed_tasks["completed_at"].dt.date
            daily_counts = completed_tasks.groupby("date").size().reset_index(name="count")
            daily_counts = daily_counts.sort_values("date")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_counts["date"], daily_counts["count"], marker='o', color='#e91e63', linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Tasks Completed")
            ax.set_title("Daily Completion Trend")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No completed tasks to show trend")
    
    # TAB 5: Smart Trend Analysis
    with tab5:
        st.subheader("🎯 Smart AI Analysis")
        
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        pending = len(my_tasks[my_tasks["status"] == "Pending"])
        rate = (completed / total * 100) if total > 0 else 0
        
        st.metric("Overall Completion Rate", f"{rate:.1f}%")
        
        if rate >= 70:
            st.success("🎉 **Excellent productivity!** You're crushing your goals!")
        elif rate >= 50:
            st.info("👍 **Good progress!** Keep up the momentum!")
        else:
            st.warning("💪 **Room for improvement!** Focus on completing pending tasks.")
        
        # AI recommendations
        st.subheader("🤖 AI Recommendations")
        
        high_priority = my_tasks[my_tasks["priority"] == "Y"]
        if not high_priority.empty:
            st.write(f"📌 You have **{len(high_priority)}** high-priority tasks")
        
        overdue = my_tasks[my_tasks["deadline"] < pd.Timestamp.now()]
        if not overdue.empty:
            st.warning(f"⚠️ **{len(overdue)}** tasks are overdue!")

# ---------- 11. AI FEATURES ----------
def show_ai_features():
    st.title("🤖 AI Features")
    
    tab1, tab2 = st.tabs(["🎯 AI Daily Plan", "🧠 Train AI Model"])
    
    # TAB 1: Daily AI Plan
    with tab1:
        st.subheader("🎯 Your AI-Powered Daily Plan")
        
        my_tasks = get_my_tasks()
        pending = my_tasks[my_tasks["status"] == "Pending"]
        
        if pending.empty:
            st.success("🎉 No pending tasks! You're all caught up!")
        else:
            st.write("**Top 5 tasks AI recommends you focus on today:**")
            
            # Sort by AI prediction
            top_tasks = pending.sort_values("ai_prediction", ascending=False).head(5)
            
            for i, (_, task) in enumerate(top_tasks.iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([4, 1, 1])
                    
                    col1.write(f"**{i}. {task['task']}**")
                    col1.caption(f"{CATEGORIES[task['category']]['icon']} {task['category']}")
                    
                    col2.write(f"**AI: {task['ai_prediction']:.0f}%**")
                    col3.write(time_left_str(task['deadline']))
                    
                    if st.button(f"✅ Complete #{int(task['id'])}", key=f"ai_complete_{task['id']}"):
                        complete_task(int(task['id']))
                        st.success("Task completed!")
                        time.sleep(0.5)
                        st.rerun()
                    
                    st.divider()
    
    # TAB 2: Train Model
    with tab2:
        st.subheader("🧠 Train AI Model")
        
        my_tasks = get_my_tasks()
        completed = my_tasks[my_tasks["status"] == "Completed"]
        
        st.write(f"**Training Data Available:** {len(completed)} completed tasks")
        
        if len(completed) < 10:
            st.warning(f"⚠️ Need at least 10 completed tasks to train AI. You have {len(completed)}.")
        else:
            if st.button("🚀 Train AI Model", type="primary"):
                with st.spinner("Training AI model..."):
                    # Simple model training
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
                        
                        # Update predictions
                        for idx, row in st.session_state.df_tasks.iterrows():
                            prob = predict_prob(row)
                            st.session_state.df_tasks.at[idx, "ai_prediction"] = round(prob * 100, 2)
                        
                        save_data()
                        st.success("✅ AI Model trained successfully!")
                    else:
                        st.error("Need both completed and pending tasks")

# ---------- 12. EXPORTS ----------
def show_exports():
    st.title("📤 Export & Reports")
    
    my_tasks = get_my_tasks()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📥 CSV", "📊 Excel", "📄 PDF", "📧 Email Summary"])
    
    # TAB 1: CSV Export
    with tab1:
        st.subheader("📥 Export to CSV")
        
        if st.button("Generate CSV", type="primary"):
            csv = my_tasks.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download CSV File",
                data=csv,
                file_name=f"assan_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(f"✅ CSV ready! {len(my_tasks)} tasks exported")
    
    # TAB 2: Excel Export
    with tab2:
        st.subheader("📊 Export to Excel")
        
        if st.button("Generate Excel", type="primary"):
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    my_tasks.to_excel(writer, sheet_name='All Tasks', index=False)
                    
                    # Summary sheet
                    summary = my_tasks.groupby("category").agg({
                        "status": ["count", lambda x: (x == "Completed").sum()]
                    }).reset_index()
                    summary.columns = ["Category", "Total", "Completed"]
                    summary["Rate %"] = (summary["Completed"] / summary["Total"] * 100).round(1)
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=output,
                    file_name=f"assan_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                st.success("✅ Excel ready with multiple sheets!")
            except:
                st.error("❌ openpyxl not installed. Use CSV export instead.")
    
    # TAB 3: PDF Report
    with tab3:
        st.subheader("📄 PDF Report")
        st.info("📝 PDF generation requires reportlab library. Use CSV/Excel for now.")
    
    # TAB 4: Email Summary
    with tab4:
        st.subheader("📧 Email Summary")
        
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        pending = total - completed
        rate = (completed / total * 100) if total > 0 else 0
        
        summary_text = f"""
**Assan Productivity Summary**

User: {st.session_state.current_user}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

═══════════════════════════════

📊 STATISTICS:
• Total Tasks: {total}
• Completed: {completed}
• Pending: {pending}
• Completion Rate: {rate:.1f}%

═══════════════════════════════

Copy this summary and email it to yourself or your team!
        """
        
        st.text_area("Email Summary", summary_text, height=300)
        
        if st.button("📋 Copy to Clipboard"):
            st.info("📋 Copy the text above and paste into your email!")

# ---------- 13. REMINDERS ----------
def show_reminders_section():
    """Show reminders in dashboard"""
    my_tasks = get_my_tasks()
    pending = my_tasks[my_tasks["status"] == "Pending"]
    
    urgent_tasks = []
    warning_tasks = []
    
    for _, task in pending.iterrows():
        if not pd.isna(task["deadline"]):
            diff = (task["deadline"] - pd.Timestamp.now()).total_seconds()
            
            if diff <= 0:
                urgent_tasks.append(("🔴 OVERDUE", task))
            elif diff <= 3600:  # 1 hour
                urgent_tasks.append(("🟠 URGENT", task))
            elif diff <= 86400:  # 24 hours
                warning_tasks.append(("🟡 DUE SOON", task))
    
    if urgent_tasks:
        st.subheader("🔔 Urgent Reminders")
        for label, task in urgent_tasks[:5]:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.error(f"{label}: {task['task']}")
            col2.write(time_left_str(task['deadline']))
            if col3.button("✅", key=f"remind_complete_{task['id']}"):
                complete_task(int(task['id']))
                st.rerun()

# ---------- 14. SETTINGS ----------
def show_settings():
    st.title("⚙️ Settings")
    
    tab1, tab2 = st.tabs(["👤 User Info", "📁 Data Management"])
    
    with tab1:
        st.subheader("👤 User Information")
        
        my_tasks = get_my_tasks()
        
        col1, col2 = st.columns(2)
        col1.write(f"**Username:** {st.session_state.current_user}")
        col1.write(f"**Total Tasks:** {len(my_tasks)}")
        col1.write(f"**Completed:** {len(my_tasks[my_tasks['status'] == 'Completed'])}")
        
        col2.write(f"**Active Habits:** {len(st.session_state.df_habits[st.session_state.df_habits['active'] == True])}")
        col2.write(f"**Team Members:** {len(st.session_state.df_users)}")
        col2.write(f"**Comments:** {len(st.session_state.df_comments)}")
    
    with tab2:
        st.subheader("📁 Data Management")
        
        st.write("**Export Files:**")
        
        # List export files
        export_files = []
        for pattern in ['export_*.csv', 'export_*.xlsx', 'report_*.pdf']:
            import glob
            export_files.extend(glob.glob(pattern))
        
        if export_files:
            for file in export_files:
                st.write(f"📄 {file}")
        else:
            st.info("No export files yet")
        
        st.divider()
        
        if st.button("💾 Save All Data Now", type="primary"):
            save_data()
            st.success("✅ All data saved!")

# ---------- MAIN ----------
def main():
    # Load data on first run
    if st.session_state.df_tasks.empty and os.path.exists(DATA_FILE):
        load_data()
    
    # Show login or main app
    if st.session_state.current_user is None:
        show_login()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
