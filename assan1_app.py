# ==============================================
# ASSAN - STREAMLIT PRODUCTIVITY WEB APP
# Full Web-Based Implementation
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
warnings.filterwarnings('ignore')

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Assan - Productivity App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- FILES ----------
DATA_FILE = "tasks.csv"
HABITS_FILE = "habits.csv"
USERS_FILE = "users.csv"
COMMENTS_FILE = "comments.csv"
REMINDER_LOG = "reminders.csv"

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
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'df_tasks' not in st.session_state:
    st.session_state.df_tasks = pd.DataFrame()
if 'df_habits' not in st.session_state:
    st.session_state.df_habits = pd.DataFrame()
if 'df_users' not in st.session_state:
    st.session_state.df_users = pd.DataFrame()
if 'df_comments' not in st.session_state:
    st.session_state.df_comments = pd.DataFrame()
if 'task_id_counter' not in st.session_state:
    st.session_state.task_id_counter = 1
if 'habit_id_counter' not in st.session_state:
    st.session_state.habit_id_counter = 1
if 'comment_id_counter' not in st.session_state:
    st.session_state.comment_id_counter = 1

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
        for col, default in [("category", "Other"), ("tags", ""), ("ai_prediction", np.nan), 
                            ("habit_id", np.nan), ("recurrence", "none"), ("assigned_to", ""), 
                            ("created_by", ""), ("shared", False), ("deadline", pd.NaT)]:
            if col not in st.session_state.df_tasks.columns:
                st.session_state.df_tasks[col] = default
        st.session_state.task_id_counter = int(st.session_state.df_tasks["id"].max()) + 1 if not st.session_state.df_tasks.empty else 1
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
        st.session_state.habit_id_counter = int(st.session_state.df_habits["habit_id"].max()) + 1 if not st.session_state.df_habits.empty else 1
    else:
        st.session_state.df_habits = pd.DataFrame(columns=[
            "habit_id","habit_name","recurrence","category","active","created_at","last_completed","total_completions"
        ])
    
    # Load Comments
    if os.path.exists(COMMENTS_FILE):
        st.session_state.df_comments = pd.read_csv(COMMENTS_FILE)
        st.session_state.df_comments["timestamp"] = pd.to_datetime(st.session_state.df_comments["timestamp"], errors='coerce')
        st.session_state.comment_id_counter = int(st.session_state.df_comments["comment_id"].max()) + 1 if not st.session_state.df_comments.empty else 1
    else:
        st.session_state.df_comments = pd.DataFrame(columns=["comment_id", "task_id", "username", "comment", "timestamp"])

# ---------- SAVE DATA ----------
def save_data():
    st.session_state.df_tasks.to_csv(DATA_FILE, index=False)
    st.session_state.df_habits.to_csv(HABITS_FILE, index=False)
    st.session_state.df_users.to_csv(USERS_FILE, index=False)
    st.session_state.df_comments.to_csv(COMMENTS_FILE, index=False)

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
    """Simple AI prediction"""
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

# ---------- LOGIN PAGE ----------
def show_login():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #e91e63; font-size: 3rem;'>🎯 ASSAN</h1>
            <p style='font-size: 1.2rem; color: #666;'>Your Productivity Companion</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 👤 Login")
        username = st.text_input("Enter your name:", key="login_username")
        
        if st.button("🚀 Start", type="primary", use_container_width=True):
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
        
        my_tasks = st.session_state.df_tasks[
            st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
        ]
        
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        pending = total - completed
        
        col1, col2 = st.columns(2)
        col1.metric("Total Tasks", total)
        col2.metric("Completed", completed)
        
        if total > 0:
            completion_rate = (completed / total * 100)
            st.progress(completion_rate / 100)
            st.caption(f"Completion Rate: {completion_rate:.1f}%")
        
        st.divider()
        
        menu = st.radio("📋 Menu", [
            "🏠 Dashboard",
            "➕ Add Task",
            "📝 My Tasks",
            "🔥 Habits",
            "👥 Team",
            "📊 Analytics",
            "⚙️ Settings"
        ])
        
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_user = None
            st.rerun()
    
    # Main Content
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "➕ Add Task":
        show_add_task()
    elif menu == "📝 My Tasks":
        show_my_tasks()
    elif menu == "🔥 Habits":
        show_habits()
    elif menu == "👥 Team":
        show_team()
    elif menu == "📊 Analytics":
        show_analytics()
    elif menu == "⚙️ Settings":
        show_settings()

# ---------- DASHBOARD ----------
def show_dashboard():
    st.title("🏠 Dashboard")
    
    my_tasks = st.session_state.df_tasks[
        st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
    ]
    
    # Metrics
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
    
    # Urgent Tasks
    pending_tasks = my_tasks[my_tasks["status"] == "Pending"].copy()
    if not pending_tasks.empty:
        st.subheader("🔔 Urgent Tasks")
        
        urgent_tasks = []
        for _, task in pending_tasks.iterrows():
            if not pd.isna(task["deadline"]):
                diff = (task["deadline"] - pd.Timestamp.now()).total_seconds()
                if diff < 3600:  # Less than 1 hour
                    urgent_tasks.append(task)
        
        if urgent_tasks:
            for task in urgent_tasks[:5]:
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"**{task['task']}**")
                col2.write(time_left_str(task['deadline']))
                if col3.button("✅", key=f"complete_urgent_{task['id']}"):
                    complete_task(int(task['id']))
        else:
            st.info("No urgent tasks")
    
    st.divider()
    
    # Recent Activity
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
        st.subheader("🔥 Active Habits")
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        if not active_habits.empty:
            for _, habit in active_habits.head(5).iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                st.write(f"🔥 {habit['habit_name']}: {streak} day streak")
        else:
            st.info("No active habits")

# ---------- ADD TASK ----------
def show_add_task():
    st.title("➕ Add New Task")
    
    with st.form("add_task_form"):
        task_name = st.text_input("📝 Task Name", placeholder="Enter task description...")
        
        col1, col2 = st.columns(2)
        with col1:
            priority = st.selectbox("⚡ Priority", ["Normal", "High"])
        with col2:
            category = st.selectbox("📂 Category", list(CATEGORIES.keys()))
        
        deadline = st.date_input("📅 Deadline (Optional)")
        deadline_time = st.time_input("⏰ Time (Optional)")
        
        tags = st.text_input("🏷️ Tags (comma separated)", placeholder="urgent, important")
        
        submitted = st.form_submit_button("➕ Add Task", type="primary", use_container_width=True)
        
        if submitted:
            if task_name.strip():
                # Combine date and time
                if deadline:
                    deadline_dt = datetime.combine(deadline, deadline_time)
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
                
                st.success(f"✅ Task added successfully! AI Prediction: {round(prob * 100, 2)}%")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Please enter a task name")

# ---------- MY TASKS ----------
def show_my_tasks():
    st.title("📝 My Tasks")
    
    my_tasks = st.session_state.df_tasks[
        st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
    ].copy()
    
    if my_tasks.empty:
        st.info("No tasks yet. Add your first task!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Completed"])
    with col2:
        priority_filter = st.selectbox("Priority", ["All", "High", "Normal"])
    with col3:
        category_filter = st.selectbox("Category", ["All"] + list(CATEGORIES.keys()))
    
    # Apply filters
    filtered = my_tasks.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if priority_filter != "All":
        pri_val = "Y" if priority_filter == "High" else "N"
        filtered = filtered[filtered["priority"] == pri_val]
    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]
    
    st.write(f"**Showing {len(filtered)} tasks**")
    
    # Display tasks
    for _, task in filtered.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            status_icon = "✅" if task["status"] == "Completed" else "⏳"
            priority_icon = "⚡" if task["priority"] == "Y" else ""
            
            col1.write(f"{status_icon} {priority_icon} **{task['task']}**")
            col1.caption(f"{CATEGORIES[task['category']]['icon']} {task['category']} | AI: {task['ai_prediction']:.0f}%")
            
            col2.write(time_left_str(task['deadline']))
            
            if task["status"] == "Pending":
                if col3.button("✅ Done", key=f"complete_{task['id']}"):
                    complete_task(int(task['id']))
            
            if col4.button("🗑️", key=f"delete_{task['id']}"):
                delete_task(int(task['id']))
            
            st.divider()

def complete_task(task_id):
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "status"] = "Completed"
    st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "completed_at"] = pd.Timestamp.now()
    save_data()
    st.success("Task completed!")
    time.sleep(0.5)
    st.rerun()

def delete_task(task_id):
    st.session_state.df_tasks = st.session_state.df_tasks[st.session_state.df_tasks["id"] != task_id]
    save_data()
    st.success("Task deleted!")
    time.sleep(0.5)
    st.rerun()

# ---------- HABITS ----------
def show_habits():
    st.title("🔥 Habits")
    
    tab1, tab2 = st.tabs(["Active Habits", "Create Habit"])
    
    with tab1:
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        
        if active_habits.empty:
            st.info("No habits yet. Create your first habit!")
        else:
            for _, habit in active_habits.iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    col1.write(f"### {habit['habit_name']}")
                    col1.caption(f"{CATEGORIES[habit['category']]['icon']} {habit['category']} | {habit['recurrence'].title()}")
                    col2.metric("🔥 Streak", f"{streak} days")
                    col3.metric("Total", habit['total_completions'])
                    st.divider()
    
    with tab2:
        with st.form("create_habit"):
            habit_name = st.text_input("Habit Name", placeholder="Exercise daily")
            
            col1, col2 = st.columns(2)
            with col1:
                recurrence = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
            with col2:
                category = st.selectbox("Category", list(CATEGORIES.keys()))
            
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
                    st.session_state.habit_id_counter += 1
                    
                    # Create first task
                    create_task_from_habit(st.session_state.habit_id_counter - 1, habit_name, recurrence, category)
                    
                    save_data()
                    st.success("Habit created!")
                    time.sleep(1)
                    st.rerun()

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

# ---------- TEAM ----------
def show_team():
    st.title("👥 Team Management")
    
    tab1, tab2 = st.tabs(["Team Members", "Add Member"])
    
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
                
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.write(f"### {user['username']}")
                col1.caption(f"Role: {user['role']}")
                col2.metric("Tasks", total)
                col3.metric("Done", completed)
                st.divider()
    
    with tab2:
        with st.form("add_member"):
            username = st.text_input("Username")
            role = st.selectbox("Role", ["member", "manager"])
            
            if st.form_submit_button("➕ Add Member"):
                if username.strip() and username not in st.session_state.df_users["username"].values:
                    new_user = pd.DataFrame([{
                        "username": username,
                        "role": role,
                        "added_at": pd.Timestamp.now(),
                        "added_by": st.session_state.current_user
                    }])
                    
                    st.session_state.df_users = pd.concat([st.session_state.df_users, new_user], ignore_index=True)
                    save_data()
                    st.success(f"Added {username}")
                    time.sleep(1)
                    st.rerun()

# ---------- ANALYTICS ----------
def show_analytics():
    st.title("📊 Analytics")
    
    my_tasks = st.session_state.df_tasks[
        st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
    ]
    
    if my_tasks.empty:
        st.info("No data yet")
        return
    
    # Category breakdown
    st.subheader("By Category")
    category_stats = my_tasks.groupby("category").agg({
        "status": ["count", lambda x: (x == "Completed").sum()]
    }).reset_index()
    category_stats.columns = ["Category", "Total", "Completed"]
    category_stats["Rate %"] = (category_stats["Completed"] / category_stats["Total"] * 100).round(1)
    
    st.dataframe(category_stats, use_container_width=True)
    
    # Weekly trend
    st.subheader("Weekly Activity")
    completed_tasks = my_tasks[my_tasks["status"] == "Completed"].copy()
    if not completed_tasks.empty:
        completed_tasks["date"] = completed_tasks["completed_at"].dt.date
        daily_counts = completed_tasks.groupby("date").size().reset_index(name="count")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_counts["date"], daily_counts["count"], marker='o', color='#e91e63')
        ax.set_xlabel("Date")
        ax.set_ylabel("Tasks Completed")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ---------- SETTINGS ----------
def show_settings():
    st.title("⚙️ Settings")
    
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export as CSV", use_container_width=True):
            my_tasks = st.session_state.df_tasks[
                st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
            ]
            csv = my_tasks.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"assan_tasks_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export as Excel", use_container_width=True):
            my_tasks = st.session_state.df_tasks[
                st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
            ]
            # Note: Excel export requires openpyxl
            st.info("Excel export requires openpyxl. Use CSV for now.")
    
    st.divider()
    
    st.subheader("User Information")
    st.write(f"**Username:** {st.session_state.current_user}")
    my_tasks = st.session_state.df_tasks[
        st.session_state.df_tasks["assigned_to"] == st.session_state.current_user
    ]
    st.write(f"**Total Tasks:** {len(my_tasks)}")
    st.write(f"**Completed:** {len(my_tasks[my_tasks['status'] == 'Completed'])}")
    st.write(f"**Active Habits:** {len(st.session_state.df_habits[st.session_state.df_habits['active'] == True])}")

# ---------- MAIN ----------
def main():
    # Load data on first run
    if st.session_state.df_tasks.empty:
        load_data()
    
    # Show login or main app
    if st.session_state.current_user is None:
        show_login()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
