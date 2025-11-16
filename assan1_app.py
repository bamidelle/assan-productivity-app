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
        <div style='text-align: center; padding: 4rem 2rem;'>
            <div style='font-size: 5rem; margin-bottom: 1rem;'>🎯</div>
            <h1 style='font-size: 48px; margin-bottom: 0.5rem;'>ASSAN</h1>
            <p style='font-size: 20px; color: #C9C9D1; font-weight: 500; margin-bottom: 0.5rem;'>Productivity Studio</p>
            <p style='font-size: 16px; color: #C9C9D1;'>32 Powerful Features • AI-Powered • Team Collaboration</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card' style='padding: 2.5rem;'>", unsafe_allow_html=True)
        st.markdown("### 👋 Welcome Back")
        username = st.text_input("Enter your name", key="login_username", placeholder="Your name...")
        
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            if username.strip():
                st.session_state.current_user = username.strip()
                
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
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- MAIN APP ----------
def show_main_app():
    # Navbar
    st.markdown(f"""
        <div class='navbar'>
            <div class='navbar-title'>
                <span style='font-size: 32px;'>🎯</span>
                <span>ASSAN</span>
                <span style='font-size: 16px; color: #C9C9D1; font-weight: 400; margin-left: 1rem;'>
                    Welcome, {st.session_state.current_user}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
            <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 1.5rem; 
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👤</div>
                <div style='font-size: 20px; font-weight: 700; color: #F4F4F9;'>{st.session_state.current_user}</div>
            </div>
        """, unsafe_allow_html=True)
        
        my_tasks = get_my_tasks()
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        
        col1, col2 = st.columns(2)
        col1.metric("Tasks", total)
        col2.metric("Done", completed)
        
        if total > 0:
            completion_rate = (completed / total * 100)
            st.progress(completion_rate / 100)
            st.caption(f"Progress: {completion_rate:.1f}%")
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        menu = st.radio("Navigation", [
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
        
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_user = None
            st.rerun()
    
    # Content Router
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

# ---------- DASHBOARD ----------
def show_dashboard():
    st.title("🏠 Dashboard")
    
    my_tasks = get_my_tasks()
    
    col1, col2, col3, col4 = st.columns(4)
    total = len(my_tasks)
    completed = len(my_tasks[my_tasks["status"] == "Completed"])
    pending = len(my_tasks[my_tasks["status"] == "Pending"])
    high_priority = len(my_tasks[my_tasks["priority"] == "Y"])
    
    col1.metric("📋 Total Tasks", total)
    col2.metric("✅ Completed", completed)
    col3.metric("⏳ Pending", pending)
    col4.metric("⚡ High Priority", high_priority)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    show_reminders_section()
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Recent Tasks")
        recent = my_tasks.sort_values("created_at", ascending=False).head(5)
        if not recent.empty:
            for _, task in recent.iterrows():
                status_icon = "✅" if task["status"] == "Completed" else "⏳"
                st.markdown(f"""
                <div class='task-card'>
                    {status_icon} <strong>{task['task']}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No tasks yet")
    
    with col2:
        st.markdown("### 🔥 Habit Streaks")
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        if not active_habits.empty:
            for _, habit in active_habits.head(5).iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                st.markdown(f"""
                <div class='task-card'>
                    🔥 <strong>{habit['habit_name']}</strong>: {streak} day streak
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active habits")

# ---------- ADD TASK ----------
def show_add_task():
    st.title("➕ Add New Task")
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    with st.form("add_task_form"):
        task_name = st.text_input("Task Name", placeholder="Enter task description...")
        
        col1, col2 = st.columns(2)
        with col1:
            priority = st.selectbox("Priority", ["Normal", "High"])
        with col2:
            category = st.selectbox("Category", list(CATEGORIES.keys()))
        
        col1, col2 = st.columns(2)
        with col1:
            deadline_date = st.date_input("Deadline (Optional)", value=None)
        with col2:
            deadline_time = st.time_input("Time", value=datetime.now().time())
        
        tags = st.text_input("Tags (comma separated)", placeholder="urgent, important")
        
        submitted = st.form_submit_button("➕ Add Task", type="primary", use_container_width=True)
        
        if submitted:
            if task_name.strip():
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
                
                st.success(f"✅ Task added! AI Prediction: {round(prob * 100, 2)}%")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter a task name")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- VIEW TASKS ----------
def show_view_tasks():
    st.title("📝 My Tasks")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("📭 No tasks yet. Create your first task!")
        return
    
    st.write(f"**Total: {len(my_tasks)} tasks**")
    
    for _, task in my_tasks.iterrows():
        status_icon = "✅" if task["status"] == "Completed" else "⏳"
        priority_icon = "⚡" if task["priority"] == "Y" else ""
        
        st.markdown(f"""
        <div class='task-card'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <strong>#{int(task['id'])}</strong> {status_icon} {priority_icon} <strong>{task['task']}</strong>
                    <br>
                    <span style='color: #C9C9D1; font-size: 15px;'>
                        {CATEGORIES[task['category']]['icon']} {task['category']} | 
                        AI: {task['ai_prediction']:.0f}% | 
                        {task['created_at'].strftime('%Y-%m-%d')}
                    </span>
                </div>
                <div style='text-align: right;'>
                    <strong>{time_left_str(task['deadline'])}</strong><br>
                    <span style='font-size: 15px;'>{task['status']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- REMOVE TASK ----------
def show_remove_task():
    st.title("🗑️ Remove Task")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No tasks to remove")
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- COMPLETE TASK ----------
def show_complete_task():
    st.title("✅ Complete Task")
    
    my_tasks = get_my_tasks()
    pending = my_tasks[my_tasks["status"] == "Pending"]
    
    if pending.empty:
        st.success("🎉 All tasks completed!")
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in pending.iterrows()}
    selected = st.selectbox("Select task to complete:", list(task_options.keys()))
    
    if selected:
        task_id = task_options[selected]
        task = pending[pending["id"] == task_id].iloc[0]
        
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 1.5rem; margin: 1rem 0;'>
            <h3>{task['task']}</h3>
            <p><strong>Category:</strong> {CATEGORIES[task['category']]['icon']} {task['category']}</p>
            <p><strong>Priority:</strong> {'⚡ High' if task['priority'] == 'Y' else '📍 Normal'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Mark as Completed", type="primary", use_container_width=True):
            complete_task(task_id)
            
            if not pd.isna(task["habit_id"]):
                streak = calculate_streak(int(task["habit_id"]))
                st.success(f"🎉 Task completed! 🔥 {streak} day streak!")
            else:
                st.success("🎉 Task completed!")
            
            time.sleep(1)
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- EDIT TASK ----------
def show_edit_task():
    st.title("✏️ Edit Task")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No tasks to edit")
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FILTER TASKS ----------
def show_filter_tasks():
    st.title("🔍 Filter & Search Tasks")
    
    my_tasks = get_my_tasks()
    
    if my_tasks.empty:
        st.info("No tasks to filter")
        return
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Completed"])
    with col2:
        priority_filter = st.selectbox("Priority", ["All", "High", "Normal"])
    with col3:
        category_filter = st.selectbox("Category", ["All"] + list(CATEGORIES.keys()))
    
    tag_search = st.text_input("Search by Tag", placeholder="Enter tag...")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
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
    
    for _, task in filtered.iterrows():
        status_icon = "✅" if task["status"] == "Completed" else "⏳"
        priority_icon = "⚡" if task["priority"] == "Y" else ""
        
        st.markdown(f"""
        <div class='task-card'>
            <strong>#{int(task['id'])}</strong> {status_icon} {priority_icon} {task['task']} 
            <span style='float: right;'>{time_left_str(task['deadline'])} | {task['status']}</span>
        </div>
        """, unsafe_allow_html=True)

# ---------- HABITS ----------
def show_habits():
    st.title("🔥 Habits")
    
    tab1, tab2, tab3 = st.tabs(["Active Habits", "Create Habit", "Habit Dashboard"])
    
    with tab1:
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        
        if active_habits.empty:
            st.info("No habits yet. Create your first habit!")
        else:
            for _, habit in active_habits.iterrows():
                streak = calculate_streak(int(habit["habit_id"]))
                
                with st.expander(f"🔥 {habit['habit_name']} - {streak} day streak"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Frequency", habit['recurrence'].title())
                    col2.metric("Streak", f"{streak} days")
                    col3.metric("Total", int(habit['total_completions']))
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        with st.form("create_habit"):
            habit_name = st.text_input("Habit Name", placeholder="Exercise daily...")
            
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
                    create_task_from_habit(st.session_state.habit_id_counter, habit_name, recurrence, category)
                    st.session_state.habit_id_counter += 1
                    save_data()
                    
                    st.success("✅ Habit created!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🏆 Habit Leaderboard")
        
        active_habits = st.session_state.df_habits[st.session_state.df_habits["active"] == True]
        
        if not active_habits.empty:
            for i, habit in enumerate(active_habits.iterrows()):
                _, h = habit
                streak = calculate_streak(int(h["habit_id"]))
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                st.markdown(f"""
                <div class='card'>
                    {medal} <strong>{h['habit_name']}</strong>: {streak} day streak ({int(h['total_completions'])} total)
                </div>
                """, unsafe_allow_html=True)

# ---------- TEAM ----------
def show_team():
    st.title("👥 Team Management")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Members", "Add Member", "Assign Task", "Comments", "Dashboard"])
    
    with tab1:
        if st.session_state.df_users.empty:
            st.info("No team members yet")
        else:
            for _, user in st.session_state.df_users.iterrows():
                user_tasks = st.session_state.df_tasks[st.session_state.df_tasks["assigned_to"] == user["username"]]
                total = len(user_tasks)
                completed = len(user_tasks[user_tasks["status"] == "Completed"])
                
                with st.expander(f"👤 {user['username']} ({user['role']})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tasks", total)
                    col2.metric("Completed", completed)
                    if total > 0:
                        col3.metric("Rate", f"{(completed/total*100):.1f}%")
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        with st.form("add_member"):
            username = st.text_input("Username")
            role = st.selectbox("Role", ["member", "manager"])
            
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
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        my_tasks = get_my_tasks()
        
        if my_tasks.empty:
            st.info("No tasks to assign")
        elif len(st.session_state.df_users) <= 1:
            st.info("Add team members first")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in my_tasks.iterrows()}
            selected_task = st.selectbox("Select task:", list(task_options.keys()))
            
            user_options = st.session_state.df_users["username"].tolist()
            selected_user = st.selectbox("Assign to:", user_options)
            
            if st.button("📤 Assign Task", type="primary"):
                task_id = task_options[selected_task]
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "assigned_to"] = selected_user
                st.session_state.df_tasks.loc[st.session_state.df_tasks["id"] == task_id, "shared"] = True
                save_data()
                st.success(f"✅ Assigned to {selected_user}")
                time.sleep(1)
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        my_tasks = get_my_tasks()
        
        if my_tasks.empty:
            st.info("No tasks")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            task_options = {f"#{int(row['id'])} - {row['task']}": int(row['id']) for _, row in my_tasks.iterrows()}
            selected_task = st.selectbox("Task:", list(task_options.keys()))
            
            task_id = task_options[selected_task]
            task_comments = st.session_state.df_comments[st.session_state.df_comments["task_id"] == task_id]
            
            if not task_comments.empty:
                for _, comment in task_comments.iterrows():
                    st.info(f"👤 **{comment['username']}** ({comment['timestamp'].strftime('%Y-%m-%d %H:%M')})\n\n{comment['comment']}")
            
            with st.form("add_comment"):
                new_comment = st.text_area("Add comment:")
                
                if st.form_submit_button("💬 Add Comment"):
                    if new_comment.strip():
                        new_row = pd.DataFrame([{
                            "comment_id": st.session_state.comment_id_counter,
                            "task_id": task_id,
                            "username": st.session_state.current_user,
                            "comment": new_comment,
                            "timestamp": pd.Timestamp.now()
                        }])
                        
                        st.session_state.df_comments = pd.concat([st.session_state.df_comments, new_row], ignore_index=True)
                        st.session_state.comment_id_counter += 1
                        save_data()
                        st.success("✅ Comment added!")
                        time.sleep(1)
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab5:
        if len(st.session_state.df_users) <= 1:
            st.info("Add team members")# ==============================================
# ASSAN - FIREBASE STUDIO DARK THEME
# Complete Productivity App with Modern UI
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
    page_title="Assan - Productivity Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- FIREBASE STUDIO DARK THEME CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* === GLOBAL STYLES === */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* === MAIN BACKGROUND - Firebase Studio Dark Gradient === */
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #15161F 40%, #1E1F2B 100%) !important;
        background-attachment: fixed;
    }
    
    /* === MAIN CONTENT CONTAINER === */
    .main .block-container {
        background: transparent;
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] * {
        color: #F4F4F9 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #C9C9D1 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* === HEADINGS === */
    h1 {
        color: #F4F4F9 !important;
        font-weight: 800 !important;
        font-size: 32px !important;
        margin-bottom: 2rem !important;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(76, 110, 245, 0.3);
    }
    
    h2 {
        color: #F4F4F9 !important;
        font-weight: 700 !important;
        font-size: 22px !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 20px !important;
    }
    
    /* === TEXT COLORS === */
    p, span, div, label {
        color: #FFFFFF !important;
        font-size: 16px;
    }
    
    .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* === BUTTONS - Firebase Studio Style === */
    .stButton > button {
        background: linear-gradient(90deg, #4C6EF5, #5C7CFA) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(76, 110, 245, 0.3) !important;
        text-transform: none !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5C7CFA, #6C8CFB) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(76, 110, 245, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* === FORM SUBMIT BUTTONS === */
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #4C6EF5, #5C7CFA) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 20px rgba(76, 110, 245, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #5C7CFA, #6C8CFB) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(76, 110, 245, 0.5) !important;
    }
    
    /* === INPUT FIELDS === */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background: #1C1C28 !important;
        border: 1px solid #2A2A3A !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        padding: 0.75rem 1rem !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #4C6EF5 !important;
        box-shadow: 0 0 0 3px rgba(76, 110, 245, 0.2) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #C9C9D1 !important;
    }
    
    /* === SELECTBOX === */
    [data-baseweb="select"] {
        background: #1C1C28 !important;
        border-radius: 10px !important;
    }
    
    [data-baseweb="select"] > div {
        background: #1C1C28 !important;
        border: 1px solid #2A2A3A !important;
        color: #FFFFFF !important;
    }
    
    /* === LABELS === */
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label,
    .stDateInput > label,
    .stTimeInput > label {
        color: #C9C9D1 !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* === CARDS / CONTAINERS === */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* === TASK CARDS === */
    .task-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .task-card:hover {
        background: rgba(76, 110, 245, 0.1);
        border-color: rgba(76, 110, 245, 0.3);
        transform: translateX(4px);
    }
    
    /* === METRICS === */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #C9C9D1 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 16px;
        color: #C9C9D1;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4C6EF5, #5C7CFA);
        color: #FFFFFF !important;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        padding: 1rem !important;
        font-size: 16px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(76, 110, 245, 0.3) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0 0 12px 12px;
        padding: 1rem;
    }
    
    /* === DIVIDER === */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
    }
    
    /* === ALERTS === */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        border-left-width: 4px !important;
        padding: 1rem 1.25rem !important;
        font-weight: 500 !important;
        color: #FFFFFF !important;
    }
    
    /* === SUCCESS ALERT === */
    .stSuccess {
        background: rgba(72, 187, 120, 0.1) !important;
        border-left-color: #48BB78 !important;
    }
    
    /* === ERROR ALERT === */
    .stError {
        background: rgba(245, 101, 101, 0.1) !important;
        border-left-color: #F56565 !important;
    }
    
    /* === WARNING ALERT === */
    .stWarning {
        background: rgba(237, 137, 54, 0.1) !important;
        border-left-color: #ED8936 !important;
    }
    
    /* === INFO ALERT === */
    .stInfo {
        background: rgba(76, 110, 245, 0.1) !important;
        border-left-color: #4C6EF5 !important;
    }
    
    /* === DATAFRAME === */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: rgba(76, 110, 245, 0.2) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }
    
    .dataframe td {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #FFFFFF !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* === PROGRESS BAR === */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4C6EF5, #5C7CFA) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* === RADIO BUTTONS === */
    .stRadio > label {
        font-weight: 600 !important;
        color: #F4F4F9 !important;
        font-size: 16px !important;
    }
    
    .stRadio > div {
        background: transparent !important;
    }
    
    .stRadio [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        transition: all 0.3s ease !important;
        color: #FFFFFF !important;
    }
    
    .stRadio [role="radiogroup"] label:hover {
        background: rgba(76, 110, 245, 0.1) !important;
        border-color: rgba(76, 110, 245, 0.3) !important;
    }
    
    /* === DOWNLOAD BUTTON === */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #48BB78, #38A169) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 20px rgba(72, 187, 120, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #38A169, #2F855A) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(72, 187, 120, 0.5) !important;
    }
    
    /* === CAPTION TEXT === */
    .caption {
        color: #C9C9D1 !important;
        font-size: 15px !important;
    }
    
    /* === SPINNER === */
    .stSpinner > div {
        border-top-color: #4C6EF5 !important;
    }
    
    /* === NAVBAR STYLE === */
    .navbar {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem 3rem;
        margin: -2rem -3rem 2rem -3rem;
        border-radius: 0;
    }
    
    .navbar-title {
        font-size: 28px;
        font-weight: 800;
        color: #F4F4F9;
        text-shadow: 0 0 30px rgba(76, 110, 245, 0.4);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* === SECTION DIVIDER === */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(76, 110, 245, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* === SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(76, 110, 245, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(76, 110, 245, 0.7);
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
    "Work": {"icon": "🏢", "color": "#4C6EF5"},
    "Personal": {"icon": "🏠", "color": "#48BB78"},
    "Learning": {"icon": "🎓", "color": "#9F7AEA"},
    "Health": {"icon": "💪", "color": "#F56565"},
    "Creative": {"icon": "🎨", "color": "#ED8936"},
    "Other": {"icon": "📌", "color": "#5C7CFA"}
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
    
    new_task = pd.DataFrame([{
        "id": st.session_state.task_id_counter,
        "task": name,
        "priority": "Y",
        "status": "Pending",
        "created_at
