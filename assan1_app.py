# ==============================================
# COMPLETE PRODUCTIVITY APP - LESSONS 15-21 (MERGED)
# Full-Featured Multi-User Task Management System
# Includes: Reminders, Habits, Team, NN training, Dashboard & Analytics
# ==============================================
import os
import sys
import time
import threading
import queue
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------- FILES ----------
USER_FILE = "username.txt"
DATA_FILE = "tasks.csv"
REMINDER_LOG = "reminders.csv"
HABITS_FILE = "habits.csv"
USERS_FILE = "users.csv"
COMMENTS_FILE = "comments.csv"
MODEL_FILE = "model.h5"
SCALER_FILE = "scaler.pkl"

# ---------- COLORS ----------
MAGENTA = "\033[95m"
ORANGE = "\033[38;5;208m"
RED = "\033[91m"
DEEP_RED = "\033[38;5;160m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
PURPLE = "\033[35m"
RESET = "\033[0m"
BOLD = "\033[1m"
BLINK = "\033[5m"

# ---------- CATEGORIES ----------
CATEGORIES = {
    "Work": {"icon": "🏢", "color": BLUE},
    "Personal": {"icon": "🏠", "color": GREEN},
    "Learning": {"icon": "🎓", "color": PURPLE},
    "Health": {"icon": "💪", "color": RED},
    "Creative": {"icon": "🎨", "color": MAGENTA},
    "Other": {"icon": "📌", "color": CYAN}
}

RECURRENCE_TYPES = {
    "daily": {"name": "Daily", "icon": "📅"},
    "weekly": {"name": "Weekly", "icon": "📆"},
    "monthly": {"name": "Monthly", "icon": "🗓️"},
    "none": {"name": "One-time", "icon": "📌"}
}

# ---------- GLOBALS ----------
reminder_queue = queue.Queue()
reminder_thread = None
stop_reminder_thread = threading.Event()
shown_reminders = set()

# ---------- UTILITY FUNCTIONS ----------
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

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
        except:
            continue
    parsed = pd.to_datetime(s, errors='coerce')
    return parsed if not pd.isna(parsed) else pd.NaT

def time_left_parts(deadline_ts):
    if pd.isna(deadline_ts):
        return ("No deadline", None)
    now = now_dt()
    try:
        dl = deadline_ts.to_pydatetime() if isinstance(deadline_ts, pd.Timestamp) else pd.to_datetime(deadline_ts).to_pydatetime()
    except:
        dl = pd.to_datetime(deadline_ts, errors='coerce')
        if pd.isna(dl):
            return ("Invalid", None)
        dl = dl.to_pydatetime()
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
    if not tags_list or len(tags_list) == 0:
        return ""
    return CYAN + " [" + ", ".join(tags_list) + "]" + RESET

def get_category_display(category):
    if category not in CATEGORIES:
        category = "Other"
    cat_info = CATEGORIES[category]
    return f"{cat_info['color']}{cat_info['icon']} {category}{RESET}"

def get_user_color(username):
    colors = [CYAN, GREEN, YELLOW, MAGENTA, BLUE, PURPLE]
    hash_val = sum(ord(c) for c in str(username))
    return colors[hash_val % len(colors)]

# ---------- LOAD USER ----------
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        current_user = f.read().strip()
else:
    current_user = input("Enter your name: ").strip()
    with open(USER_FILE, "w") as f:
        f.write(current_user)

# ---------- LOAD DATA ----------
if os.path.exists(USERS_FILE):
    df_users = pd.read_csv(USERS_FILE)
    if "added_at" in df_users.columns:
        df_users["added_at"] = pd.to_datetime(df_users["added_at"], errors='coerce')
else:
    df_users = pd.DataFrame(columns=["username", "role", "added_at", "added_by"])
    df_users = pd.concat([df_users, pd.DataFrame([{
        "username": current_user, "role": "owner", "added_at": pd.Timestamp.now(), "added_by": "self"
    }])], ignore_index=True)
    df_users.to_csv(USERS_FILE, index=False)

if current_user not in df_users["username"].values:
    df_users = pd.concat([df_users, pd.DataFrame([{
        "username": current_user, "role": "owner", "added_at": pd.Timestamp.now(), "added_by": "self"
    }])], ignore_index=True)
    df_users.to_csv(USERS_FILE, index=False)

if os.path.exists(DATA_FILE):
    df_tasks = pd.read_csv(DATA_FILE, low_memory=False)
    for c in ["created_at", "completed_at", "deadline"]:
        if c in df_tasks.columns:
            df_tasks[c] = pd.to_datetime(df_tasks[c], errors='coerce')
    # ensure columns
    defaults = {
        "category": "Other", "tags": "", "ai_prediction": np.nan,
        "habit_id": np.nan, "recurrence": "none", "assigned_to": current_user,
        "created_by": current_user, "shared": False, "deadline": pd.NaT, "estimated_minutes": 30
    }
    for col, default in defaults.items():
        if col not in df_tasks.columns:
            df_tasks[col] = default
else:
    df_tasks = pd.DataFrame(columns=["id","task","priority","status","created_at","completed_at",
        "deadline","ai_prediction","category","tags","habit_id","recurrence","assigned_to","created_by","shared","estimated_minutes"])

df_tasks = df_tasks.sort_values("id") if not df_tasks.empty else df_tasks
task_id_counter = int(df_tasks["id"].max()) + 1 if not df_tasks.empty else 1

if os.path.exists(HABITS_FILE):
    df_habits = pd.read_csv(HABITS_FILE)
    for c in ["created_at", "last_completed"]:
        if c in df_habits.columns:
            df_habits[c] = pd.to_datetime(df_habits[c], errors='coerce')
else:
    df_habits = pd.DataFrame(columns=["habit_id","habit_name","recurrence","category","active","created_at","last_completed","total_completions"])
habit_id_counter = int(df_habits["habit_id"].max()) + 1 if not df_habits.empty else 1

if os.path.exists(COMMENTS_FILE):
    df_comments = pd.read_csv(COMMENTS_FILE)
    if "timestamp" in df_comments.columns:
        df_comments["timestamp"] = pd.to_datetime(df_comments["timestamp"], errors='coerce')
else:
    df_comments = pd.DataFrame(columns=["comment_id", "task_id", "username", "comment", "timestamp"])
comment_id_counter = int(df_comments["comment_id"].max()) + 1 if not df_comments.empty else 1

if os.path.exists(REMINDER_LOG):
    df_reminders = pd.read_csv(REMINDER_LOG)
    if "timestamp" in df_reminders.columns:
        df_reminders["timestamp"] = pd.to_datetime(df_reminders["timestamp"], errors='coerce')
else:
    df_reminders = pd.DataFrame(columns=["task_id", "task_name", "alert_type", "timestamp"])

def log_reminder(task_id, task_name, alert_type):
    global df_reminders
    df_reminders = pd.concat([df_reminders, pd.DataFrame([{
        "task_id": task_id, "task_name": task_name, "alert_type": alert_type, "timestamp": pd.Timestamp.now()
    }])], ignore_index=True)
    df_reminders.to_csv(REMINDER_LOG, index=False)

# ---------- REMINDERS ----------
def check_reminders_background():
    global df_tasks, reminder_queue, shown_reminders
    while not stop_reminder_thread.is_set():
        try:
            if not df_tasks.empty:
                pending = df_tasks[(df_tasks["status"] == "Pending") & (df_tasks["assigned_to"] == current_user)].copy()
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
                    if reminder_key and reminder_key not in shown_reminders:
                        reminder_queue.put({"task_id": task_id, "task_name": task["task"], "alert_type": alert_type, "time_left": secs})
                        shown_reminders.add(reminder_key)
                        log_reminder(task_id, task["task"], alert_type)
            time.sleep(30)
        except:
            time.sleep(30)

def start_reminder_system():
    global reminder_thread
    if reminder_thread is None or not reminder_thread.is_alive():
        stop_reminder_thread.clear()
        reminder_thread = threading.Thread(target=check_reminders_background, daemon=True)
        reminder_thread.start()

def stop_reminder_system():
    stop_reminder_thread.set()
    if reminder_thread:
        reminder_thread.join(timeout=1)

def show_pending_reminders():
    count = reminder_queue.qsize()
    if count == 0:
        print(f"{GREEN}✅ No pending reminders!{RESET}")
        return
    print(f"\n{ORANGE}{'='*50}{RESET}\n{BOLD}{ORANGE}🔔 {count} ACTIVE REMINDER(S):{RESET}\n{ORANGE}{'='*50}{RESET}\n")
    temp = []
    while not reminder_queue.empty():
        temp.append(reminder_queue.get())
    for rem in temp:
        tid, name, atype = rem["task_id"], rem["task_name"], rem["alert_type"]
        if atype == "overdue":
            print(f"{DEEP_RED}{BLINK}🔴 OVERDUE:{RESET} {DEEP_RED}Task #{tid}: {name}{RESET}")
        elif atype == "urgent":
            print(f"{DEEP_RED}🟠 URGENT (15 min):{RESET} {DEEP_RED}Task #{tid}: {name}{RESET}")
        elif atype == "warning":
            print(f"{YELLOW}🟡 WARNING (1 hour):{RESET} Task #{tid}: {name}")
    print(f"\n{ORANGE}{'='*50}{RESET}")
    for rem in temp:
        reminder_queue.put(rem)

# ---------- NN Helpers (existing) ----------
def prepare_data_for_nn(df):
    if df.empty or "status" not in df.columns:
        return None, None, None
    d = df.copy()
    d["priority_num"] = d["priority"].map({"Y":1, "N":0}).fillna(0).astype(int)
    d["task_len"] = d["task"].astype(str).apply(len)
    d["days_to_deadline"] = (pd.to_datetime(d["deadline"], errors='coerce') - pd.Timestamp.now()).dt.total_seconds() / 86400
    d["days_to_deadline"] = d["days_to_deadline"].fillna(0)
    d["completed"] = (d["status"] == "Completed").astype(int)
    cats = pd.get_dummies(d["category"], prefix="cat")
    X = pd.concat([d[["priority_num", "task_len", "days_to_deadline"]], cats], axis=1)
    y = d["completed"]
    if len(y.unique()) < 2:
        return None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

def build_nn_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def tqdm_callback():
    class TqdmCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\rEpoch {epoch+1}/10", end='')
    return TqdmCallback()

def plot_model_performance(history):
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['accuracy'], label='Accuracy', color='#00FF00')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#00FFFF')
        plt.plot(history.history['loss'], label='Loss', color='#FF0000')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='#FF9999')
        plt.title("Neural Network Training Performance")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

def train_nn_model():
    global df_tasks
    X, y, scaler = prepare_data_for_nn(df_tasks)
    if X is None:
        print(f"{RED}❌ Not enough varied task data to train neural network (need completed and pending tasks){RESET}")
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_nn_model(X.shape[1])
    print(f"\n{BOLD}{MAGENTA}Training Neural Network...{RESET}")
    history = model.fit(
        X_train, y_train, epochs=10, batch_size=32,
        validation_data=(X_test, y_test), verbose=0,
        callbacks=[tqdm_callback()]
    )
    model.save(MODEL_FILE)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n{GREEN}✅ Neural network trained and saved! Accuracy: {history.history['accuracy'][-1]:.2%}{RESET}")
    plot_model_performance(history)
    return model, scaler

def predict_nn_prob(model, scaler, row):
    if model is None or scaler is None:
        return 0.8 if str(row.get("priority", "N")).upper() == "Y" else 0.3
    d = pd.DataFrame([{
        "priority_num": 1 if str(row.get("priority", "N")).upper() == "Y" else 0,
        "task_len": len(str(row.get("task", ""))),
        "days_to_deadline": (parse_deadline_input(row.get("deadline", "")) - pd.Timestamp.now()).total_seconds() / 86400 if row.get("deadline") else 0,
        "category": row.get("category", "Other")
    }])
    cats = pd.get_dummies(d["category"], prefix="cat")
    # ensure all categories present
    for c in CATEGORIES:
        col = f"cat_{c}"
        if col not in cats:
            cats[col] = 0
    d = pd.concat([d[["priority_num", "task_len", "days_to_deadline"]], cats[[f"cat_{c}" for c in CATEGORIES]]], axis=1)
    X = scaler.transform(d)
    return float(model.predict(X, verbose=0)[0][0])

# ---------- TASK MANAGEMENT (existing functions) ----------
def add_task():
    global df_tasks, task_id_counter
    model = tf.keras.models.load_model(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    scaler = pickle.load(open(SCALER_FILE, 'rb')) if os.path.exists(SCALER_FILE) else None
    while True:
        print(f"\n{BOLD}{MAGENTA}Add Task (type 'Home' to return to menu){RESET}\n{'─'*50}")
        name = input("📝 Task: ").strip()
        if name.lower() == "home":
            print(f"\n{GREEN}Returning to main menu...{RESET}")
            break
        if not name:
            print(f"{RED}❌ Empty task, please enter a task or type 'Home'{RESET}")
            continue
        priority = input("⚡ Priority (Y/N): ").strip().upper()
        if priority not in ["Y", "N"]:
            priority = "N"
        deadline = input("📅 Deadline (e.g., 2025-12-31 23:59, empty for none): ").strip()
        deadline_ts = parse_deadline_input(deadline)
        print("\n📂 Category:")
        cats = list(CATEGORIES.keys())
        for i, c in enumerate(cats, 1):
            print(f" {i}. {get_category_display(c)}")
        cc = input("Number: ").strip()
        category = cats[int(cc)-1] if cc.isdigit() and 1<=int(cc)<=len(cats) else "Other"
        tags = input("🏷️ Tags (comma-separated, e.g., urgent,work): ").strip()
        est = input("⏱ Estimated minutes (default 30): ").strip()
        try:
            est_min = int(est) if est else 30
        except:
            est_min = 30
        prob = predict_nn_prob(model, scaler, {"priority": priority, "task": name, "deadline": deadline_ts, "category": category})
        df_tasks = pd.concat([df_tasks, pd.DataFrame([{
            "id": task_id_counter, "task": name, "priority": priority, "status": "Pending",
            "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": deadline_ts,
            "ai_prediction": round(prob*100,2), "category": category, "tags": tags,
            "habit_id": np.nan, "recurrence": "none", "assigned_to": current_user,
            "created_by": current_user, "shared": False, "estimated_minutes": est_min
        }])], ignore_index=True)
        df_tasks.to_csv(DATA_FILE, index=False)
        task_id_counter += 1
        print(f"\n{GREEN}✅ Added '{name}'! AI: {round(prob*100,2)}% likely to complete{RESET}")
        # continue loop to allow multiple additions

def view_tasks():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{GREEN}✅ No tasks!{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}📋 Your Tasks{RESET}\n{'='*60}")
    for _, t in my.iterrows():
        tid = int(t["id"])
        stat = f"{GREEN}✅ Completed{RESET}" if t["status"] == "Completed" else f"{YELLOW}❌ Pending{RESET}"
        pri = "⚡ High" if t["priority"] == "Y" else "Regular"
        tleft, _ = time_left_parts(t["deadline"])
        dl = f"{DEEP_RED}{tleft}{RESET}" if "OVERDUE" in tleft or "DEEP" in tleft else (f"{RED}{tleft}{RESET}" if "OVERDUE" in tleft else tleft)
        tags = format_tags(parse_tags(t["tags"]))
        cat = get_category_display(t["category"])
        try:
            ai_val = float(t.get("ai_prediction", np.nan))
        except:
            ai_val = np.nan
        prob = f"{GREEN}✓ {ai_val}%{RESET}" if (not pd.isna(ai_val) and ai_val >= 50) else (f"{YELLOW}⚠️ {ai_val}%{RESET}" if not pd.isna(ai_val) else "")
        print(f"#{tid} {t['task']} {tags}\n {cat} | {pri} | {stat} | Deadline: {dl} | AI: {prob}")
    print("="*60)

def remove_task():
    global df_tasks
    view_tasks()
    if df_tasks.empty:
        return
    try:
        tid = int(input("\nTask ID to remove: ").strip())
        if tid not in df_tasks["id"].values:
            print(f"{RED}❌ Task not found{RESET}")
            return
        task_name = df_tasks[df_tasks["id"] == tid]["task"].iloc[0]
        df_tasks = df_tasks[df_tasks["id"] != tid]
        df_tasks.to_csv(DATA_FILE, index=False)
        print(f"\n{GREEN}✅ Removed '{task_name}'{RESET}")
    except:
        print(f"{RED}❌ Invalid ID{RESET}")

def mark_task_completed():
    global df_tasks, df_habits
    view_tasks()
    if df_tasks.empty:
        return
    try:
        tid = int(input("\nTask ID to complete: ").strip())
        if tid not in df_tasks["id"].values:
            print(f"{RED}❌ Task not found{RESET}")
            return
        if df_tasks[df_tasks["id"] == tid]["status"].iloc[0] == "Completed":
            print(f"{RED}❌ Already completed{RESET}")
            return
        df_tasks.loc[df_tasks["id"] == tid, "status"] = "Completed"
        df_tasks.loc[df_tasks["id"] == tid, "completed_at"] = pd.Timestamp.now()
        task = df_tasks[df_tasks["id"] == tid].iloc[0]
        if not pd.isna(task["habit_id"]):
            hid = int(task["habit_id"])
            if hid in df_habits["habit_id"].values:
                df_habits.loc[df_habits["habit_id"] == hid, "total_completions"] = df_habits.loc[df_habits["habit_id"] == hid, "total_completions"].fillna(0) + 1
                df_habits.loc[df_habits["habit_id"] == hid, "last_completed"] = pd.Timestamp.now()
                # create next instance from habit row
                habit_row = df_habits[df_habits["habit_id"]==hid].iloc[0]
                create_task_from_habit(hid, habit_row["habit_name"], habit_row["recurrence"], habit_row["category"])
        df_tasks.to_csv(DATA_FILE, index=False)
        df_habits.to_csv(HABITS_FILE, index=False)
        print(f"\n{GREEN}✅ Marked '{task['task']}' as completed!{RESET}")
    except Exception as e:
        print(f"{RED}❌ Invalid ID: {e}{RESET}")

def productivity_chart():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No tasks{RESET}")
        return
    summary = my.groupby("category").agg({
        "status": ["count", lambda x: (x == "Completed").sum()]
    }).reset_index()
    summary.columns = ["Category", "Total", "Completed"]
    summary["Pending"] = summary["Total"] - summary["Completed"]
    try:
        plt.figure(figsize=(8, 5))
        bar_width = 0.35
        x = np.arange(len(summary))
        plt.bar(x - bar_width/2, summary["Completed"], bar_width, label="Completed", color="#00FF00")
        plt.bar(x + bar_width/2, summary["Pending"], bar_width, label="Pending", color="#FF4444")
        plt.xlabel("Category")
        plt.ylabel("Tasks")
        plt.title("Productivity Chart")
        plt.xticks(x, summary["Category"])
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        print(summary)

def weekly_performance_summary():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No tasks{RESET}")
        return
    start = now_dt() - timedelta(days=7)
    week = my[(pd.to_datetime(my["created_at"], errors='coerce') >= start)]
    total = len(week)
    completed = len(week[week["status"] == "Completed"])
    rate = round((completed / total * 100) if total > 0 else 0, 1)
    print(f"\n{BOLD}{MAGENTA}📅 Weekly Summary{RESET}\n{'='*50}")
    print(f"Tasks: {total} | Completed: {GREEN}{completed}{RESET} | Rate: {rate}%")
    print("="*50)

def top_productive_days():
    my = df_tasks[(df_tasks["assigned_to"] == current_user) & (df_tasks["status"] == "Completed")]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No completed tasks{RESET}")
        return
    days = my.groupby(pd.to_datetime(my["completed_at"], errors='coerce').dt.date).size().reset_index(name="count")
    days = days.sort_values("count", ascending=False).head(5)
    print(f"\n{BOLD}{MAGENTA}🏆 Top Productive Days{RESET}\n{'='*50}")
    for _, d in days.iterrows():
        print(f"{d['completed_at']}: {GREEN}{d['count']} tasks{RESET}")
    print("="*50)

def edit_task():
    global df_tasks
    view_tasks()
    if df_tasks.empty:
        return
    try:
        tid = int(input("\nTask ID to edit: ").strip())
        if tid not in df_tasks["id"].values:
            print(f"{RED}❌ Task not found{RESET}")
            return
        task = df_tasks[df_tasks["id"] == tid].iloc[0]
        print(f"\nEditing '{task['task']}'")
        name = input(f"📝 New task (current: {task['task']}, Enter to keep): ").strip()
        priority = input(f"⚡ Priority (current: {task['priority']}, Y/N or Enter to keep): ").strip().upper()
        deadline = input(f"📅 Deadline (current: {task['deadline']}, e.g., 2025-12-31 23:59): ").strip()
        print("\n📂 Category:")
        cats = list(CATEGORIES.keys())
        for i, c in enumerate(cats, 1):
            print(f" {i}. {get_category_display(c)}")
        cc = input(f"Number (current: {task['category']}): ").strip()
        tags = input(f"🏷️ Tags (current: {task['tags']}): ").strip()
        if name:
            df_tasks.loc[df_tasks["id"] == tid, "task"] = name
        if priority in ["Y", "N"]:
            df_tasks.loc[df_tasks["id"] == tid, "priority"] = priority
        if deadline:
            df_tasks.loc[df_tasks["id"] == tid, "deadline"] = parse_deadline_input(deadline)
        if cc.isdigit() and 1 <= int(cc) <= len(cats):
            df_tasks.loc[df_tasks["id"] == tid, "category"] = cats[int(cc)-1]
        if tags:
            df_tasks.loc[df_tasks["id"] == tid, "tags"] = tags
        # update AI prediction
        model = tf.keras.models.load_model(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
        scaler = pickle.load(open(SCALER_FILE, 'rb')) if os.path.exists(SCALER_FILE) else None
        prob = predict_nn_prob(model, scaler, {
            "priority": df_tasks.loc[df_tasks["id"] == tid, "priority"].iloc[0],
            "task": df_tasks.loc[df_tasks["id"] == tid, "task"].iloc[0],
            "deadline": df_tasks.loc[df_tasks["id"] == tid, "deadline"].iloc[0],
            "category": df_tasks.loc[df_tasks["id"] == tid, "category"].iloc[0]
        })
        df_tasks.loc[df_tasks["id"] == tid, "ai_prediction"] = round(prob*100,2)
        df_tasks.to_csv(DATA_FILE, index=False)
        print(f"\n{GREEN}✅ Updated task!{RESET}")
    except Exception as e:
        print(f"{RED}❌ Invalid input: {e}{RESET}")

def filter_tasks():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{GREEN}✅ No tasks!{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}🔍 Filter Tasks{RESET}\n{'─'*50}")
    print("1. Pending 2. Completed 3. High Priority 4. Category")
    choice = input("Choose: ").strip()
    if choice == "1":
        filtered = my[my["status"] == "Pending"]
    elif choice == "2":
        filtered = my[my["status"] == "Completed"]
    elif choice == "3":
        filtered = my[my["priority"] == "Y"]
    elif choice == "4":
        print("\n📂 Category:")
        cats = list(CATEGORIES.keys())
        for i, c in enumerate(cats, 1):
            print(f" {i}. {get_category_display(c)}")
        cc = input("Number: ").strip()
        category = cats[int(cc)-1] if cc.isdigit() and 1<=int(cc)<=len(cats) else "Other"
        filtered = my[my["category"] == category]
    else:
        print(f"{RED}❌ Invalid choice{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}📋 Filtered Tasks{RESET}\n{'='*50}")
    for _, t in filtered.iterrows():
        tid = int(t["id"])
        stat = f"{GREEN}✅ Completed{RESET}" if t["status"] == "Completed" else f"{YELLOW}❌ Pending{RESET}"
        pri = "⚡ High" if t["priority"] == "Y" else "Regular"
        tleft, _ = time_left_parts(t["deadline"])
        dl = f"{DEEP_RED}{tleft}{RESET}" if "OVERDUE" in tleft else tleft
        tags = format_tags(parse_tags(t["tags"]))
        cat = get_category_display(t["category"])
        prob = f"{GREEN}✓ {t['ai_prediction']}%{RESET}" if t["ai_prediction"] >= 50 else f"{YELLOW}⚠️ {t['ai_prediction']}%{RESET}"
        print(f"#{tid} {t['task']} {tags}\n {cat} | {pri} | {stat} | Deadline: {dl} | AI: {prob}")
    print("="*50)

def productivity_trend():
    my = df_tasks[(df_tasks["assigned_to"] == current_user) & (df_tasks["status"] == "Completed")]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No completed tasks{RESET}")
        return
    days = my.groupby(pd.to_datetime(my["completed_at"], errors='coerce').dt.date).size().reset_index(name="count")
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(days["completed_at"], days["count"], marker='o', color="#00FF00")
        plt.xlabel("Date")
        plt.ylabel("Tasks Completed")
        plt.title("Productivity Trend")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        print(days)

def smart_productivity_trend():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No tasks{RESET}")
        return
    model = tf.keras.models.load_model(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    scaler = pickle.load(open(SCALER_FILE, 'rb')) if os.path.exists(SCALER_FILE) else None
    if model is None:
        print(f"\n{YELLOW}⚠️ Train neural network first (option 27){RESET}")
        return
    days = my.groupby(pd.to_datetime(my["created_at"], errors='coerce').dt.date).size().reset_index(name="total")
    comp = my[my["status"] == "Completed"].groupby(pd.to_datetime(my["completed_at"], errors='coerce').dt.date).size().reset_index(name="completed")
    merged = pd.merge(days, comp, left_on="created_at", right_on="completed_at", how="left").fillna(0)
    merged["rate"] = merged["completed"] / merged["total"] * 100
    prob = predict_nn_prob(model, scaler, {"priority": "Y", "task": "Sample", "deadline": "", "category": "Other"}) * 100
    print(f"\n{BOLD}{MAGENTA}📈 Smart Productivity Trend{RESET}\n{'='*50}")
    print(f"Recent Rate: {round(merged['rate'].mean(), 1)}%")
    print(f"AI Predicts: {GREEN}{round(prob, 1)}% completion for high-priority tasks{RESET}")
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(merged["created_at"], merged["rate"], marker='o', color="#00FF00")
        plt.xlabel("Date")
        plt.ylabel("Completion Rate (%)")
        plt.title("Smart Productivity Trend")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        print(merged)

def daily_ai_plan():
    my = df_tasks[(df_tasks["assigned_to"] == current_user) & (df_tasks["status"] == "Pending")]
    if my.empty:
        print(f"\n{GREEN}✅ No pending tasks!{RESET}")
        return
    model = tf.keras.models.load_model(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    scaler = pickle.load(open(SCALER_FILE, 'rb')) if os.path.exists(SCALER_FILE) else None
    if model and scaler:
        my = my.copy()
        my["ai_prediction"] = my.apply(lambda row: predict_nn_prob(model, scaler, {
            "priority": row["priority"],
            "task": row["task"],
            "deadline": row["deadline"],
            "category": row["category"]
        }) * 100, axis=1)
        df_tasks.update(my[["id","ai_prediction"]].set_index("id"))
        df_tasks.to_csv(DATA_FILE, index=False)
    my = my.sort_values("ai_prediction", ascending=False)
    print(f"\n{BOLD}{MAGENTA}📋 Daily AI Plan{RESET}\n{'='*50}")
    for _, t in my.head(5).iterrows():
        tid = int(t["id"])
        tleft, _ = time_left_parts(t["deadline"])
        dl = f"{DEEP_RED}{tleft}{RESET}" if "OVERDUE" in tleft else tleft
        tags = format_tags(parse_tags(t["tags"]))
        cat = get_category_display(t["category"])
        prob = f"{GREEN}✓ {round(t['ai_prediction'],2)}%{RESET}" if t["ai_prediction"] >= 50 else f"{YELLOW}⚠️ {round(t['ai_prediction'],2)}%{RESET}"
        print(f"#{tid} {t['task']} {tags}\n {cat} | Deadline: {dl} | AI: {prob}")
    print("="*50)

def view_tasks_by_category():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{GREEN}✅ No tasks!{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}📂 Tasks by Category{RESET}\n{'='*50}")
    for cat in CATEGORIES:
        cat_tasks = my[my["category"] == cat]
        if not cat_tasks.empty:
            print(f"\n{get_category_display(cat)}")
            for _, t in cat_tasks.iterrows():
                tid = int(t["id"])
                stat = f"{GREEN}✅{RESET}" if t["status"] == "Completed" else f"{YELLOW}❌{RESET}"
                tags = format_tags(parse_tags(t["tags"]))
                print(f" #{tid} {t['task']} {tags} {stat}")
    print("="*50)

def search_tasks_by_tags():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{GREEN}✅ No tasks!{RESET}")
        return
    tag = input("\n🏷️ Tag to search: ").strip().lower()
    filtered = my[my["tags"].apply(lambda x: tag in parse_tags(x))]
    if filtered.empty:
        print(f"\n{YELLOW}⚠️ No tasks with tag '{tag}'{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}🔍 Tasks with Tag '{tag}'{RESET}\n{'='*50}")
    for _, t in filtered.iterrows():
        tid = int(t["id"])
        stat = f"{GREEN}✅ Completed{RESET}" if t["status"] == "Completed" else f"{YELLOW}❌ Pending{RESET}"
        pri = "⚡ High" if t["priority"] == "Y" else "Regular"
        tleft, _ = time_left_parts(t["deadline"])
        dl = f"{DEEP_RED}{tleft}{RESET}" if "OVERDUE" in tleft else tleft
        tags = format_tags(parse_tags(t["tags"]))
        cat = get_category_display(t["category"])
        prob = f"{GREEN}✓ {t['ai_prediction']}%{RESET}" if t["ai_prediction"] >= 50 else f"{YELLOW}⚠️ {t['ai_prediction']}%{RESET}"
        print(f"#{tid} {t['task']} {tags}\n {cat} | {pri} | {stat} | Deadline: {dl} | AI: {prob}")
    print("="*50)

def category_summary():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No tasks{RESET}")
        return
    summary = my.groupby("category").agg({
        "status": ["count", lambda x: (x == "Completed").sum()]
    }).reset_index()
    summary.columns = ["Category", "Total", "Completed"]
    summary["Completion Rate (%)"] = (summary["Completed"] / summary["Total"] * 100).round(2)
    print(f"\n{BOLD}{MAGENTA}📊 Category Summary{RESET}\n{'='*50}")
    for _, s in summary.iterrows():
        print(f"{get_category_display(s['Category'])}: {s['Total']} tasks | {GREEN}{s['Completed']} done{RESET} | {s['Completion Rate (%)']}%")
    print("="*50)

# ---------- HABITS (existing) ----------
def create_task_from_habit(hid, name, rec, cat):
    global df_tasks, task_id_counter
    if rec == "daily":
        dl = now_dt().replace(hour=23, minute=59, second=59, microsecond=0)
    elif rec == "weekly":
        days = (6 - now_dt().weekday()) % 7
        dl = (now_dt() + timedelta(days=days)).replace(hour=23, minute=59, second=59, microsecond=0)
    elif rec == "monthly":
        nm = (now_dt().replace(day=1) + timedelta(days=32)).replace(day=1)
        dl = nm - timedelta(seconds=1)
    else:
        dl = pd.NaT
    model = tf.keras.models.load_model(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    scaler = pickle.load(open(SCALER_FILE, 'rb')) if os.path.exists(SCALER_FILE) else None
    prob = predict_nn_prob(model, scaler, {"priority":"Y","task":name,"deadline":dl,"category":cat})
    df_tasks = pd.concat([df_tasks, pd.DataFrame([{
        "id": task_id_counter, "task": name, "priority": "Y", "status": "Pending",
        "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": dl,
        "ai_prediction": round(prob*100,2), "category": cat, "tags": f"habit,{rec}",
        "habit_id": hid, "recurrence": rec, "assigned_to": current_user,
        "created_by": current_user, "shared": False, "estimated_minutes": 30
    }])], ignore_index=True)
    df_tasks.to_csv(DATA_FILE, index=False)
    task_id_counter += 1

def create_recurring_task():
    global df_habits, habit_id_counter
    print(f"\n{BOLD}{MAGENTA}Create Habit{RESET}\n{'─'*50}")
    name = input("📝 Name: ").strip()
    if not name:
        print(f"{RED}❌ Empty name{RESET}")
        return
    print("\n🔄 1. Daily 2. Weekly 3. Monthly")
    rc = input("Choose: ").strip()
    rec = {"1":"daily","2":"weekly","3":"monthly"}.get(rc,"daily")
    print("\n📂 Category:")
    cats = list(CATEGORIES.keys())
    for i,c in enumerate(cats,1):
        print(f" {i}. {get_category_display(c)}")
    cc = input("Number: ").strip()
    cat = cats[int(cc)-1] if cc.isdigit() and 1<=int(cc)<=len(cats) else "Health"
    df_habits = pd.concat([df_habits, pd.DataFrame([{
        "habit_id": habit_id_counter, "habit_name": name, "recurrence": rec, "category": cat,
        "active": True, "created_at": pd.Timestamp.now(), "last_completed": pd.NaT, "total_completions": 0
    }])], ignore_index=True)
    df_habits.to_csv(HABITS_FILE, index=False)
    create_task_from_habit(habit_id_counter, name, rec, cat)
    habit_id_counter += 1
    print(f"\n{GREEN}✅ Created {RECURRENCE_TYPES[rec]['icon']} {RECURRENCE_TYPES[rec]['name']} habit: '{name}'{RESET}")

def view_all_habits():
    if df_habits.empty:
        print(f"\n{YELLOW}⚠️ No habits yet{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}💪 Habits & Streaks{RESET}\n{'='*50}")
    for _,h in df_habits.iterrows():
        if not h["active"]:
            continue
        hid = int(h["habit_id"])
        streak = calculate_streak(hid)
        rd = RECURRENCE_TYPES[h["recurrence"]]["icon"]+" "+RECURRENCE_TYPES[h["recurrence"]]["name"]
        sd = f"{GREEN}🔥 {streak} streak{RESET}" if streak>0 else f"{YELLOW}⚠️ Start streak!{RESET}"
        print(f"\n{BOLD}#{hid} {h['habit_name']}{RESET}\n {rd} | {get_category_display(h['category'])}\n {sd} | Total: {int(h.get('total_completions',0))}")
        tt = df_tasks[(df_tasks["habit_id"]==hid)&(pd.to_datetime(df_tasks["created_at"], errors='coerce').dt.date==now_dt().date())&(df_tasks["status"]=="Pending")]
        print(f" {YELLOW}⚠️ No pending{RESET}" if tt.empty else f" {GREEN}✓ Task #{int(tt.iloc[0]['id'])}{RESET}")
    print("="*50)

def habit_dashboard():
    if df_habits.empty:
        print(f"\n{YELLOW}⚠️ No habits{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}📊 Habit Dashboard{RESET}\n{'='*50}")
    ah = len(df_habits[df_habits["active"]==True])
    tc = df_habits["total_completions"].sum() if "total_completions" in df_habits.columns else 0
    print(f"\n{CYAN}Active Habits:{RESET} {ah}\n{CYAN}Total Completions:{RESET} {int(tc)}")
    print(f"\n{BOLD}🏆 Streak Leaderboard:{RESET}")
    streaks = [(h["habit_name"],calculate_streak(int(h["habit_id"])),h["recurrence"])
               for _,h in df_habits.iterrows() if h["active"]]
    streaks.sort(key=lambda x:x[1],reverse=True)
    for i,(n,s,r) in enumerate(streaks[:5],1):
        m = "🥇" if s>=7 else "🥈" if s>=3 else "🥉"
        c = GREEN if s>=7 else YELLOW if s>=3 else CYAN
        print(f" {m} {i}. {n} {RECURRENCE_TYPES[r]['icon']} - {c}{s} streak{RESET}")
    print(f"\n{BOLD}📂 By Category:{RESET}")
    if not df_habits.empty:
        summary = df_habits.groupby("category")["total_completions"].sum().reset_index()
        for _, s in summary.iterrows():
            if s["total_completions"] > 0:
                print(f" {get_category_display(s['category'])}: {int(s['total_completions'])}")
        try:
            plt.figure(figsize=(8, 5))
            plt.bar(summary["category"], summary["total_completions"], color="#00FF00")
            plt.xlabel("Category")
            plt.ylabel("Total Completions")
            plt.title("Habit Completions by Category")
            plt.tight_layout()
            plt.show()
        except Exception:
            pass
    print("="*50)

# ---------- TEAM (existing) ----------
def add_team_member():
    global df_users
    print(f"\n{BOLD}{MAGENTA}Add Team Member{RESET}\n{'─'*50}")
    un = input("👤 Username: ").strip()
    if not un or un in df_users["username"].values:
        print(f"{RED}❌ Invalid or existing username{RESET}")
        return
    print("\n1. Member 2. Manager")
    rc = input("Choose: ").strip()
    role = "manager" if rc=="2" else "member"
    df_users = pd.concat([df_users, pd.DataFrame([{
        "username": un, "role": role, "added_at": pd.Timestamp.now(), "added_by": current_user
    }])], ignore_index=True)
    df_users.to_csv(USERS_FILE, index=False)
    print(f"\n{GREEN}✅ Added {get_user_color(un)}{un}{RESET} as {role}{RESET}")

def view_team_members():
    if df_users.empty:
        print(f"\n{YELLOW}⚠️ No team members{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}🧑‍🤝‍🧑 Team Members{RESET}\n{'='*50}")
    for _,u in df_users.iterrows():
        un = u["username"]
        ut = df_tasks[df_tasks["assigned_to"]==un]
        tot = len(ut)
        comp = len(ut[ut["status"]=="Completed"])
        pend = tot-comp
        rb = "👑" if u["role"]=="owner" else "⭐" if u["role"]=="manager" else "👤"
        print(f"\n{rb} {get_user_color(un)}{BOLD}{un}{RESET} ({u['role']})\n Tasks: {tot} | {GREEN}{comp} done{RESET} | {YELLOW}{pend} pending{RESET}")
    print("="*50)

def assign_task_to_someone():
    global df_tasks
    my = df_tasks[(df_tasks["created_by"]==current_user)|(df_tasks["assigned_to"]==current_user)]
    if my.empty:
        print(f"\n{YELLOW}⚠️ No tasks{RESET}")
        return
    view_tasks()
    try:
        tid = int(input("\nTask ID: ").strip())
        if tid not in df_tasks["id"].values:
            print(f"{RED}❌ Task not found{RESET}")
            return
        print(f"\n{BOLD}Team:{RESET}")
        for i,un in enumerate(df_users["username"].values,1):
            print(f" {i}. {get_user_color(un)}{un}{RESET}")
        ch = input("\nNumber: ").strip()
        if not ch.isdigit() or int(ch)<1 or int(ch)>len(df_users):
            print(f"{RED}❌ Invalid choice{RESET}")
            return
        na = df_users.iloc[int(ch)-1]["username"]
        df_tasks.loc[df_tasks["id"]==tid, "assigned_to"] = na
        df_tasks.loc[df_tasks["id"]==tid, "shared"] = True
        df_tasks.to_csv(DATA_FILE, index=False)
        tn = df_tasks[df_tasks["id"]==tid].iloc[0]["task"]
        print(f"\n{GREEN}✅ Assigned '{tn}' to {get_user_color(na)}{na}{RESET}!")
    except Exception as e:
        print(f"{RED}❌ Error assigning task: {e}{RESET}")

def view_my_assigned_tasks():
    my = df_tasks[df_tasks["assigned_to"]==current_user]
    if my.empty:
        print(f"\n{GREEN}✅ No assigned tasks!{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}📥 Your Assigned Tasks{RESET}\n{'='*50}")
    for _, t in my.iterrows():
        tid = int(t["id"])
        stat = f"{GREEN}✅ Completed{RESET}" if t["status"] == "Completed" else f"{YELLOW}❌ Pending{RESET}"
        pri = "⚡ High" if t["priority"] == "Y" else "Regular"
        tleft, _ = time_left_parts(t["deadline"])
        dl = f"{DEEP_RED}{tleft}{RESET}" if "OVERDUE" in tleft else tleft
        tags = format_tags(parse_tags(t["tags"]))
        cat = get_category_display(t["category"])
        prob = f"{GREEN}✓ {t['ai_prediction']}%{RESET}" if t["ai_prediction"] >= 50 else f"{YELLOW}⚠️ {t['ai_prediction']}%{RESET}"
        print(f"#{tid} {t['task']} {tags}\n {cat} | {pri} | {stat} | Deadline: {dl} | AI: {prob}")
    print("="*50)

def add_comment_to_task():
    global df_comments, comment_id_counter
    view_tasks()
    try:
        tid = int(input("\nTask ID: ").strip())
        if tid not in df_tasks["id"].values:
            print(f"{RED}❌ Task not found{RESET}")
            return
        tc = df_comments[df_comments["task_id"]==tid].sort_values("timestamp")
        if not tc.empty:
            print(f"\n{BOLD}Comments:{RESET}")
            for _,c in tc.iterrows():
                ts = pd.to_datetime(c['timestamp'], errors='coerce')
                ts_str = ts.strftime('%Y-%m-%d %H:%M') if not pd.isna(ts) else str(c['timestamp'])
                print(f" {get_user_color(c['username'])}{c['username']}{RESET} ({ts_str}): {c['comment']}")
        txt = input(f"\n💬 Comment: ").strip()
        if not txt:
            print(f"{RED}❌ Empty comment{RESET}")
            return
        df_comments = pd.concat([df_comments, pd.DataFrame([{
            "comment_id": comment_id_counter, "task_id": tid, "username": current_user,
            "comment": txt, "timestamp": pd.Timestamp.now()
        }])], ignore_index=True)
        df_comments.to_csv(COMMENTS_FILE, index=False)
        comment_id_counter += 1
        print(f"\n{GREEN}✅ Comment added!{RESET}")
    except Exception as e:
        print(f"{RED}❌ Error adding comment: {e}{RESET}")

def team_dashboard():
    if len(df_users) <= 1:
        print(f"\n{YELLOW}⚠️ Add team members first{RESET}")
        return
    print(f"\n{BOLD}{MAGENTA}📊 Team Dashboard{RESET}\n{'='*50}")
    tot = len(df_tasks)
    comp = len(df_tasks[df_tasks["status"]=="Completed"])
    rate = round((comp/tot*100) if tot>0 else 0,1)
    print(f"\n{CYAN}Team Members:{RESET} {len(df_users)}\n{CYAN}Total Tasks:{RESET} {tot}\n{CYAN}Completed:{RESET} {GREEN}{comp}{RESET} ({rate}%)")
    print(f"\n{BOLD}Performance Leaderboard:{RESET}")
    perfs = []
    for _,u in df_users.iterrows():
        un = u["username"]
        ut = df_tasks[df_tasks["assigned_to"]==un]
        t = len(ut)
        c = len(ut[ut["status"]=="Completed"])
        r = round((c/t*100) if t>0 else 0,1)
        perfs.append((un,t,c,r))
    perfs.sort(key=lambda x:x[3],reverse=True)
    for i,(un,t,c,r) in enumerate(perfs,1):
        m = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else " "
        print(f" {m} {get_user_color(un)}{un}{RESET}: {t} tasks | {c} done | {r}%")
    sh = df_tasks[df_tasks["shared"]==True]
    print(f"\n{CYAN}Shared Tasks:{RESET} {len(sh)}")
    rc = df_comments.sort_values("timestamp",ascending=False).head(5)
    if not rc.empty:
        print(f"\n{BOLD}Recent Comments:{RESET}")
        for _,c in rc.iterrows():
            ts = pd.to_datetime(c['timestamp'], errors='coerce')
            ts_str = ts.strftime('%Y-%m-%d %H:%M') if not pd.isna(ts) else str(c['timestamp'])
            txt = c["comment"][:50]+"..." if len(c["comment"])>50 else c["comment"]
            print(f" {get_user_color(c['username'])}{c['username']}{RESET} ({ts_str}): {txt}")
    try:
        plt.figure(figsize=(8, 5))
        plt.bar([p[0] for p in perfs], [p[3] for p in perfs], color="#00FF00")
        plt.xlabel("Team Member")
        plt.ylabel("Completion Rate (%)")
        plt.title("Team Performance")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass
    print("="*50)

# ------------------ LESSON 20 & 21 FEATURES (merged intelligence) ------------------

# Persistence helper
def save_all():
    try:
        df_tasks.to_csv(DATA_FILE, index=False)
        df_habits.to_csv(HABITS_FILE, index=False)
        df_reminders.to_csv(REMINDER_LOG, index=False)
        df_users.to_csv(USERS_FILE, index=False)
        df_comments.to_csv(COMMENTS_FILE, index=False)
    except Exception as e:
        print(f"{YELLOW}Warning saving files: {e}{RESET}")

# Smart scheduler (Lesson20)
def smart_daily_scheduler(available_minutes=None):
    pending = df_tasks[(df_tasks["assigned_to"]==current_user) & (df_tasks["status"]=="Pending")].copy()
    if pending.empty:
        print("No pending tasks.")
        return
    pending["ai_prob"] = pending["ai_prediction"].apply(lambda x: float(x) if not pd.isna(x) else 50.0)
    pending["priority_num"] = pending["priority"].apply(lambda x:2 if str(x).upper()=="Y" else 1)
    pending["estimated_minutes"] = pending["estimated_minutes"].apply(lambda x: int(x) if not pd.isna(x) else 30)
    def urgency(dl):
        if pd.isna(dl): return 1.0
        try:
            secs = (pd.to_datetime(dl).to_pydatetime() - now_dt()).total_seconds()
        except:
            return 1.0
        if secs <= 0: return 3.0
        if secs <= 3600: return 2.0
        if secs <= 14400: return 1.5
        return 1.0
    pending["urgency"] = pending["deadline"].apply(urgency)
    pending["score"] = pending.apply(lambda r: (r["ai_prob"]/100.0) * r["priority_num"] * r["urgency"] / math.sqrt(r["estimated_minutes"]+1), axis=1)
    pending = pending.sort_values(by="score", ascending=False)
    if available_minutes is None:
        end_of_day = now_dt().replace(hour=18, minute=0, second=0, microsecond=0)
        avail = max(30, int((end_of_day - now_dt()).total_seconds()/60))
    else:
        avail = max(0, int(available_minutes))
    plan = []
    used = 0
    for _, r in pending.iterrows():
        est = int(r["estimated_minutes"])
        if used + est <= avail or not plan:
            plan.append(r)
            used += est
    print(f"\n{MAGENTA}📅 Smart Daily Scheduler — Available minutes: {avail}{RESET}")
    start_time = now_dt()
    for i, r in enumerate(plan,1):
        block_start = start_time
        block_end = block_start + timedelta(minutes=int(r["estimated_minutes"]))
        start_time = block_end + timedelta(minutes=5)
        dl_str = pd.to_datetime(r["deadline"]).strftime("%H:%M") if not pd.isna(r["deadline"]) else "No deadline"
        print(f"{i}. {r['task']} [{r['category']}] | {block_start.strftime('%H:%M')} - {block_end.strftime('%H:%M')} | Est {int(r['estimated_minutes'])}m | AI:{round(r['ai_prob'],2)}% | DL:{dl_str}")
    print(GREEN + f"Total scheduled: {used} minutes of {avail} available." + RESET)
    input("Press Enter to return...")

# Productivity Score System (Lesson21)
def compute_task_metrics(period_days=1):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=period_days)
    df = df_tasks.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
    df_period = df[df["created_at"].notna() & (df["created_at"] >= start)].copy()
    created = len(df_period)
    completed = len(df_period[df_period["status"]=="Completed"])
    overdue = len(df_period[(df_period["status"]!="Completed") & (pd.notna(df_period["deadline"])) & (pd.to_datetime(df_period["deadline"], errors='coerce') < now)])
    avg_time = None
    if completed > 0:
        comp = df_period[df_period["status"]=="Completed"].copy()
        comp["delta"] = (pd.to_datetime(comp["completed_at"], errors='coerce') - pd.to_datetime(comp["created_at"], errors='coerce')).dt.total_seconds()
        avg_time = comp["delta"].mean()
    return {"created":created,"completed":completed,"overdue":overdue,"avg_time_secs":avg_time}

def compute_focus_bursts(period_days=1):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=period_days)
    comp = df_tasks[(df_tasks["status"]=="Completed") & (pd.to_datetime(df_tasks["completed_at"], errors='coerce') >= start)].copy()
    if comp.empty:
        return {"bursts":0,"avg_burst_len_secs":0}
    comp["delta"] = (pd.to_datetime(comp["completed_at"], errors='coerce') - pd.to_datetime(comp["created_at"], errors='coerce')).dt.total_seconds().fillna(0)
    bursts = comp[comp["delta"] >= 25*60]
    avg = bursts["delta"].mean() if len(bursts)>0 else 0
    return {"bursts": int(len(bursts)), "avg_burst_len_secs": avg}

def compute_active_idle_proxy(period_days=1):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=period_days)
    interruptions = 0
    if not df_reminders.empty:
        interruptions = len(df_reminders[pd.to_datetime(df_reminders["timestamp"], errors='coerce') >= start])
    comp = df_tasks[(df_tasks["status"]=="Completed") & (pd.to_datetime(df_tasks["completed_at"], errors='coerce') >= start)].copy()
    total_work_secs = 0
    if not comp.empty:
        comp["delta"] = (pd.to_datetime(comp["completed_at"], errors='coerce') - pd.to_datetime(comp["created_at"], errors='coerce')).dt.total_seconds().fillna(0)
        total_work_secs = comp["delta"].sum()
    factor = 300
    effective_productive = max(0, total_work_secs - interruptions * factor)
    total_activity = total_work_secs + interruptions * factor
    productive_ratio = (effective_productive / total_activity) if total_activity>0 else 0
    return {"interruptions":int(interruptions), "total_work_secs":int(total_work_secs), "productive_ratio":productive_ratio}

def compute_time_accuracy_penalty(period_days=1):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=period_days)
    comp = df_tasks[(df_tasks["status"]=="Completed") & (pd.to_datetime(df_tasks["completed_at"], errors='coerce') >= start)].copy()
    if comp.empty:
        return {"avg_ratio":1.0, "penalty":0.0}
    comp["actual_secs"] = (pd.to_datetime(comp["completed_at"], errors='coerce') - pd.to_datetime(comp["created_at"], errors='coerce')).dt.total_seconds().fillna(0)
    comp["est_secs"] = comp["estimated_minutes"].apply(lambda x: int(x)*60 if not pd.isna(x) else 1800)
    comp["ratio"] = comp.apply(lambda r: (r["actual_secs"]/r["est_secs"]) if r["est_secs"]>0 else 1.0, axis=1)
    avg_ratio = comp["ratio"].mean()
    penalty = 0.0
    if avg_ratio > 1:
        penalty = min(20, (avg_ratio - 1) * 10)
    return {"avg_ratio":avg_ratio, "penalty":penalty}

def compute_priority_score_weight(period_days=1):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=period_days)
    dfp = df_tasks[(pd.to_datetime(df_tasks["created_at"], errors='coerce') >= start)]
    if dfp.empty:
        return {"priority_completion_rate":0.0, "weight":0.0}
    total_prior = len(dfp[dfp["priority"]=="Y"])
    if total_prior == 0:
        return {"priority_completion_rate":0.0, "weight":0.0}
    comp_prior = len(dfp[(dfp["priority"]=="Y") & (dfp["status"]=="Completed")])
    rate = comp_prior / total_prior
    weight = rate * 30
    return {"priority_completion_rate":rate, "weight":weight}

def compute_productivity_score(period_days=1):
    tm = compute_task_metrics(period_days=period_days)
    focus = compute_focus_bursts(period_days=period_days)
    active = compute_active_idle_proxy(period_days=period_days)
    time_acc = compute_time_accuracy_penalty(period_days=period_days)
    pri = compute_priority_score_weight(period_days=period_days)

    completion_rate = (tm["completed"]/tm["created"]) if tm["created"]>0 else 0.0
    completion_component = completion_rate * 40
    priority_component = pri["weight"]
    time_penalty = time_acc["penalty"]
    time_component = max(0, 15 - (time_penalty * 15/20))
    burst_count = focus["bursts"]
    avg_burst = focus["avg_burst_len_secs"]
    burst_score = min(15, burst_count * 3 + (avg_burst/60)/5)
    focus_component = min(15, burst_score)
    raw = completion_component + priority_component + time_component + focus_component
    final = max(0, min(100, raw))
    breakdown = {
        "completion_component": round(completion_component,2),
        "priority_component": round(priority_component,2),
        "time_component": round(time_component,2),
        "focus_component": round(focus_component,2),
        "final_score": round(final,2),
        "details":{"tasks_created":tm["created"], "tasks_completed":tm["completed"], "overdue":tm["overdue"],
                   "interruptions": active["interruptions"], "avg_completion_secs":tm["avg_time_secs"],
                   "avg_time_ratio":time_acc["avg_ratio"] if "avg_ratio" in time_acc else None}
    }
    return breakdown

# Habit score
def compute_habit_score(habit_id, lookback_days=30):
    if df_habits.empty or int(habit_id) not in list(df_habits["habit_id"].astype(int).values):
        return {"error":"habit not found"}
    hrow = df_habits[df_habits["habit_id"]==int(habit_id)].iloc[0]
    recurrence = hrow.get("recurrence","daily")
    start = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    instances = df_tasks[(df_tasks["habit_id"]==int(habit_id)) & (pd.to_datetime(df_tasks["created_at"], errors='coerce') >= start)]
    if recurrence == "daily":
        expected = lookback_days
    elif recurrence == "weekly":
        expected = math.ceil(lookback_days/7)
    elif recurrence == "monthly":
        expected = math.ceil(lookback_days/30)
    else:
        expected = max(1, len(instances))
    completed = len(instances[instances["status"]=="Completed"])
    adherence = (completed / expected) if expected>0 else 0
    streak = 0
    comps = df_tasks[(df_tasks["habit_id"]==int(habit_id)) & (df_tasks["status"]=="Completed")].sort_values("completed_at", ascending=False)
    if not comps.empty:
        expected_date = now_dt().date()
        for _, r in comps.iterrows():
            comp_date = pd.to_datetime(r["completed_at"]).date()
            if recurrence == "daily":
                if comp_date >= expected_date:
                    streak += 1
                    expected_date = expected_date - timedelta(days=1)
                else:
                    break
            elif recurrence == "weekly":
                if comp_date >= expected_date - timedelta(weeks=1):
                    streak += 1
                    expected_date = expected_date - timedelta(weeks=1)
                else:
                    break
            else:
                if comp_date >= expected_date - timedelta(days=30):
                    streak += 1
                    expected_date = expected_date - timedelta(days=30)
                else:
                    break
    score = min(100, adherence*70 + min(30, streak*3))
    return {"habit_id":int(habit_id),"habit_name":hrow["habit_name"], "recurrence":recurrence, "expected":expected, "completed":completed, "adherence":round(adherence,3), "streak":streak, "score":round(score,2)}

# Behavioral analytics
def behavioral_analytics(period_days=30):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=period_days)
    comp = df_tasks[(df_tasks["status"]=="Completed") & (pd.to_datetime(df_tasks["completed_at"], errors='coerce') >= start)].copy()
    if not comp.empty:
        comp["day"] = pd.to_datetime(comp["completed_at"], errors='coerce').dt.floor("D")
        daily = comp.groupby("day").size().reindex(pd.date_range(start.floor('D'), now.floor('D'), freq='D'), fill_value=0)
        daily.index = [d.date() for d in daily.index]
        daily_counts = daily.to_dict()
    else:
        daily_counts = {}
    top_hours = {}
    if not comp.empty:
        comp["hour"] = pd.to_datetime(comp["completed_at"], errors='coerce').dt.hour
        hour_counts = comp.groupby("hour").size().sort_values(ascending=False)
        top_hours = hour_counts.head(6).to_dict()
    cat_stats = {}
    if "category" in df_tasks.columns:
        for cat, grp in df_tasks.groupby("category"):
            total = len(grp)
            completed_grp = grp[grp["status"]=="Completed"]
            avg_overrun = None
            if not completed_grp.empty:
                completed_grp = completed_grp.copy()
                completed_grp["delta_secs"] = (pd.to_datetime(completed_grp["completed_at"], errors='coerce') - pd.to_datetime(completed_grp["created_at"], errors='coerce')).dt.total_seconds().fillna(0)
                completed_grp["est_secs"] = completed_grp["estimated_minutes"].apply(lambda x: int(x)*60 if not pd.isna(x) else 1800)
                completed_grp["over"] = (completed_grp["delta_secs"] - completed_grp["est_secs"])
                avg_overrun = completed_grp["over"].mean() if len(completed_grp)>0 else 0
            cat_stats[cat] = {"total": int(total), "completed": int((grp["status"]=="Completed").sum()), "avg_overrun_secs": float(avg_overrun) if avg_overrun is not None else None}
    interruptions = {}
    if not df_reminders.empty:
        recent = df_reminders[pd.to_datetime(df_reminders["timestamp"], errors='coerce') >= start]
        interruptions["total"] = len(recent)
        interruptions["by_type"] = recent["alert_type"].value_counts().to_dict()
    else:
        interruptions["total"] = 0
        interruptions["by_type"] = {}
    alerts = []
    tm7 = compute_task_metrics(period_days=7)
    if tm7["overdue"] >= max(3, tm7["created"]//5):
        alerts.append({"type":"miss_deadline_risk", "message":f"High overdue count in last 7 days: {tm7['overdue']}."})
    recent_comp = df_tasks[(df_tasks["status"]=="Completed") & (pd.to_datetime(df_tasks["completed_at"], errors='coerce') >= pd.Timestamp.now() - pd.Timedelta(days=7))].copy()
    total_work_secs = 0
    if not recent_comp.empty:
        recent_comp["delta"] = (pd.to_datetime(recent_comp["completed_at"], errors='coerce') - pd.to_datetime(recent_comp["created_at"], errors='coerce')).dt.total_seconds().fillna(0)
        total_work_secs = recent_comp["delta"].sum()
    if total_work_secs > 10*3600:
        alerts.append({"type":"overwork", "message":f"High work detected: {int(total_work_secs/3600)} hours completed in last 7 days."})
    inter7 = compute_active_idle_proxy(period_days=7)
    if inter7["interruptions"] > 20:
        alerts.append({"type":"high_context_switch", "message":f"Many interruptions ({inter7['interruptions']}) in last 7 days."})
    return {"period_days":period_days, "daily_completed_counts":daily_counts, "top_hours":top_hours, "category_stats":cat_stats, "interruptions":interruptions, "alerts":alerts}

# Admin export
def export_admin_report(filename="admin_report.csv", period_days=30):
    prod = compute_productivity_score(period_days=period_days)
    ba = behavioral_analytics(period_days=period_days)
    rows = []
    rows.append({"metric":"productivity_score","value":prod["final_score"]})
    for k,v in prod.items():
        if k!="final_score":
            rows.append({"metric":f"prod_{k}","value":str(v)})
    rows.append({"metric":"ba_period_days","value":ba["period_days"]})
    rows.append({"metric":"ba_top_hours","value":str(ba["top_hours"])})
    rows.append({"metric":"ba_interruptions","value":str(ba["interruptions"])})
    for _, h in df_habits.iterrows():
        hid = int(h["habit_id"])
        hs = compute_habit_score(hid, lookback_days=period_days)
        rows.append({"metric":f"habit_{hid}_score","value":hs.get("score")})
        rows.append({"metric":f"habit_{hid}_streak","value":hs.get("streak")})
    out = pd.DataFrame(rows)
    out.to_csv(filename, index=False)
    return filename

# Productivity Dashboard (Option 28)
def user_panel():
    prod = compute_productivity_score(period_days=1)
    print(f"\n{MAGENTA}User Panel — Productivity Today{RESET}")
    print(f"Score: {GREEN}{prod['final_score']}{RESET}/100")
    print("Breakdown:")
    print(f" - Completion: {prod['completion_component']}")
    print(f" - Priority: {prod['priority_component']}")
    print(f" - Time accuracy: {prod['time_component']}")
    print(f" - Focus: {prod['focus_component']}")
    details = prod.get("details", {})
    print(f"Details: Tasks created {details.get('tasks_created')}, completed {details.get('tasks_completed')}, overdue {details.get('overdue')}")
    # Habit quick view
    print("\nHabits snapshot:")
    if df_habits.empty:
        print(" No habits yet.")
    else:
        for _, h in df_habits.iterrows():
            hid = int(h["habit_id"])
            hs = compute_habit_score(hid, lookback_days=14)
            trend = "Improving" if hs.get("score",0)>=50 and hs.get("streak",0)>0 else "Needs Attention"
            print(f" - {h['habit_name']}: Score {hs.get('score',0)} | Streak {hs.get('streak',0)} | {trend}")
    ba = behavioral_analytics(period_days=14)
    print("\nInsights:")
    top_hours = ba.get("top_hours", {})
    if top_hours:
        print(f" - Peak productive hour(s): {', '.join([str(k)+':00' for k in top_hours.keys()])}")
    else:
        print(" - Not enough data to determine peak hours.")
    cat_stats = ba.get("category_stats", {})
    slow = sorted([(k,v.get("avg_overrun_secs") or 0) for k,v in cat_stats.items()], key=lambda x: x[1], reverse=True)
    if slow:
        print(f" - Most delayed category (avg overrun): {slow[0][0]} ({slow[0][1]} secs avg overrun)")
    inter = ba.get("interruptions", {})
    print(f" - Interruptions in period: {inter.get('total',0)}")
    input("\nPress Enter to continue...")

def admin_panel():
    print("\n" + MAGENTA + BOLD + "Admin Panel — Team / Admin Insights" + RESET)
    prod30 = compute_productivity_score(period_days=30)
    print(f"Productivity Score (30 days aggregate): {GREEN}{prod30['final_score']}{RESET}/100")
    ba = behavioral_analytics(period_days=30)
    print("\nTop productive hours (30d):")
    for h,c in ba.get("top_hours", {}).items():
        print(f" - {h}:00 -> {c} tasks completed")
    print("\nCategory stats (sample):")
    cs = ba.get("category_stats", {})
    for k,v in list(cs.items())[:6]:
        print(f" - {k}: total {v['total']}, completed {v['completed']}, avg_overrun {v['avg_overrun_secs']}")
    if ba.get("alerts"):
        print("\n" + RED + "⚠️ Alerts:" + RESET)
        for a in ba["alerts"]:
            print(" -", a["message"])
    exp = input("\nExport admin CSV report? (Y/N): ").strip().upper()
    if exp == "Y":
        fname = input("Filename (default admin_report.csv): ").strip() or "admin_report.csv"
        path = export_admin_report(filename=fname, period_days=30)
        print("Exported to:", path)
    input("\nPress Enter to continue...")

def productivity_dashboard():
    while True:
        print("\n" + MAGENTA + BOLD + "📊 Productivity Dashboard" + RESET)
        print("1. User Panel")
        print("2. Admin Panel")
        print("3. Back to Main Menu")
        choice = input("Choose: ").strip()
        if choice == "1":
            user_panel()
        elif choice == "2":
            admin_panel()
        elif choice == "3":
            break
        else:
            print("Invalid choice.")

# ---------- MENU ----------
def show_menu():
    rc = reminder_queue.qsize()
    badge = f" {RED}[{rc} alerts!]{RESET}" if rc > 0 else ""
    mt = len(df_tasks[df_tasks["assigned_to"] == current_user])
    print(f"{ORANGE}{'━'*40}{RESET}")
    print(f"{ORANGE}🏠 Welcome, {MAGENTA}{current_user}{RESET}! ({mt} tasks){badge}")
    labels = [
        "Add Task", "View Tasks", "Remove Task", "Mark Completed", "Productivity Chart",
        "Weekly Summary", "Top Productive Days", "Edit Task", "Filter Tasks", "Productivity Trend",
        "Smart Productivity Trend", "Daily AI Plan", "Save & Exit", "View Reminders",
        "Tasks by Category", "Search by Tags", "Category Summary", "Create Habit", "View Habits",
        "Habit Dashboard", "Add Team Member 👥", "View Team 🧑‍🤝‍🧑", "Assign Task 📤",
        "My Assigned Tasks 📥", "Add Comment 💬", "Team Dashboard 📊", "Train AI Model 🧠",
        "Productivity Dashboard (User/Admin)"  # 28
    ]
    for i, label in enumerate(labels, 1):
        print(f"{i}️⃣ {label}")
    print(f"{ORANGE}{'━'*40}{RESET}")

# ---------- MAIN ----------
def main():
    start_reminder_system()
    try:
        while True:
            clear_console()
            show_menu()
            choice = input("Choose an option: ").strip()
            if choice == "1":
                add_task()
            elif choice == "2":
                view_tasks()
            elif choice == "3":
                remove_task()
            elif choice == "4":
                mark_task_completed()
            elif choice == "5":
                productivity_chart()
            elif choice == "6":
                weekly_performance_summary()
            elif choice == "7":
                top_productive_days()
            elif choice == "8":
                edit_task()
            elif choice == "9":
                filter_tasks()
            elif choice == "10":
                productivity_trend()
            elif choice == "11":
                smart_productivity_trend()
            elif choice == "12":
                daily_ai_plan()
            elif choice == "13":
                # Save & Exit
                df_tasks.to_csv(DATA_FILE, index=False)
                df_habits.to_csv(HABITS_FILE, index=False)
                df_users.to_csv(USERS_FILE, index=False)
                df_comments.to_csv(COMMENTS_FILE, index=False)
                stop_reminder_system()
                print(f"\n{GREEN}✅ Saved and exited{RESET}")
                break
            elif choice == "14":
                show_pending_reminders()
            elif choice == "15":
                view_tasks_by_category()
            elif choice == "16":
                search_tasks_by_tags()
            elif choice == "17":
                category_summary()
            elif choice == "18":
                create_recurring_task()
            elif choice == "19":
                view_all_habits()
            elif choice == "20":
                habit_dashboard()
            elif choice == "21":
                add_team_member()
            elif choice == "22":
                view_team_members()
            elif choice == "23":
                assign_task_to_someone()
            elif choice == "24":
                view_my_assigned_tasks()
            elif choice == "25":
                add_comment_to_task()
            elif choice == "26":
                team_dashboard()
            elif choice == "27":
                train_nn_model()
            elif choice == "28":
                productivity_dashboard()
            else:
                print(f"{RED}❌ Invalid option{RESET}")
            # Pause for user
            input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        # Save state on Ctrl-C
        df_tasks.to_csv(DATA_FILE, index=False)
        df_habits.to_csv(HABITS_FILE, index=False)
        df_users.to_csv(USERS_FILE, index=False)
        df_comments.to_csv(COMMENTS_FILE, index=False)
        stop_reminder_system()
        print(f"\n{GREEN}✅ Saved and exited{RESET}")

if __name__ == "__main__":
    main()
