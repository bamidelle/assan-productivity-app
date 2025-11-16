# ==============================================
# ASSAN - COMPLETE PRODUCTIVITY APP
# Full Implementation with ALL Features
# ==============================================

import os
import sys
import time
import threading
import queue
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For Colab compatibility
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available - using simple ML")

try:
    import pickle
    PICKLE_AVAILABLE = True
except:
    PICKLE_AVAILABLE = False

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
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except:
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

def calculate_streak(habit_id):
    if df_tasks.empty:
        return 0
    habit_tasks = df_tasks[(df_tasks["habit_id"] == habit_id) & (df_tasks["status"] == "Completed")].sort_values("completed_at", ascending=False)
    if habit_tasks.empty:
        return 0
    habit_info = df_habits[df_habits["habit_id"] == habit_id].iloc[0]
    recurrence = habit_info["recurrence"]
    streak = 0
    expected_date = now_dt().date()
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
        elif recurrence == "monthly":
            if completed_date >= expected_date - timedelta(days=30):
                streak += 1
                expected_date = completed_date - timedelta(days=30)
            else:
                break
    return streak

# ---------- LOAD USER ----------
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        current_user = f.read().strip()
else:
    current_user = input("Enter your name: ").strip()
    with open(USER_FILE, "w") as f:
        f.write(current_user)

print(f"\n{GREEN}{'='*60}{RESET}")
print(f"{BOLD}{MAGENTA}🎉 ASSAN - YOUR PRODUCTIVITY COMPANION 🎉{RESET}")
print(f"{GREEN}{'='*60}{RESET}")
print(f"{CYAN}Welcome, {MAGENTA}{current_user}{RESET}!{RESET}")
print(f"{GREEN}{'='*60}{RESET}\n")

# ---------- LOAD DATA ----------
if os.path.exists(USERS_FILE):
    df_users = pd.read_csv(USERS_FILE)
    df_users["added_at"] = pd.to_datetime(df_users["added_at"], errors='coerce')
else:
    df_users = pd.DataFrame(columns=["username", "role", "added_at", "added_by"])
    df_users = pd.concat([df_users, pd.DataFrame([{"username": current_user, "role": "owner", "added_at": pd.Timestamp.now(), "added_by": "self"}])], ignore_index=True)
    df_users.to_csv(USERS_FILE, index=False)

if current_user not in df_users["username"].values:
    df_users = pd.concat([df_users, pd.DataFrame([{"username": current_user, "role": "owner", "added_at": pd.Timestamp.now(), "added_by": "self"}])], ignore_index=True)
    df_users.to_csv(USERS_FILE, index=False)

if os.path.exists(DATA_FILE):
    df_tasks = pd.read_csv(DATA_FILE, low_memory=False)
    for c in ["created_at", "completed_at", "deadline"]:
        if c in df_tasks.columns:
            df_tasks[c] = pd.to_datetime(df_tasks[c], errors='coerce')
    for col, default in [("category", "Other"), ("tags", ""), ("ai_prediction", np.nan), ("habit_id", np.nan), ("recurrence", "none"), ("assigned_to", current_user), ("created_by", current_user), ("shared", False), ("deadline", pd.NaT)]:
        if col not in df_tasks.columns:
            df_tasks[col] = default
else:
    df_tasks = pd.DataFrame(columns=["id","task","priority","status","created_at","completed_at","deadline","ai_prediction","category","tags","habit_id","recurrence","assigned_to","created_by","shared"])

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
    df_comments["timestamp"] = pd.to_datetime(df_comments["timestamp"], errors='coerce')
else:
    df_comments = pd.DataFrame(columns=["comment_id", "task_id", "username", "comment", "timestamp"])
comment_id_counter = int(df_comments["comment_id"].max()) + 1 if not df_comments.empty else 1

if os.path.exists(REMINDER_LOG):
    df_reminders = pd.read_csv(REMINDER_LOG)
    df_reminders["timestamp"] = pd.to_datetime(df_reminders["timestamp"], errors='coerce')
else:
    df_reminders = pd.DataFrame(columns=["task_id", "task_name", "alert_type", "timestamp"])

def log_reminder(task_id, task_name, alert_type):
    global df_reminders
    df_reminders = pd.concat([df_reminders, pd.DataFrame([{"task_id": task_id, "task_name": task_name, "alert_type": alert_type, "timestamp": pd.Timestamp.now()}])], ignore_index=True)
    df_reminders.to_csv(REMINDER_LOG, index=False)

# ---------- REMINDERS ----------
def check_reminders_background():
    global df_tasks
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
                        reminder_queue.put({"task_id": task_id, "task_name": task["task"], "alert_type": alert_type})
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
        print(f"{GREEN}🔔 Reminder system activated{RESET}")

def stop_reminder_system():
    stop_reminder_thread.set()
    if reminder_thread:
        reminder_thread.join(timeout=1)

def show_pending_reminders():
    count = reminder_queue.qsize()
    if count == 0:
        print(f"{GREEN}✅ No reminders!{RESET}")
        return
    print(f"\n{ORANGE}{'='*50}{RESET}\n{BOLD}{ORANGE}🔔 {count} REMINDER(S):{RESET}\n{ORANGE}{'='*50}{RESET}\n")
    temp = []
    while not reminder_queue.empty():
        temp.append(reminder_queue.get())
    for rem in temp:
        tid, name, atype = rem["task_id"], rem["task_name"], rem["alert_type"]
        if atype == "overdue":
            print(f"{RED}🔴 OVERDUE: Task #{tid}: {name}{RESET}")
        elif atype == "urgent":
            print(f"{ORANGE}🟠 URGENT: Task #{tid}: {name}{RESET}")
        elif atype == "warning":
            print(f"{YELLOW}🟡 WARNING: Task #{tid}: {name}{RESET}")
    print(f"\n{ORANGE}{'='*50}{RESET}")
    for rem in temp:
        reminder_queue.put(rem)

# ---------- ML ----------
def train_nn_model():
    global df_tasks
    if not TF_AVAILABLE:
        print(f"{YELLOW}TensorFlow not available, using simple ML{RESET}")
        return train_simple_model()
    
    if len(df_tasks[df_tasks["status"]=="Completed"]) < 10:
        print(f"{RED}Need 10+ completed tasks{RESET}")
        return None, None
    
    d = df_tasks.copy()
    d["priority_num"] = d["priority"].map({"Y":1, "N":0}).fillna(0).astype(int)
    d["task_len"] = d["task"].astype(str).apply(len)
    d["days_to_deadline"] = (d["deadline"] - pd.Timestamp.now()).dt.total_seconds() / 86400
    d["days_to_deadline"] = d["days_to_deadline"].fillna(0)
    d["completed"] = (d["status"] == "Completed").astype(int)
    
    X = d[["priority_num", "task_len", "days_to_deadline"]]
    y = d["completed"]
    
    if len(y.unique()) < 2:
        print(f"{RED}Need both completed & pending tasks{RESET}")
        return None, None
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(16, activation='relu', input_dim=X.shape[1]),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f"\n{MAGENTA}Training...{RESET}")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    model.save(MODEL_FILE)
    if PICKLE_AVAILABLE:
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
    
    print(f"{GREEN}✅ Trained! Accuracy: {history.history['accuracy'][-1]:.2%}{RESET}")
    return model, scaler

def train_simple_model():
    d = df_tasks.copy()
    if d.empty:
        return None, None
    d["priority_num"] = d["priority"].map({"Y":1,"N":0}).fillna(0).astype(int)
    d["task_len"] = d["task"].astype(str).apply(len)
    d["completed"] = (d["status"]=="Completed").astype(int)
    X = d[["priority_num","task_len"]]
    y = d["completed"]
    if len(y.unique()) < 2:
        return None, None
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    print(f"{GREEN}✅ Simple model trained{RESET}")
    return model, None

def predict_prob(row):
    if TF_AVAILABLE and os.path.exists(MODEL_FILE):
        try:
            model = tf.keras.models.load_model(MODEL_FILE)
            if PICKLE_AVAILABLE and os.path.exists(SCALER_FILE):
                with open(SCALER_FILE, 'rb') as f:
                    scaler = pickle.load(f)
                d = pd.DataFrame([{
                    "priority_num": 1 if str(row.get("priority", "N")).upper() == "Y" else 0,
                    "task_len": len(str(row.get("task", ""))),
                    "days_to_deadline": (parse_deadline_input(row.get("deadline", "")) - pd.Timestamp.now()).total_seconds() / 86400 if row.get("deadline") else 0
                }])
                X = scaler.transform(d)
                return float(model.predict(X, verbose=0)[0][0])
        except:
            pass
    return 0.8 if str(row.get("priority","N")).upper()=="Y" else 0.3

# ---------- AUTO-DOWNLOAD HELPER ----------
def trigger_download(filename):
    """Force download in Google Colab"""
    try:
        from google.colab import files
        import time
        time.sleep(0.5)  # Small delay to ensure file is written
        files.download(filename)
        return True
    except:
        return False

# ---------- EXPORT ----------
def export_to_csv():
    filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    my_tasks = df_tasks[df_tasks["assigned_to"] == current_user].copy()
    
    # Save to CSV
    my_tasks.to_csv(filename, index=False)
    
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN}✅ CSV FILE CREATED!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"📄 File: {MAGENTA}{filename}{RESET}")
    print(f"📊 Records: {len(my_tasks)} tasks")
    print(f"\n{CYAN}Starting download...{RESET}")
    
    # Force download
    if trigger_download(filename):
        print(f"{GREEN}✅ Download started! Check your browser downloads.{RESET}")
    else:
        print(f"{YELLOW}⚠️ Auto-download unavailable.{RESET}")
        print(f"📁 Manual: Files panel → {filename} → ⋮ → Download")
    
    print(f"{GREEN}{'='*60}{RESET}\n")
    return filename

def export_to_excel():
    try:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        my_tasks = df_tasks[df_tasks["assigned_to"] == current_user].copy()
        
        # Create Excel file
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            my_tasks.to_excel(writer, sheet_name='All Tasks', index=False)
            
            if not my_tasks.empty:
                summary = my_tasks.groupby("category").agg({
                    "status": ["count", lambda x: (x == "Completed").sum()]
                }).reset_index()
                summary.columns = ["Category", "Total", "Completed"]
                summary["Pending"] = summary["Total"] - summary["Completed"]
                summary["Rate %"] = (summary["Completed"] / summary["Total"] * 100).round(1)
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            pending = my_tasks[my_tasks["status"] == "Pending"]
            if not pending.empty:
                pending.to_excel(writer, sheet_name='Pending', index=False)
            
            completed = my_tasks[my_tasks["status"] == "Completed"]
            if not completed.empty:
                completed.to_excel(writer, sheet_name='Completed', index=False)
            
            if not df_habits.empty:
                df_habits.to_excel(writer, sheet_name='Habits', index=False)
        
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{BOLD}{GREEN}✅ EXCEL FILE CREATED!{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"📄 File: {MAGENTA}{filename}{RESET}")
        print(f"📊 Records: {len(my_tasks)} tasks")
        print(f"📋 Sheets: All Tasks, Summary, Pending, Completed")
        if not df_habits.empty:
            print(f"         + Habits sheet")
        print(f"\n{CYAN}Starting download...{RESET}")
        
        # Force download
        if trigger_download(filename):
            print(f"{GREEN}✅ Download started! Check your browser downloads.{RESET}")
        else:
            print(f"{YELLOW}⚠️ Auto-download unavailable.{RESET}")
            print(f"📁 Manual: Files panel → {filename} → ⋮ → Download")
        
        print(f"{GREEN}{'='*60}{RESET}\n")
        return filename
            
    except ImportError:
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}❌ MISSING LIBRARY{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        print(f"\n{YELLOW}Run: {CYAN}!pip install openpyxl{RESET}")
        print(f"Then restart the app.\n")
        return None
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}\n")
        return None

def generate_pdf_report():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors as rl_colors
        
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        my_tasks = df_tasks[df_tasks["assigned_to"] == current_user]
        
        # Create PDF
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(f"<b>Assan Productivity Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        user_para = Paragraph(f"<b>User:</b> {current_user}", styles['Normal'])
        story.append(user_para)
        
        date_para = Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal'])
        story.append(date_para)
        story.append(Spacer(1, 20))
        
        # Summary
        total = len(my_tasks)
        completed = len(my_tasks[my_tasks["status"] == "Completed"])
        pending = total - completed
        rate = round((completed / total * 100) if total > 0 else 0, 1)
        
        summary_title = Paragraph("<b>Summary Statistics</b>", styles['Heading2'])
        story.append(summary_title)
        story.append(Spacer(1, 8))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Tasks', str(total)],
            ['Completed', str(completed)],
            ['Pending', str(pending)],
            ['Completion Rate', f"{rate}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[150, 100])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, rl_colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Tasks
        task_title = Paragraph("<b>Task List (Top 30)</b>", styles['Heading2'])
        story.append(task_title)
        story.append(Spacer(1, 8))
        
        task_data = [['ID', 'Task', 'Status', 'Priority']]
        for _, t in my_tasks.head(30).iterrows():
            task_text = str(t['task'])[:45] + ('...' if len(str(t['task'])) > 45 else '')
            task_data.append([
                str(int(t['id'])),
                task_text,
                str(t['status'])[:10],
                '⭐' if t['priority'] == 'Y' else ''
            ])
        
        task_table = Table(task_data, colWidths=[40, 300, 80, 50])
        task_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, rl_colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#F0F0F0')])
        ]))
        story.append(task_table)
        
        # Build PDF
        doc.build(story)
        
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{BOLD}{GREEN}✅ PDF REPORT CREATED!{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"📄 File: {MAGENTA}{filename}{RESET}")
        print(f"📊 Pages: Professional report with tables")
        print(f"\n{CYAN}Starting download...{RESET}")
        
        # Force download
        if trigger_download(filename):
            print(f"{GREEN}✅ Download started! Check your browser downloads.{RESET}")
        else:
            print(f"{YELLOW}⚠️ Auto-download unavailable.{RESET}")
            print(f"📁 Manual: Files panel → {filename} → ⋮ → Download")
        
        print(f"{GREEN}{'='*60}{RESET}\n")
        return filename
            
    except ImportError:
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}❌ MISSING LIBRARY: reportlab{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        print(f"\n{BOLD}Install first:{RESET}")
        print(f"{CYAN}!pip install reportlab{RESET}")
        print(f"\n{YELLOW}Run this in a SEPARATE cell, then restart the app.{RESET}\n")
        return None
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}\n")
        return None

def list_export_files():
    """NEW: Show all export files in the current directory"""
    import glob
    exports = []
    for pattern in ['export_*.csv', 'export_*.xlsx', 'report_*.pdf']:
        exports.extend(glob.glob(pattern))
    
    if not exports:
        print(f"\n{YELLOW}No export files found yet.{RESET}")
        print(f"Create exports using options 28, 29, or 30.\n")
        return
    
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}📁 YOUR EXPORT FILES:{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")
    
    exports.sort(reverse=True)
    for i, file in enumerate(exports, 1):
        size = os.path.getsize(file)
        size_kb = round(size / 1024, 1)
        file_type = "📄 CSV" if file.endswith('.csv') else "📗 Excel" if file.endswith('.xlsx') else "📕 PDF"
        print(f"{GREEN}{i}.{RESET} {file_type} {MAGENTA}{file}{RESET} ({size_kb} KB)")
    
    print(f"\n{BOLD}{CYAN}📥 TO DOWNLOAD ANY FILE:{RESET}")
    print(f"{YELLOW}1.{RESET} Click {BOLD}📁 Files{RESET} icon on left")
    print(f"{YELLOW}2.{RESET} Find your file")
    print(f"{YELLOW}3.{RESET} Click {BOLD}⋮{RESET} (three dots)")
    print(f"{YELLOW}4.{RESET} Click {BOLD}{GREEN}Download{RESET}")
    print(f"\n{CYAN}{'='*60}{RESET}\n")

def email_summary():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    total = len(my)
    comp = len(my[my["status"] == "Completed"])
    rate = round((comp / total * 100) if total > 0 else 0, 1)
    print(f"\n{MAGENTA}Email Summary{RESET}\n{'='*50}")
    print(f"Tasks: {total} | Done: {comp} ({rate}%)")
    print("="*50)

# ---------- HABITS ----------
def create_task_from_habit(hid, name, rec, cat):
    global df_tasks, task_id_counter
    if rec == "daily":
        dl = now_dt().replace(hour=23, minute=59)
    elif rec == "weekly":
        days = (6 - now_dt().weekday()) % 7
        dl = (now_dt() + timedelta(days=days)).replace(hour=23, minute=59)
    else:
        nm = (now_dt().replace(day=1) + timedelta(days=32)).replace(day=1)
        dl = nm - timedelta(seconds=1)
    
    prob = predict_prob({"priority":"Y","task":name,"deadline":dl,"category":cat})
    df_tasks = pd.concat([df_tasks, pd.DataFrame([{
        "id": task_id_counter, "task": name, "priority": "Y", "status": "Pending",
        "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": dl,
        "ai_prediction": round(prob*100,2), "category": cat, "tags": f"habit,{rec}",
        "habit_id": hid, "recurrence": rec, "assigned_to": current_user,
        "created_by": current_user, "shared": False
    }])], ignore_index=True)
    df_tasks.to_csv(DATA_FILE, index=False)
    task_id_counter += 1

def create_recurring_task():
    global df_habits, habit_id_counter
    print(f"\n{MAGENTA}Create Habit{RESET}")
    name = input("Name: ").strip()
    if not name:
        return
    print("1.Daily 2.Weekly 3.Monthly")
    rc = input("Choose: ").strip()
    rec = {"1":"daily","2":"weekly","3":"monthly"}.get(rc,"daily")
    print("\nCategory:")
    cats = list(CATEGORIES.keys())
    for i,c in enumerate(cats,1):
        print(f"{i}. {c}")
    cc = input("Number: ").strip()
    cat = cats[int(cc)-1] if cc.isdigit() and 1<=int(cc)<=len(cats) else "Health"
    df_habits = pd.concat([df_habits, pd.DataFrame([{
        "habit_id": habit_id_counter, "habit_name": name, "recurrence": rec, "category": cat,
        "active": True, "created_at": pd.Timestamp.now(), "last_completed": pd.NaT, "total_completions": 0
    }])], ignore_index=True)
    df_habits.to_csv(HABITS_FILE, index=False)
    create_task_from_habit(habit_id_counter, name, rec, cat)
    habit_id_counter += 1
    print(f"{GREEN}✅ Created habit: {name}{RESET}")

def view_all_habits():
    if df_habits.empty:
        print(f"{YELLOW}No habits{RESET}")
        return
    print(f"\n{MAGENTA}Habits{RESET}\n{'='*50}")
    for _,h in df_habits.iterrows():
        if not h["active"]:
            continue
        hid = int(h["habit_id"])
        streak = calculate_streak(hid)
        print(f"#{hid} {h['habit_name']} | {h['recurrence']} | 🔥 {streak} streak")
    print("="*50)

def habit_dashboard():
    if df_habits.empty:
        print(f"{YELLOW}No habits{RESET}")
        return
    print(f"\n{MAGENTA}Habit Dashboard{RESET}\n{'='*50}")
    streaks = [(h["habit_name"],calculate_streak(int(h["habit_id"]))) for _,h in df_habits.iterrows() if h["active"]]
    streaks.sort(key=lambda x:x[1],reverse=True)
    for i,(n,s) in enumerate(streaks[:5],1):
        m = "🥇" if i==1 else "🥈" if i==2 else "🥉"
        print(f"{m} {n}: {s} streak")
    print("="*50)

# ---------- TEAM ----------
def add_team_member():
    global df_users
    un = input("Username: ").strip()
    if not un or un in df_users["username"].values:
        print(f"{RED}Invalid{RESET}")
        return
    role = "manager" if input("Manager? (y/n): ").lower()=="y" else "member"
    df_users = pd.concat([df_users, pd.DataFrame([{
        "username": un, "role": role, "added_at": pd.Timestamp.now(), "added_by": current_user
    }])], ignore_index=True)
    df_users.to_csv(USERS_FILE, index=False)
    print(f"{GREEN}✅ Added {un}{RESET}")

def view_team_members():
    if df_users.empty:
        print(f"{YELLOW}No members{RESET}")
        return
    print(f"\n{MAGENTA}Team{RESET}\n{'='*50}")
    for _,u in df_users.iterrows():
        un = u["username"]
        ut = df_tasks[df_tasks["assigned_to"]==un]
        print(f"{un} ({u['role']}): {len(ut)} tasks")
    print("="*50)

def assign_task():
    global df_tasks
    view_tasks()
    tid = int(input("Task ID: "))
    if tid not in df_tasks["id"].values:
        print(f"{RED}Not found{RESET}")
        return
    for i,un in enumerate(df_users["username"].values,1):
        print(f"{i}. {un}")
    ch = input("Number: ").strip()
    if ch.isdigit() and 1<=int(ch)<=len(df_users):
        na = df_users.iloc[int(ch)-1]["username"]
        df_tasks.loc[df_tasks["id"]==tid, "assigned_to"] = na
        df_tasks.loc[df_tasks["id"]==tid, "shared"] = True
        df_tasks.to_csv(DATA_FILE, index=False)
        print(f"{GREEN}✅ Assigned to {na}{RESET}")

def add_comment():
    global df_comments, comment_id_counter
    view_tasks()
    tid = int(input("Task ID: "))
    if tid not in df_tasks["id"].values:
        print(f"{RED}Not found{RESET}")
        return
    txt = input("Comment: ").strip()
    if txt:
        df_comments = pd.concat([df_comments, pd.DataFrame([{
            "comment_id": comment_id_counter, "task_id": tid, "username": current_user,
            "comment": txt, "timestamp": pd.Timestamp.now()
        }])], ignore_index=True)
        df_comments.to_csv(COMMENTS_FILE, index=False)
        comment_id_counter += 1
        print(f"{GREEN}✅ Added{RESET}")

def team_dashboard():
    if len(df_users) <= 1:
        print(f"{YELLOW}Add members first{RESET}")
        return
    print(f"\n{MAGENTA}Team Dashboard{RESET}\n{'='*50}")
    for _,u in df_users.iterrows():
        un = u["username"]
        ut = df_tasks[df_tasks["assigned_to"]==un]
        comp = len(ut[ut["status"]=="Completed"])
        total = len(ut)
        rate = round((comp/total*100) if total>0 else 0,1)
        print(f"{un}: {total} tasks | {comp} done | {rate}%")
    print("="*50)

# ---------- CORE TASKS ----------
def add_task():
    global df_tasks, task_id_counter
    while True:
        print(f"\n{MAGENTA}Add Task (type 'Home' to stop){RESET}")
        name = input("Task: ").strip()
        if name.lower() == "home":
            break
        if not name:
            continue
        pr = input("Priority (Y/N): ").strip().upper()
        if pr not in ["Y","N"]:
            pr = "N"
        dl = input("Deadline (YYYY-MM-DD HH:MM): ").strip()
        dl_ts = parse_deadline_input(dl)
        print("\nCategory:")
        cats = list(CATEGORIES.keys())
        for i,c in enumerate(cats,1):
            print(f"{i}. {c}")
        cc = input("Number: ").strip()
        cat = cats[int(cc)-1] if cc.isdigit() and 1<=int(cc)<=len(cats) else "Other"
        tags = input("Tags: ").strip()
        prob = predict_prob({"priority": pr, "task": name, "deadline": dl, "category": cat})
        df_tasks = pd.concat([df_tasks, pd.DataFrame([{
            "id": task_id_counter, "task": name, "priority": pr, "status": "Pending",
            "created_at": pd.Timestamp.now(), "completed_at": pd.NaT, "deadline": dl_ts,
            "ai_prediction": round(prob*100,2), "category": cat, "tags": tags,
            "habit_id": np.nan, "recurrence": "none", "assigned_to": current_user,
            "created_by": current_user, "shared": False
        }])], ignore_index=True)
        df_tasks.to_csv(DATA_FILE, index=False)
        task_id_counter += 1
        print(f"{GREEN}✅ Added '{name}' | AI: {round(prob*100,2)}%{RESET}")

def view_tasks():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{GREEN}✅ No tasks{RESET}")
        return
    print(f"\n{MAGENTA}Your Tasks{RESET}\n{'='*60}")
    for _, t in my.iterrows():
        tid = int(t["id"])
        stat = "✅" if t["status"] == "Completed" else "❌"
        pri = "⚡" if t["priority"] == "Y" else ""
        tleft, _ = time_left_parts(t["deadline"])
        dl = f"{RED}{tleft}{RESET}" if "OVERDUE" in tleft else tleft
        tags = format_tags(parse_tags(t["tags"]))
        print(f"#{tid} {t['task']} {pri}{tags}")
        print(f"  {get_category_display(t['category'])} | {stat} | {dl} | AI:{t['ai_prediction']:.0f}%")
    print("="*60)

def remove_task():
    global df_tasks
    view_tasks()
    if df_tasks.empty:
        return
    try:
        tid = int(input("Task ID to remove: "))
        if tid in df_tasks["id"].values:
            df_tasks = df_tasks[df_tasks["id"] != tid]
            df_tasks.to_csv(DATA_FILE, index=False)
            print(f"{GREEN}✅ Removed{RESET}")
        else:
            print(f"{RED}Not found{RESET}")
    except:
        print(f"{RED}Invalid ID{RESET}")

def mark_completed():
    global df_tasks, df_habits
    view_tasks()
    if df_tasks.empty:
        return
    try:
        tid = int(input("Task ID to complete: "))
        if tid not in df_tasks["id"].values:
            print(f"{RED}Not found{RESET}")
            return
        task = df_tasks[df_tasks["id"] == tid].iloc[0]
        if task["status"] == "Completed":
            print(f"{YELLOW}Already completed{RESET}")
            return
        df_tasks.loc[df_tasks["id"] == tid, "status"] = "Completed"
        df_tasks.loc[df_tasks["id"] == tid, "completed_at"] = pd.Timestamp.now()
        if not pd.isna(task["habit_id"]):
            hid = int(task["habit_id"])
            df_habits.loc[df_habits["habit_id"] == hid, "total_completions"] += 1
            df_habits.loc[df_habits["habit_id"] == hid, "last_completed"] = pd.Timestamp.now()
            df_habits.to_csv(HABITS_FILE, index=False)
            create_task_from_habit(hid, task["task"], task["recurrence"], task["category"])
            streak = calculate_streak(hid)
            print(f"{GREEN}✅ Completed! 🔥 {streak} day streak!{RESET}")
        else:
            print(f"{GREEN}✅ Completed{RESET}")
        df_tasks.to_csv(DATA_FILE, index=False)
    except:
        print(f"{RED}Invalid ID{RESET}")

def edit_task():
    global df_tasks
    view_tasks()
    try:
        tid = int(input("Task ID to edit: "))
        if tid not in df_tasks["id"].values:
            print(f"{RED}Not found{RESET}")
            return
        task = df_tasks[df_tasks["id"] == tid].iloc[0]
        name = input(f"New name (current: {task['task']}): ").strip()
        pr = input(f"Priority Y/N (current: {task['priority']}): ").strip().upper()
        dl = input(f"Deadline (current: {task['deadline']}): ").strip()
        if name:
            df_tasks.loc[df_tasks["id"] == tid, "task"] = name
        if pr in ["Y","N"]:
            df_tasks.loc[df_tasks["id"] == tid, "priority"] = pr
        if dl:
            df_tasks.loc[df_tasks["id"] == tid, "deadline"] = parse_deadline_input(dl)
        df_tasks.to_csv(DATA_FILE, index=False)
        print(f"{GREEN}✅ Updated{RESET}")
    except:
        print(f"{RED}Invalid input{RESET}")

def filter_tasks():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{GREEN}✅ No tasks{RESET}")
        return
    print("1.Pending 2.Completed 3.High Priority 4.Category")
    ch = input("Choose: ").strip()
    if ch == "1":
        filtered = my[my["status"] == "Pending"]
    elif ch == "2":
        filtered = my[my["status"] == "Completed"]
    elif ch == "3":
        filtered = my[my["priority"] == "Y"]
    elif ch == "4":
        cats = list(CATEGORIES.keys())
        for i,c in enumerate(cats,1):
            print(f"{i}. {c}")
        cc = input("Number: ").strip()
        cat = cats[int(cc)-1] if cc.isdigit() and 1<=int(cc)<=len(cats) else "Other"
        filtered = my[my["category"] == cat]
    else:
        print(f"{RED}Invalid{RESET}")
        return
    print(f"\n{MAGENTA}Filtered Tasks{RESET}\n{'='*50}")
    for _, t in filtered.iterrows():
        print(f"#{int(t['id'])} {t['task']} | {t['status']}")
    print("="*50)

def productivity_chart():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{YELLOW}No tasks{RESET}")
        return
    summary = my.groupby("category").agg({"status": ["count", lambda x: (x == "Completed").sum()]}).reset_index()
    summary.columns = ["Category", "Total", "Completed"]
    print(f"\n{MAGENTA}Productivity by Category{RESET}\n{'='*50}")
    for _, s in summary.iterrows():
        rate = round((s["Completed"] / s["Total"] * 100) if s["Total"] > 0 else 0, 1)
        print(f"{s['Category']}: {s['Total']} tasks | {s['Completed']} done | {rate}%")
    print("="*50)

def weekly_summary():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    start = now_dt() - timedelta(days=7)
    week = my[my["created_at"] >= start]
    total = len(week)
    comp = len(week[week["status"] == "Completed"])
    rate = round((comp / total * 100) if total > 0 else 0, 1)
    print(f"\n{MAGENTA}Weekly Summary{RESET}\n{'='*50}")
    print(f"Tasks: {total} | Completed: {comp} | Rate: {rate}%")
    print("="*50)

def top_days():
    my = df_tasks[(df_tasks["assigned_to"] == current_user) & (df_tasks["status"] == "Completed")]
    if my.empty:
        print(f"{YELLOW}No completed tasks{RESET}")
        return
    days = my.groupby(my["completed_at"].dt.date).size().reset_index(name="count")
    days = days.sort_values("count", ascending=False).head(5)
    print(f"\n{MAGENTA}Top 5 Days{RESET}\n{'='*50}")
    for _, d in days.iterrows():
        print(f"{d['completed_at']}: {d['count']} tasks")
    print("="*50)

def productivity_trend():
    my = df_tasks[(df_tasks["assigned_to"] == current_user) & (df_tasks["status"] == "Completed")]
    if my.empty:
        print(f"{YELLOW}No completed tasks{RESET}")
        return
    days = my.groupby(my["completed_at"].dt.date).size().reset_index(name="count")
    print(f"\n{MAGENTA}Daily Trend{RESET}\n{'='*50}")
    for _, d in days.iterrows():
        print(f"{d['completed_at']}: {d['count']} tasks")
    print("="*50)

def smart_trend():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{YELLOW}No tasks{RESET}")
        return
    total = len(my)
    comp = len(my[my["status"] == "Completed"])
    rate = round((comp / total * 100) if total > 0 else 0, 1)
    print(f"\n{MAGENTA}Smart Analysis{RESET}\n{'='*50}")
    print(f"Overall completion rate: {rate}%")
    if rate >= 70:
        print(f"{GREEN}Excellent productivity!{RESET}")
    elif rate >= 50:
        print(f"{YELLOW}Good progress, keep it up!{RESET}")
    else:
        print(f"{RED}Focus on completing more tasks{RESET}")
    print("="*50)

def daily_ai_plan():
    my = df_tasks[(df_tasks["assigned_to"] == current_user) & (df_tasks["status"] == "Pending")]
    if my.empty:
        print(f"{GREEN}✅ No pending tasks{RESET}")
        return
    my = my.sort_values("ai_prediction", ascending=False)
    print(f"\n{MAGENTA}Daily AI Plan{RESET}\n{'='*50}")
    print("Top 5 tasks to focus on:")
    for _, t in my.head(5).iterrows():
        tleft, _ = time_left_parts(t["deadline"])
        print(f"#{int(t['id'])} {t['task']} | AI:{t['ai_prediction']:.0f}% | {tleft}")
    print("="*50)

def view_by_category():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{GREEN}✅ No tasks{RESET}")
        return
    print(f"\n{MAGENTA}Tasks by Category{RESET}\n{'='*50}")
    for cat in CATEGORIES:
        cat_tasks = my[my["category"] == cat]
        if not cat_tasks.empty:
            print(f"\n{cat}:")
            for _, t in cat_tasks.iterrows():
                stat = "✅" if t["status"] == "Completed" else "❌"
                print(f"  #{int(t['id'])} {t['task']} {stat}")
    print("="*50)

def search_by_tags():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{GREEN}✅ No tasks{RESET}")
        return
    tag = input("Tag to search: ").strip().lower()
    filtered = my[my["tags"].apply(lambda x: tag in parse_tags(x))]
    if filtered.empty:
        print(f"{YELLOW}No tasks with tag '{tag}'{RESET}")
        return
    print(f"\n{MAGENTA}Tasks with '{tag}'{RESET}\n{'='*50}")
    for _, t in filtered.iterrows():
        print(f"#{int(t['id'])} {t['task']}")
    print("="*50)

def category_summary():
    my = df_tasks[df_tasks["assigned_to"] == current_user]
    if my.empty:
        print(f"{YELLOW}No tasks{RESET}")
        return
    summary = my.groupby("category").agg({"status": ["count", lambda x: (x == "Completed").sum()]}).reset_index()
    summary.columns = ["Category", "Total", "Completed"]
    print(f"\n{MAGENTA}Category Summary{RESET}\n{'='*50}")
    for _, s in summary.iterrows():
        rate = round((s["Completed"] / s["Total"] * 100) if s["Total"] > 0 else 0, 1)
        print(f"{s['Category']}: {s['Total']} total | {s['Completed']} done | {rate}%")
    print("="*50)

# ---------- MENU ----------
def show_menu():
    rc = reminder_queue.qsize()
    badge = f" {RED}[{rc}!]{RESET}" if rc > 0 else ""
    mt = len(df_tasks[df_tasks["assigned_to"] == current_user])
    print(f"\n{ORANGE}{'━'*60}{RESET}")
    print(f"{ORANGE}{BOLD}{current_user}{RESET} - {mt} tasks{badge}")
    print(f"{ORANGE}{'━'*60}{RESET}")
    
    menu = [
        "1.Add Task", "2.View Tasks", "3.Remove", "4.Complete", "5.Chart",
        "6.Weekly", "7.Top Days", "8.Edit", "9.Filter", "10.Trend",
        "11.Smart Trend", "12.AI Plan", "13.Save & Exit", "14.Reminders",
        "15.By Category", "16.Search Tags", "17.Cat Summary", "18.Create Habit",
        "19.View Habits", "20.Habit Dash", "21.Add Member", "22.View Team",
        "23.Assign", "24.My Assigned", "25.Comment", "26.Team Dash",
        "27.Train AI", "28.Export CSV", "29.Export Excel", "30.Gen PDF", 
        "31.Email", "32.📁 List Files"
    ]
    
    for i in range(0, len(menu), 4):
        print("  " + " | ".join(menu[i:i+4]))
    
    print(f"{ORANGE}{'━'*60}{RESET}")

# ---------- MAIN ----------
start_reminder_system()

while True:
    show_menu()
    choice = input(f"\n{CYAN}Choose: {RESET}").strip()
    
    try:
        if choice == "1":
            add_task()
        elif choice == "2":
            view_tasks()
            input("\nPress Enter...")
        elif choice == "3":
            remove_task()
            input("\nPress Enter...")
        elif choice == "4":
            mark_completed()
            input("\nPress Enter...")
        elif choice == "5":
            productivity_chart()
            input("\nPress Enter...")
        elif choice == "6":
            weekly_summary()
            input("\nPress Enter...")
        elif choice == "7":
            top_days()
            input("\nPress Enter...")
        elif choice == "8":
            edit_task()
            input("\nPress Enter...")
        elif choice == "9":
            filter_tasks()
            input("\nPress Enter...")
        elif choice == "10":
            productivity_trend()
            input("\nPress Enter...")
        elif choice == "11":
            smart_trend()
            input("\nPress Enter...")
        elif choice == "12":
            daily_ai_plan()
            input("\nPress Enter...")
        elif choice == "13":
            df_tasks.to_csv(DATA_FILE, index=False)
            df_habits.to_csv(HABITS_FILE, index=False)
            df_users.to_csv(USERS_FILE, index=False)
            df_comments.to_csv(COMMENTS_FILE, index=False)
            stop_reminder_system()
            print(f"\n{GREEN}{'='*60}{RESET}")
            print(f"{GREEN}✅ Saved! Goodbye {current_user}!{RESET}")
            print(f"{GREEN}{'='*60}{RESET}\n")
            break
        elif choice == "14":
            show_pending_reminders()
            input("\nPress Enter...")
        elif choice == "15":
            view_by_category()
            input("\nPress Enter...")
        elif choice == "16":
            search_by_tags()
            input("\nPress Enter...")
        elif choice == "17":
            category_summary()
            input("\nPress Enter...")
        elif choice == "18":
            create_recurring_task()
            input("\nPress Enter...")
        elif choice == "19":
            view_all_habits()
            input("\nPress Enter...")
        elif choice == "20":
            habit_dashboard()
            input("\nPress Enter...")
        elif choice == "21":
            add_team_member()
            input("\nPress Enter...")
        elif choice == "22":
            view_team_members()
            input("\nPress Enter...")
        elif choice == "23":
            assign_task()
            input("\nPress Enter...")
        elif choice == "24":
            view_tasks()
            input("\nPress Enter...")
        elif choice == "25":
            add_comment()
            input("\nPress Enter...")
        elif choice == "26":
            team_dashboard()
            input("\nPress Enter...")
        elif choice == "27":
            train_nn_model()
            input("\nPress Enter...")
        elif choice == "28":
            export_to_csv()
            input("\nPress Enter...")
        elif choice == "29":
            export_to_excel()
            input("\nPress Enter...")
        elif choice == "30":
            generate_pdf_report()
            input("\nPress Enter...")
        elif choice == "31":
            email_summary()
            input("\nPress Enter...")
        elif choice == "32":
            list_export_files()
            input("\nPress Enter...")
        else:
            print(f"{RED}Invalid option{RESET}")
            input("\nPress Enter...")
    
    except KeyboardInterrupt:
        df_tasks.to_csv(DATA_FILE, index=False)
        df_habits.to_csv(HABITS_FILE, index=False)
        df_users.to_csv(USERS_FILE, index=False)
        df_comments.to_csv(COMMENTS_FILE, index=False)
        stop_reminder_system()
        print(f"\n{GREEN}✅ Saved! Goodbye!{RESET}\n")
        break
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        input("\nPress Enter...")

print(f"\n{CYAN}Thank you for using Assan!{RESET}")
print(f"{YELLOW}Your data is saved in CSV files.{RESET}\n")
