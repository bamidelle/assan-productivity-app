# assan1_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ================================
# Page configuration
# ================================
st.set_page_config(
    page_title="Assan Productivity App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Custom CSS for gradients, fonts, text color
# ================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
    }

    .stButton>button {
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        color: white;
        font-size: 16px;
        font-weight: bold;
    }

    .stTextInput>div>input {
        background-color: rgba(255,255,255,0.1);
        color: white;
        font-size: 16px;
    }

    .stSelectbox>div>div>div>span {
        color: white;
    }

    .stDataFrame table {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# App Header
# ================================
st.title("Assan Productivity App 🚀")
st.subheader("Track tasks, predict productivity, and manage your day efficiently.")

# ================================
# Sidebar for User Input
# ================================
st.sidebar.header("User Info")
user_name = st.sidebar.text_input("Enter your name:")

st.sidebar.header("Add a Task")
task_name = st.sidebar.text_input("Task name")
task_hours = st.sidebar.number_input("Estimated hours", min_value=0, max_value=24, step=1)
task_priority = st.sidebar.selectbox("Priority", ["Low", "Medium", "High"])
add_task_btn = st.sidebar.button("Add Task")

# ================================
# Session state for tasks
# ================================
if "tasks" not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=["Task", "Hours", "Priority"])

# ================================
# Add task logic
# ================================
if add_task_btn and task_name:
    st.session_state.tasks = pd.concat(
        [st.session_state.tasks, pd.DataFrame({"Task":[task_name], "Hours":[task_hours], "Priority":[task_priority]})],
        ignore_index=True
    )

# ================================
# Display tasks
# ================================
st.subheader(f"{user_name}'s Task List" if user_name else "Task List")
st.dataframe(st.session_state.tasks)

# ================================
# Productivity prediction (simple ML)
# ================================
st.subheader("Predict Your Productivity 🚀")
if not st.session_state.tasks.empty:
    df = st.session_state.tasks.copy()
    # Encode priority
    priority_map = {"Low": 0, "Medium": 1, "High": 2}
    df["PriorityCode"] = df["Priority"].map(priority_map)

    X = df[["Hours", "PriorityCode"]]
    y = df["Hours"] * np.random.uniform(0.8, 1.2, size=len(df))  # Simulated productivity

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write("Predicted productivity hours for tasks (simple ML simulation):")
    df["PredictedHours"] = model.predict(X)
    st.dataframe(df[["Task", "Hours", "PredictedHours", "Priority"]])
    st.write(f"Model mean squared error: {mse:.2f}")
else:
    st.info("Add tasks to see productivity predictions.")

# ================================
# Footer
# ================================
st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
st.caption("Assan Productivity App | Designed with Streamlit 🚀")
