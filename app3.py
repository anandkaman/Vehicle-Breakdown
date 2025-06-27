import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import os
import altair as alt

# ------------------ Configuration ------------------
st.set_page_config(page_title="Vehicle Monitoring", layout="wide")

# Custom styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(19, 99, 246, 0.92); /* Cyan with 90% opacity */
    }
    .main {
        background-color: #f9fbfd;
    }
    h1, h2, h3, h4 {
        color: #1a73e8;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load("vehicle_breakdown_model.pkl")
base_df = pd.read_csv("cleaned_obd_data_filtered.csv")
base_df['Timestamp'] = pd.to_datetime(base_df['Timestamp'])
os.makedirs("logs", exist_ok=True)

# Vehicles and drivers
vehicle_ids = ["KA-01", "KA-02", "KA-03", "KA-04", "KA-05"]
driver_mapping = {
    "KA-01": "Rishab",
    "KA-02": "Priya",
    "KA-03": "Arjun",
    "KA-04": "Sneha",
    "KA-05": "Ravi"
}
features = [
    'Engine_Coolant_Temperature', 'Intake_Manifold_Abs_Pressure', 'Engine_RPM',
    'Vehicle_Speed', 'Intake_Air_Temperature', 'AirFlow_Rate', 'Throttle_Position',
    'Air_Temperature', 'Acc_Pedal_Pos_D', 'Acc_Pedal_Pos_E'
]

# Sidebar
page = st.sidebar.radio("Navigate", ["Dashboard", "Live Monitoring"], index=1)
vehicle_selected = st.sidebar.selectbox("Select Vehicle", vehicle_ids)
driver_name = driver_mapping[vehicle_selected]
st.sidebar.write(f"Driver: **{driver_name}**")

# Cause explanation logic
def get_causes(row):
    factor = 70
    causes = []
    if row['Vehicle_Speed'] > 1:
        if row['Engine_Coolant_Temperature'] > 100:
            causes.append("Overheating (Coolant Temp > 100°C)")
        if row['Throttle_Position'] < 70 and row['Engine_RPM'] < row['Throttle_Position'] * factor:
            causes.append("RPM too low for throttle → possible stall")
        if row['Intake_Manifold_Abs_Pressure'] > 220:
            causes.append("High manifold pressure → possible blockage")
        if row['Intake_Air_Temperature'] > 120 or row['Intake_Air_Temperature'] < -20:
            causes.append("Abnormal intake air temperature")
    if row['Acc_Pedal_Pos_D'] > 30 and row['Vehicle_Speed'] < 5:
        causes.append("Pedal pressed but vehicle not moving → possible gear issue")
    return causes

# Charting utility
def plot_line_chart(data, column, title):
    chart = alt.Chart(data).mark_line(color='#60a5fa').encode(
        x='Timestamp:T',
        y=alt.Y(column, title=title)
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

# ------------------- DASHBOARD -------------------
if page == "Dashboard":
    st.title("Vehicle Dashboard")
    folder = f"logs/{vehicle_selected}"
    if not os.path.exists(folder):
        st.warning("No logs found for this vehicle.")
    else:
        files = sorted(os.listdir(folder), reverse=True)
        if files:
            latest_file = os.path.join(folder, files[0])
            log_df = pd.read_csv(latest_file)
            st.success(f"Loaded log: {latest_file}")
            log_df['Timestamp'] = pd.to_datetime(log_df['timestamp'])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", log_df.shape[0])
            col2.metric("Breakdown Events", log_df['prediction'].sum())
            col3.metric("Avg. Speed", round(log_df['Vehicle_Speed'].mean(), 2))
            col4.metric("Avg. RPM", round(log_df['Engine_RPM'].mean(), 2))

            st.markdown("---")
            st.subheader("RPM Over Time")
            plot_line_chart(log_df, 'Engine_RPM', 'Engine RPM')

            st.markdown("---")
            st.subheader("Coolant Temperature")
            plot_line_chart(log_df, 'Engine_Coolant_Temperature', 'Coolant Temp')

            st.markdown("---")
            st.subheader("Speed Over Time")
            plot_line_chart(log_df, 'Vehicle_Speed', 'Vehicle Speed')

        else:
            st.warning("No log files found.")

# ------------------- LIVE MONITORING -------------------
elif page == "Live Monitoring":
    st.title("Live Vehicle Monitoring")

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'log_data' not in st.session_state:
        st.session_state.log_data = []

    if not st.session_state.running:
        if st.button("Start Simulation"):
            st.session_state.running = True
            st.session_state.log_data = []
            st.rerun()
    else:
        if st.button("Stop Simulation"):
            st.session_state.running = False
            if st.session_state.log_data:
                log_df = pd.DataFrame(st.session_state.log_data)
                today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_folder = f"logs/{vehicle_selected}"
                os.makedirs(log_folder, exist_ok=True)
                log_path = os.path.join(log_folder, f"{today}_log.csv")
                log_df.to_csv(log_path, index=False)
                st.success(f"Simulation stopped. Log saved to: {log_path}")
            st.rerun()

    df = base_df.head(15000)

    if st.session_state.running:
        st.success("Simulation is running...")
        placeholder = st.empty()

        for i, row in df.iterrows():
            if not st.session_state.running:
                break

            input_data = row[features].to_frame().T
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            causes = get_causes(row)
            main_cause = causes[0] if causes else "None"

            row_log = row.to_dict()
            row_log['timestamp'] = datetime.now()
            row_log['vehicle_id'] = vehicle_selected
            row_log['driver'] = driver_name
            row_log['prediction'] = int(pred)
            row_log['probability'] = round(prob, 4)
            row_log['breakdown_cause'] = main_cause
            st.session_state.log_data.append(row_log)

            with placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Engine RPM", f"{row['Engine_RPM']} RPM")
                col2.metric("Coolant Temp", f"{row['Engine_Coolant_Temperature']} °C")
                col3.metric("Speed", f"{row['Vehicle_Speed']} km/h")
                col4.metric("Prediction", "Breakdown" if pred == 1 else "Normal", delta=f"{prob*100:.2f}%")

                if pred == 1 or causes:
                    st.warning("Breakdown Warning")
                    for c in causes:
                        st.write(f"- {c}")
                else:
                    st.info("All systems normal.")

            time.sleep(0.001)

        if st.session_state.running:
            st.session_state.running = False
            if st.session_state.log_data:
                log_df = pd.DataFrame(st.session_state.log_data)
                today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_folder = f"logs/{vehicle_selected}"
                os.makedirs(log_folder, exist_ok=True)
                log_path = os.path.join(log_folder, f"{today}_log.csv")
                log_df.to_csv(log_path, index=False)
                st.success(f"Simulation completed. Log saved to: {log_path}")
