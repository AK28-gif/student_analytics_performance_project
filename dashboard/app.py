import os
import getpass
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Student Academic Analytics (PostgreSQL)", layout="wide")

@st.cache_data
def load_data_from_postgres():
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "student_db")
    db_user = os.getenv("DB_USER") or getpass.getuser()
    db_password = os.getenv("DB_PASSWORD", "")
    
    conn_params = {
        "host": db_host,
        "port": db_port,
        "dbname": db_name,
        "user": db_user
    }
    if db_password:
        conn_params["password"] = db_password

    # Use native psycopg2 connection to avoid SQLAlchemy cursor errors
    conn = psycopg2.connect(**conn_params)
    query = 'SELECT * FROM "fact_student_analytics";'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df = load_data_from_postgres()

st.title("🎓 Student Performance & Academic Risk Dashboard")
st.caption("Connected to Live PostgreSQL Analytical Data Mart")

# Sidebar Filters
st.sidebar.header("Filter Criteria")
school_choice = st.sidebar.multiselect("Select School", options=df['school'].unique(), default=df['school'].unique())
risk_filter = st.sidebar.multiselect("Risk Category", options=df['risk_level'].unique(), default=df['risk_level'].unique())

filtered_df = df[(df['school'].isin(school_choice)) & (df['risk_level'].isin(risk_filter))]

# KPI Summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tracked Students", len(filtered_df))
col2.metric("Pass Rate", f"{(len(filtered_df[filtered_df['pass_status'] == 'Pass']) / max(len(filtered_df), 1) * 100):.1f}%")
col3.metric("Avg Attendance", f"{filtered_df['attendance_pct'].mean():.1f}%")
col4.metric("Students At Risk", len(filtered_df[filtered_df['risk_level'] == 'High Risk']), delta_color="inverse")

st.divider()

# Charts
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.subheader("1. Attendance vs. Final Marks")
    fig_scatter = px.scatter(
        filtered_df, x="attendance_pct", y="final_grade",
        color="risk_level", hover_data=["student_id"],
        labels={"attendance_pct": "Attendance (%)", "final_grade": "Final Mark (0-20)"},
        color_discrete_map={"High Risk": "#e74c3c", "Normal": "#2ecc71"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with row1_col2:
    st.subheader("2. Overall Pass / Fail Ratio")
    fig_pie = px.pie(
        filtered_df, names="pass_status",
        color="pass_status",
        color_discrete_map={"Pass": "#27ae60", "Fail": "#e74c3c"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.subheader("3. Average Final Grade by School")
    school_grp = filtered_df.groupby('school')['final_grade'].mean().reset_index()
    fig_bar = px.bar(school_grp, x="school", y="final_grade", color="school")
    st.plotly_chart(fig_bar, use_container_width=True)

with row2_col2:
    st.subheader("4. Study Time vs Final Grade Distribution")
    fig_box = px.box(filtered_df, x="studytime", y="final_grade")
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

st.subheader("5. Early Academic Intervention List (High-Risk Students)")
st.dataframe(filtered_df[filtered_df['risk_level'] == 'High Risk'][
    ['student_id', 'school', 'attendance_pct', 'avg_internal_marks', 'final_grade', 'risk_level']
], use_container_width=True)

st.divider()
st.subheader("6. Live Model Inference & Risk Prediction (MLOps Integration)")

with st.expander("🔮 Predict Academic Risk for a New Student"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        in_study = st.selectbox("Daily Study Time", [1, 2, 3, 4], format_func=lambda x: f"{x}hr")
        in_age = st.slider("Age", 15, 22, 17)
    with c2:
        in_failures = st.selectbox("Past Class Failures", [0, 1, 2, 3, 4])
        in_absences = st.number_input("Absences Count", min_value=0, max_value=93, value=4)
    with c3:
        in_attendance = round(((93 - in_absences) / 93) * 100, 2)
        st.metric("Computed Attendance %", f"{in_attendance}%")
        in_medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
    with c4:
        in_internals = st.slider("Average Internal Score (0-20)", 0.0, 20.0, 11.5)
        in_fedu = st.slider("Father's Education (0-4)", 0, 4, 2)

    if st.button("Run ML Prediction"):
        import joblib
        model_p = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"))
        sample = pd.DataFrame([{
            'studytime': in_study, 'failures': in_failures, 'absences': in_absences,
            'attendance_pct': in_attendance, 'avg_internal_marks': in_internals,
            'age': in_age, 'medu': in_medu, 'fedu': in_fedu
        }])
        prob = model_p.predict_proba(sample)[0][1]
        if prob >= 0.5:
            st.error(f"🚨 At Risk of Dropout / Academic Failure (Risk Probability: {prob*100:.1f}%)")
        else:
            st.success(f"✅ Normal Performance Expected (Risk Probability: {prob*100:.1f}%)")