# ============================================================
# FOCUSGUARD AI
# Student Performance Intelligence System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FocusGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(168,85,247,0.14),
            transparent 25%
        ),
        #070914;

    color: #f8fafc;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 4rem;
    max-width: 1450px;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: rgba(8, 10, 24, 0.98);
    border-right: 1px solid rgba(255,255,255,0.08);
}


/* =========================================================
   COMPACT HEADER
   ========================================================= */

.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -1.3px;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #818cf8,
        #c084fc,
        #f472b6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
    margin-bottom: 1.2rem;
}


/* =========================================================
   GLASS CARD
   ========================================================= */

.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1.5rem;

    backdrop-filter: blur(15px);

    box-shadow:
        0 15px 50px rgba(0,0,0,0.25);
}


/* =========================================================
   KPI
   ========================================================= */

.kpi-label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.4rem;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 1.8rem;
    margin-bottom: 1rem;
}


/* =========================================================
   PREDICTION
   ========================================================= */

.prediction-box {
    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.17),
            rgba(168,85,247,0.10)
        );

    border: 1px solid rgba(129,140,248,0.35);
    border-radius: 26px;

    padding: 1.8rem;
    text-align: center;

    box-shadow:
        0 20px 70px rgba(79,70,229,0.14);
}

.prediction-value {
    font-size: 4.2rem;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #818cf8,
        #c084fc,
        #f472b6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


/* =========================================================
   INSIGHT
   ========================================================= */

.insight-card {
    background: rgba(129,140,248,0.07);
    border: 1px solid rgba(129,140,248,0.20);

    border-radius: 18px;

    padding: 1.25rem;

    margin-bottom: 0.7rem;
}


/* =========================================================
   IMPROVEMENT CARD
   ========================================================= */

.improvement-card {
    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.08),
            rgba(168,85,247,0.05)
        );

    border: 1px solid rgba(129,140,248,0.18);

    border-radius: 18px;

    padding: 1.25rem;

    margin-bottom: 0.8rem;
}

.improvement-title {
    font-size: 1rem;
    font-weight: 750;
    margin-bottom: 0.35rem;
}

.improvement-text {
    color: #94a3b8;
    line-height: 1.65;
    font-size: 0.9rem;
}


/* =========================================================
   DAY PLAN
   ========================================================= */

.day-number {
    color: #a5b4fc;
    font-weight: 800;
    font-size: 0.9rem;
}

.day-title {
    font-weight: 750;
    font-size: 0.95rem;
}

.day-description {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 3px;
}


/* =========================================================
   BADGE
   ========================================================= */

.badge {
    display: inline-block;

    padding: 0.35rem 0.75rem;

    border-radius: 999px;

    background: rgba(129,140,248,0.12);
    border: 1px solid rgba(129,140,248,0.25);

    color: #c7d2fe;

    font-size: 0.78rem;

    margin-right: 0.4rem;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    border-radius: 13px;

    border: 1px solid rgba(129,140,248,0.3);

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );

    color: white;

    font-weight: 700;

    padding: 0.75rem;

    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #64748b;

    padding-top: 3rem;

    font-size: 0.8rem;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "focusguard_gpa_model.pkl"
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "student_productivity.csv"
    )


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

@st.cache_data
def load_model_info():

    try:

        with open(
            "model_info.json",
            "r"
        ) as file:

            return json.load(file)

    except FileNotFoundError:

        return {
            "model_name": "Unknown",
            "mae": 0,
            "rmse": 0,
            "r2": 0,
            "training_samples": 0,
            "testing_samples": 0,
            "target": "GPA"
        }


# ============================================================
# LOAD EVERYTHING
# ============================================================

model = load_model()
df = load_data()
model_info = load_model_info()


# ============================================================
# HELPER FUNCTION
# ============================================================

def total_daily_hours(
    study,
    extra,
    sleep,
    social,
    physical
):

    return (
        study
        + extra
        + sleep
        + social
        + physical
    )


# ============================================================
# GPA PREDICTION
# ============================================================

def predict_gpa(
    study,
    extra,
    sleep,
    social,
    physical,
    stress
):

    total_hours = total_daily_hours(
        study,
        extra,
        sleep,
        social,
        physical
    )

    productive_hours = (
        study + extra
    )

    free_time = (
        24 - total_hours
    )

    input_data = pd.DataFrame({

        "Study_Hours_Per_Day": [
            study
        ],

        "Extracurricular_Hours_Per_Day": [
            extra
        ],

        "Sleep_Hours_Per_Day": [
            sleep
        ],

        "Social_Hours_Per_Day": [
            social
        ],

        "Physical_Activity_Hours_Per_Day": [
            physical
        ],

        "Stress_Level": [
            stress
        ],

        "Total_Tracked_Hours": [
            total_hours
        ],

        "Productive_Hours_Per_Day": [
            productive_hours
        ],

        "Free_Time_Per_Day": [
            free_time
        ]
    })

    prediction = model.predict(
        input_data
    )[0]

    return float(
        np.clip(
            prediction,
            0,
            10
        )
    )


# ============================================================
# GPA STATUS
# ============================================================

def get_status(gpa):

    if gpa >= 8.5:

        return (
            "Excellent Academic Profile",
            "🟢"
        )

    elif gpa >= 7.0:

        return (
            "Strong Academic Profile",
            "🔵"
        )

    elif gpa >= 5.5:

        return (
            "Improvement Opportunity",
            "🟡"
        )

    else:

        return (
            "Needs Attention",
            "🔴"
        )


# ============================================================
# GPA GAUGE
# ============================================================

def create_gauge(gpa):

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=gpa,

            number={
                "font": {
                    "size": 48
                },
                "suffix": " / 10"
            },

            title={
                "text": "Predicted GPA"
            },

            gauge={

                "axis": {
                    "range": [0, 10]
                },

                "bar": {
                    "thickness": 0.75
                },

                "steps": [

                    {
                        "range": [0, 5.5]
                    },

                    {
                        "range": [5.5, 7]
                    },

                    {
                        "range": [7, 8.5]
                    },

                    {
                        "range": [8.5, 10]
                    }
                ],

                "threshold": {

                    "line": {
                        "width": 4
                    },

                    "thickness": 0.8,

                    "value": gpa
                }
            }
        )
    )

    fig.update_layout(

        template="plotly_dark",

        height=330,

        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    return fig


# ============================================================
# ROUTINE RADAR
# ============================================================

def create_radar(
    study,
    sleep,
    social,
    physical,
    extra
):

    values = [

        min(
            study / 12 * 100,
            100
        ),

        min(
            sleep / 12 * 100,
            100
        ),

        min(
            social / 10 * 100,
            100
        ),

        min(
            physical / 8 * 100,
            100
        ),

        min(
            extra / 8 * 100,
            100
        )
    ]

    categories = [

        "Study",
        "Sleep",
        "Social",
        "Exercise",
        "Extra"
    ]

    fig = go.Figure()

    fig.add_trace(

        go.Scatterpolar(

            r=values + [values[0]],

            theta=categories + [categories[0]],

            fill="toself",

            name="Student Routine"
        )
    )

    fig.update_layout(

        template="plotly_dark",

        polar={

            "radialaxis": {

                "visible": True,

                "range": [
                    0,
                    100
                ]
            }
        },

        showlegend=False,

        height=430
    )

    return fig


# ============================================================
# SMART ROUTINE INSIGHTS
# ============================================================

def generate_insights(
    study,
    sleep,
    social,
    physical,
    extra,
    stress,
    gpa
):

    insights = []


    if study < 3:

        insights.append(
            "📚 Your study time is relatively low. Increasing focused study time may be worth exploring."
        )

    elif study >= 6:

        insights.append(
            "📚 Your study commitment is strong. Focus on maintaining consistency and study quality."
        )


    if sleep < 6:

        insights.append(
            "😴 Your reported sleep duration is low. Consider maintaining a more consistent rest schedule."
        )

    elif sleep >= 7:

        insights.append(
            "😴 Your reported sleep duration is in a reasonable range."
        )


    if social > 5:

        insights.append(
            "👥 A large portion of your day is allocated to social activity. Protecting focused study blocks may help with time management."
        )


    if physical < 1:

        insights.append(
            "🏃 Physical activity is relatively low in your current routine. Consider adding regular movement."
        )


    if stress == "High":

        insights.append(
            "🧠 You reported high stress. Breaking large academic tasks into smaller steps may make your workload easier to manage."
        )

    elif stress == "Medium":

        insights.append(
            "🧠 Your reported stress is moderate. Planning tasks in smaller blocks may help you stay organized."
        )

    elif stress == "Low":

        insights.append(
            "🧠 Your reported stress level is low."
        )


    if not insights:

        insights.append(
            "✨ Your routine does not show a major warning signal from the simple rule-based checks."
        )


    return insights


# ============================================================
# PERSONALIZED IMPROVEMENT PLAN
# ============================================================

def generate_improvement_plan(
    study,
    sleep,
    social,
    physical,
    extra,
    stress,
    gpa
):

    tips = []


    # --------------------------------------------------------
    # STUDY
    # --------------------------------------------------------

    if study < 3:

        tips.append({
            "icon": "📚",
            "title": "Study Focus",
            "text":
                "Your current study time is relatively low. "
                "Try adding one focused study session to your daily routine."
        })

    elif study < 5:

        tips.append({
            "icon": "📚",
            "title": "Study Consistency",
            "text":
                "Build consistency with a fixed daily study schedule "
                "and gradually improve the quality of your focused sessions."
        })

    else:

        tips.append({
            "icon": "📚",
            "title": "Study Strength",
            "text":
                "Your study commitment is strong. "
                "Focus on consistency, revision and active practice."
        })


    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    if sleep < 6:

        tips.append({
            "icon": "😴",
            "title": "Sleep Routine",
            "text":
                "Your reported sleep duration is low. "
                "Try to maintain a more consistent sleep schedule and allow enough time for rest."
        })

    elif sleep >= 7:

        tips.append({
            "icon": "😴",
            "title": "Sleep Routine",
            "text":
                "Your reported sleep duration is in a reasonable range. "
                "Try to keep your sleep schedule consistent."
        })


    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    if stress == "High":

        tips.append({
            "icon": "🧠",
            "title": "Stress Management",
            "text":
                "You reported high stress. "
                "Break difficult academic tasks into smaller goals and use short recovery breaks."
        })

    elif stress == "Medium":

        tips.append({
            "icon": "🧠",
            "title": "Stress Management",
            "text":
                "Your stress level is moderate. "
                "Planning your workload in smaller blocks may make studying easier to manage."
        })


    # --------------------------------------------------------
    # PHYSICAL ACTIVITY
    # --------------------------------------------------------

    if physical < 1:

        tips.append({
            "icon": "🏃",
            "title": "Physical Activity",
            "text":
                "Your current physical activity is relatively low. "
                "Consider adding some regular movement to your daily routine."
        })


    # --------------------------------------------------------
    # SOCIAL TIME
    # --------------------------------------------------------

    if social > 5:

        tips.append({
            "icon": "📱",
            "title": "Time Management",
            "text":
                "A large part of your day is allocated to social activity. "
                "Consider protecting specific hours for focused academic work."
        })


    # --------------------------------------------------------
    # GPA STRATEGY
    # --------------------------------------------------------

    if gpa < 5.5:

        tips.append({
            "icon": "🚀",
            "title": "Recovery Strategy",
            "text":
                "Do not try to change everything at once. "
                "Start with one or two manageable habits, stay consistent, and review your progress each week."
        })

    elif gpa < 7:

        tips.append({
            "icon": "🎯",
            "title": "Improvement Strategy",
            "text":
                "Your predicted profile shows room for improvement. "
                "Focus on consistency, planned study sessions and reducing avoidable distractions."
        })

    else:

        tips.append({
            "icon": "🏆",
            "title": "Maintain Progress",
            "text":
                "Your predicted performance is relatively strong. "
                "Focus on maintaining effective habits instead of overloading your routine."
        })


    return tips


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center">

            <div style="
                font-size:2.7rem;
                margin-bottom:0.3rem;
            ">
                🧠
            </div>

            <div style="
                font-size:1.35rem;
                font-weight:800;
            ">
                FocusGuard
            </div>

            <div style="
                color:#94a3b8;
                font-size:0.75rem;
            ">
                AI Student Intelligence
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🎯 GPA Predictor",
            "🧪 What-If Lab",
            "📊 Student Analytics",
            "🔬 AI Explanation",
            "🤖 Model Intelligence"
        ]
    )

    st.divider()

    st.markdown(
        "### 🤖 Current Model"
    )

    st.caption(
        model_info["model_name"]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "R²",
            f'{model_info["r2"]:.3f}'
        )

    with col2:

        st.metric(
            "RMSE",
            f'{model_info["rmse"]:.3f}'
        )

    st.divider()

    st.caption(
        "FocusGuard AI"
    )

    st.caption(
        "Student Performance Intelligence"
    )


# ============================================================
# COMPACT GLOBAL HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-title">
        Focus<span class="gradient-text">Guard</span> AI
    </div>

    <div class="hero-subtitle">
        Student Performance Intelligence
        &nbsp;•&nbsp;
        Predictive Analytics
        &nbsp;•&nbsp;
        Personalized Insights
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">📡 System Overview</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    Students
                </div>

                <div class="kpi-value">
                    {len(df):,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    Average GPA
                </div>

                <div class="kpi-value">
                    {df["GPA"].mean():.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    Model R²
                </div>

                <div class="kpi-value">
                    {model_info["r2"]:.3f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    ML Model
                </div>

                <div class="kpi-value"
                     style="font-size:1.15rem">

                    {model_info["model_name"]}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 How FocusGuard Works</div>',
        unsafe_allow_html=True
    )


    flow1, flow2, flow3, flow4, flow5 = st.columns(5)


    with flow1:

        st.markdown(
            """
            <div class="glass-card">

                <h3>👤</h3>

                <b>Student Habits</b>

                <p style="color:#94a3b8">
                    Daily routine and lifestyle data
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with flow2:

        st.markdown(
            """
            <div class="glass-card">

                <h3>🧹</h3>

                <b>Data Processing</b>

                <p style="color:#94a3b8">
                    Cleaning and feature engineering
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with flow3:

        st.markdown(
            """
            <div class="glass-card">

                <h3>🤖</h3>

                <b>ML Model</b>

                <p style="color:#94a3b8">
                    Machine learning prediction
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with flow4:

        st.markdown(
            """
            <div class="glass-card">

                <h3>🎯</h3>

                <b>Prediction</b>

                <p style="color:#94a3b8">
                    Estimated academic performance
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with flow5:

        st.markdown(
            """
            <div class="glass-card">

                <h3>💡</h3>

                <b>Insights</b>

                <p style="color:#94a3b8">
                    Explore routine patterns
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown(
        '<div class="section-title">✨ Explore FocusGuard</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="glass-card">

                <h3>🎯 GPA Predictor</h3>

                <p style="color:#94a3b8">
                    Enter a student's routine and
                    generate an ML-based GPA estimate.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="glass-card">

                <h3>🧪 What-If Lab</h3>

                <p style="color:#94a3b8">
                    Compare a current routine with
                    an alternative simulated routine.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="glass-card">

                <h3>🔬 AI Explanation</h3>

                <p style="color:#94a3b8">
                    Explore which features the trained
                    model relies on most.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# GPA PREDICTOR
# ============================================================

elif page == "🎯 GPA Predictor":

    st.markdown(
        '<div class="section-title">🎯 GPA Predictor</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter a student's typical daily routine."
    )


    col1, col2 = st.columns(2)


    with col1:

        study_hours = st.slider(
            "📚 Study Hours / Day",
            0.0,
            12.0,
            5.0,
            0.5
        )

        sleep_hours = st.slider(
            "😴 Sleep Hours / Day",
            0.0,
            12.0,
            7.0,
            0.5
        )

        extracurricular_hours = st.slider(
            "🎨 Extracurricular Hours / Day",
            0.0,
            8.0,
            2.0,
            0.5
        )


    with col2:

        social_hours = st.slider(
            "👥 Social Hours / Day",
            0.0,
            10.0,
            2.0,
            0.5
        )

        physical_hours = st.slider(
            "🏃 Physical Activity / Day",
            0.0,
            8.0,
            1.0,
            0.5
        )

        stress_level = st.selectbox(
            "🧠 Stress Level",
            sorted(
                df["Stress_Level"]
                .dropna()
                .unique()
            )
        )


    current_total = total_daily_hours(
        study_hours,
        extracurricular_hours,
        sleep_hours,
        social_hours,
        physical_hours
    )


    if current_total > 24:

        st.error(
            f"⚠️ Your routine adds up to {current_total:.1f} hours/day. "
            "Please reduce the selected hours so the total is 24 hours or less."
        )

    else:

        st.caption(
            f"Daily tracked time: {current_total:.1f} / 24 hours"
        )


    st.markdown("")


    if st.button(
        "🚀 ANALYZE MY ROUTINE"
    ):

        if current_total > 24:

            st.error(
                "Please create a valid 24-hour daily routine first."
            )

        else:

            prediction = predict_gpa(
                study_hours,
                extracurricular_hours,
                sleep_hours,
                social_hours,
                physical_hours,
                stress_level
            )


            status, icon = get_status(
                prediction
            )


            # =================================================
            # PREDICTION + GAUGE
            # =================================================

            left, right = st.columns([1, 1])


            with left:

                st.markdown(
                    f"""
                    <div class="prediction-box">

                        <div class="kpi-label">
                            ML-BASED GPA ESTIMATE
                        </div>

                        <div class="prediction-value">
                            {prediction:.2f}
                        </div>

                        <h3>
                            {icon} {status}
                        </h3>

                        <p style="color:#94a3b8">
                            This is a model estimate based on
                            the provided routine and learned
                            patterns in the training data.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with right:

                st.plotly_chart(
                    create_gauge(
                        prediction
                    ),
                    use_container_width=True
                )


            # =================================================
            # SMART INSIGHTS
            # =================================================

            st.markdown(
                '<div class="section-title">🧠 Smart Routine Insights</div>',
                unsafe_allow_html=True
            )


            insights = generate_insights(
                study_hours,
                sleep_hours,
                social_hours,
                physical_hours,
                extracurricular_hours,
                stress_level,
                prediction
            )


            for insight in insights:

                st.markdown(
                    f"""
                    <div class="insight-card">
                        {insight}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # PERSONALIZED IMPROVEMENT PLAN
            # =================================================

            st.markdown(
                '<div class="section-title">🎯 Personalized Improvement Plan</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Suggestions are based on the routine entered above. "
                "They are general guidance, not guaranteed ways to increase GPA."
            )


            improvement_plan = generate_improvement_plan(
                study_hours,
                sleep_hours,
                social_hours,
                physical_hours,
                extracurricular_hours,
                stress_level,
                prediction
            )


            for tip in improvement_plan:

                st.markdown(
                    f"""
                    <div class="improvement-card">

                        <div class="improvement-title">
                            {tip["icon"]} {tip["title"]}
                        </div>

                        <div class="improvement-text">
                            {tip["text"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # 7-DAY ACTION PLAN
            # =================================================

            st.markdown(
                '<div class="section-title">📅 7-Day Action Plan</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "A simple plan for turning the insights into consistent habits."
            )


            action_plan = [

                (
                    "Day 1",
                    "📚",
                    "Focused Study",
                    "Complete one distraction-free study session."
                ),

                (
                    "Day 2",
                    "😴",
                    "Sleep Routine",
                    "Maintain a consistent sleep and wake schedule."
                ),

                (
                    "Day 3",
                    "🧠",
                    "Stress Management",
                    "Break one difficult academic task into smaller steps."
                ),

                (
                    "Day 4",
                    "🏃",
                    "Physical Activity",
                    "Include some regular movement in your day."
                ),

                (
                    "Day 5",
                    "📱",
                    "Distraction Control",
                    "Protect one study block from unnecessary distractions."
                ),

                (
                    "Day 6",
                    "🔄",
                    "Weekly Review",
                    "Review what worked and what made studying difficult."
                ),

                (
                    "Day 7",
                    "🎯",
                    "Next Week Planning",
                    "Set realistic study targets for the coming week."
                )
            ]


            for (
                day,
                icon,
                title,
                description
            ) in action_plan:

                st.markdown(
                    f"""
                    <div class="improvement-card">

                        <div style="
                            display:flex;
                            align-items:center;
                            gap:15px;
                        ">

                            <div class="day-number">
                                {day}
                            </div>

                            <div style="
                                font-size:1.35rem;
                            ">
                                {icon}
                            </div>

                            <div>

                                <div class="day-title">
                                    {title}
                                </div>

                                <div class="day-description">
                                    {description}
                                </div>

                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # ROUTINE RADAR
            # =================================================

            st.markdown(
                '<div class="section-title">📊 Your Routine Profile</div>',
                unsafe_allow_html=True
            )


            st.plotly_chart(
                create_radar(
                    study_hours,
                    sleep_hours,
                    social_hours,
                    physical_hours,
                    extracurricular_hours
                ),
                use_container_width=True
            )


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            report = f"""
FOCUSGUARD AI
Student Performance Intelligence Report
========================================

Predicted GPA:
{prediction:.2f}

Performance Status:
{icon} {status}


DAILY ROUTINE
-------------

Study Hours:
{study_hours}

Sleep Hours:
{sleep_hours}

Social Hours:
{social_hours}

Physical Activity:
{physical_hours}

Extracurricular Hours:
{extracurricular_hours}

Stress Level:
{stress_level}


MODEL
-----

Model:
{model_info["model_name"]}

R²:
{model_info["r2"]:.4f}

RMSE:
{model_info["rmse"]:.4f}

MAE:
{model_info["mae"]:.4f}


IMPORTANT
---------

This report contains an ML-based estimate.

It does not guarantee a student's future GPA.

Model associations should not be interpreted
as causal effects.
"""


            st.download_button(

                "📄 Download Prediction Report",

                report,

                file_name="focusguard_prediction_report.txt",

                mime="text/plain"
            )


# ============================================================
# WHAT-IF LAB
# ============================================================

elif page == "🧪 What-If Lab":

    st.markdown(
        '<div class="section-title">🧪 Before → After What-If Lab</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Compare two hypothetical routines and see how the model's estimate changes."
    )


    st.markdown(
        "### 👤 Current Routine"
    )


    c1, c2 = st.columns(2)


    with c1:

        base_study = st.slider(
            "📚 Study Hours",
            0.0,
            12.0,
            5.0,
            0.5,
            key="base_study"
        )

        base_sleep = st.slider(
            "😴 Sleep Hours",
            0.0,
            12.0,
            7.0,
            0.5,
            key="base_sleep"
        )

        base_social = st.slider(
            "👥 Social Hours",
            0.0,
            10.0,
            2.0,
            0.5,
            key="base_social"
        )


    with c2:

        base_physical = st.slider(
            "🏃 Physical Activity",
            0.0,
            8.0,
            1.0,
            0.5,
            key="base_physical"
        )

        base_extra = st.slider(
            "🎨 Extracurricular",
            0.0,
            8.0,
            2.0,
            0.5,
            key="base_extra"
        )

        base_stress = st.selectbox(
            "🧠 Stress Level",
            sorted(
                df["Stress_Level"].dropna().unique()
            ),
            key="base_stress"
        )


    st.divider()


    st.markdown(
        "### 🚀 Alternative Routine"
    )


    c3, c4 = st.columns(2)


    with c3:

        new_study = st.slider(
            "📚 New Study Hours",
            0.0,
            12.0,
            6.0,
            0.5,
            key="new_study"
        )

        new_sleep = st.slider(
            "😴 New Sleep Hours",
            0.0,
            12.0,
            7.0,
            0.5,
            key="new_sleep"
        )

        new_social = st.slider(
            "👥 New Social Hours",
            0.0,
            10.0,
            2.0,
            0.5,
            key="new_social"
        )


    with c4:

        new_physical = st.slider(
            "🏃 New Physical Activity",
            0.0,
            8.0,
            1.5,
            0.5,
            key="new_physical"
        )

        new_extra = st.slider(
            "🎨 New Extracurricular",
            0.0,
            8.0,
            2.0,
            0.5,
            key="new_extra"
        )

        new_stress = st.selectbox(
            "🧠 New Stress Level",
            sorted(
                df["Stress_Level"].dropna().unique()
            ),
            key="new_stress"
        )


    baseline_total = total_daily_hours(
        base_study,
        base_extra,
        base_sleep,
        base_social,
        base_physical
    )


    alternative_total = total_daily_hours(
        new_study,
        new_extra,
        new_sleep,
        new_social,
        new_physical
    )


    if (
        baseline_total > 24
        or alternative_total > 24
    ):

        st.error(
            "⚠️ One or both routines exceed 24 hours/day. "
            "Please adjust the values."
        )

    else:

        baseline_gpa = predict_gpa(
            base_study,
            base_extra,
            base_sleep,
            base_social,
            base_physical,
            base_stress
        )


        new_gpa = predict_gpa(
            new_study,
            new_extra,
            new_sleep,
            new_social,
            new_physical,
            new_stress
        )


        difference = (
            new_gpa - baseline_gpa
        )


        st.markdown(
            '<div class="section-title">📈 Simulation Result</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Current Estimate",
                f"{baseline_gpa:.2f}"
            )


        with col2:

            st.metric(
                "Alternative Estimate",
                f"{new_gpa:.2f}"
            )


        with col3:

            st.metric(
                "Model Difference",
                f"{difference:+.2f}"
            )


        comparison_df = pd.DataFrame({

            "Routine": [
                "Current",
                "Alternative"
            ],

            "Estimated GPA": [
                baseline_gpa,
                new_gpa
            ]
        })


        fig = go.Figure()


        fig.add_trace(

            go.Bar(

                x=comparison_df["Routine"],

                y=comparison_df["Estimated GPA"],

                text=[
                    f"{x:.2f}"
                    for x in comparison_df["Estimated GPA"]
                ],

                textposition="outside"
            )
        )


        fig.update_layout(

            template="plotly_dark",

            title="Current vs Alternative Routine",

            yaxis_title="Estimated GPA",

            yaxis_range=[
                0,
                10
            ],

            height=430
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.info(
            "⚠️ This is a what-if model simulation. "
            "The difference is not proof that changing a single habit "
            "will causally change GPA."
        )


# ============================================================
# STUDENT ANALYTICS
# ============================================================

elif page == "📊 Student Analytics":

    st.markdown(
        '<div class="section-title">📊 Student Population Analytics</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # GPA DISTRIBUTION
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        fig = go.Figure(

            go.Histogram(

                x=df["GPA"],

                nbinsx=20
            )
        )


        fig.update_layout(

            template="plotly_dark",

            title="GPA Distribution",

            xaxis_title="GPA",

            yaxis_title="Students"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # STUDY VS GPA
    # ========================================================

    with col2:

        fig = go.Figure(

            go.Scatter(

                x=df["Study_Hours_Per_Day"],

                y=df["GPA"],

                mode="markers",

                opacity=0.65
            )
        )


        fig.update_layout(

            template="plotly_dark",

            title="Study Hours vs GPA",

            xaxis_title="Study Hours / Day",

            yaxis_title="GPA"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # STRESS VS GPA
    # ========================================================

    stress_gpa = (

        df
        .groupby("Stress_Level")["GPA"]
        .mean()
        .reset_index()
    )


    fig = go.Figure(

        go.Bar(

            x=stress_gpa["Stress_Level"],

            y=stress_gpa["GPA"],

            text=[
                f"{x:.2f}"
                for x in stress_gpa["GPA"]
            ],

            textposition="outside"
        )
    )


    fig.update_layout(

        template="plotly_dark",

        title="Average GPA by Stress Level",

        xaxis_title="Stress Level",

        yaxis_title="Average GPA"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CORRELATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🔗 Feature Relationships</div>',
        unsafe_allow_html=True
    )


    numerical_data = df.select_dtypes(
        include=np.number
    )


    correlation = numerical_data.corr()


    fig = go.Figure(

        go.Heatmap(

            z=correlation.values,

            x=correlation.columns,

            y=correlation.columns
        )
    )


    fig.update_layout(

        template="plotly_dark",

        title="Correlation Matrix",

        height=650
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # DATASET
    # ========================================================

    st.markdown(
        '<div class="section-title">🔎 Dataset Preview</div>',
        unsafe_allow_html=True
    )


    st.dataframe(

        df.head(25),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# AI EXPLANATION
# ============================================================

elif page == "🔬 AI Explanation":

    st.markdown(
        '<div class="section-title">🔬 Why Does the Model Predict This?</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Feature importance shows which signals the trained model relied on most."
    )


    trained_model = model.named_steps[
        "model"
    ]


    preprocessor_used = model.named_steps[
        "preprocessor"
    ]


    feature_names = (
        preprocessor_used
        .get_feature_names_out()
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if hasattr(
        trained_model,
        "feature_importances_"
    ):

        importance = (
            trained_model
            .feature_importances_
        )


    elif hasattr(
        trained_model,
        "coef_"
    ):

        importance = np.abs(
            trained_model.coef_
        )


    else:

        importance = np.zeros(
            len(feature_names)
        )


    importance = np.ravel(
        importance
    )


    if len(importance) != len(feature_names):

        st.warning(
            "Feature importance is not available for this model configuration."
        )

    else:

        importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance
        })


        importance_df = (

            importance_df
            .sort_values(
                "Importance",
                ascending=True
            )
            .tail(10)
        )


        fig = go.Figure(

            go.Bar(

                x=importance_df["Importance"],

                y=importance_df["Feature"],

                orientation="h"
            )
        )


        fig.update_layout(

            template="plotly_dark",

            title="Top Predictive Signals",

            xaxis_title="Model Importance",

            yaxis_title="Feature",

            height=500
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        top_feature = (
            importance_df
            .iloc[-1]["Feature"]
        )


        st.success(
            f"💡 **Top model signal:** {top_feature}"
        )


    st.info(
        "Important: feature importance describes how much "
        "the trained model relied on a feature. It does not "
        "prove that changing that feature will causally "
        "change GPA."
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "🤖 Model Intelligence":

    st.markdown(
        '<div class="section-title">🤖 Model Intelligence</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Technical performance of the machine learning system."
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Model",
            model_info["model_name"]
        )


    with col2:

        st.metric(
            "R² Score",
            f'{model_info["r2"]:.4f}'
        )


    with col3:

        st.metric(
            "RMSE",
            f'{model_info["rmse"]:.4f}'
        )


    with col4:

        st.metric(
            "MAE",
            f'{model_info["mae"]:.4f}'
        )


    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📚 Training Information</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    Training Samples
                </div>

                <div class="kpi-value">
                    {model_info["training_samples"]:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    Testing Samples
                </div>

                <div class="kpi-value">
                    {model_info["testing_samples"]:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="glass-card">

                <div class="kpi-label">
                    Prediction Target
                </div>

                <div class="kpi-value">
                    {model_info["target"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # METRIC EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📖 How to Read These Metrics</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="glass-card">

            <h4>R² Score</h4>

            <p style="color:#94a3b8">
                Indicates how much variation in the target variable
                is explained by the model. Higher is generally better.
            </p>

            <h4>RMSE</h4>

            <p style="color:#94a3b8">
                Measures the typical size of prediction errors while
                giving larger errors more weight. Lower is better.
            </p>

            <h4>MAE</h4>

            <p style="color:#94a3b8">
                Measures the average absolute prediction error.
                Lower is better.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # RESPONSIBLE INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">⚠️ Responsible Interpretation</div>',
        unsafe_allow_html=True
    )


    st.warning(
        "FocusGuard is a predictive analytics project. "
        "Its predictions should not be treated as guaranteed "
        "future GPA values, medical advice, or causal conclusions."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>FocusGuard AI</b>

        <br>

        Student Performance Intelligence

        <br><br>

        Python • Pandas • Scikit-learn • Plotly • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)