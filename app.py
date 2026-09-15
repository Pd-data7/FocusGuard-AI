# ============================================================
# FOCUSGUARD AI
# Student Performance Intelligence System
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FocusGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "focusguard_gpa_model.pkl"
DATA_PATH = BASE_DIR / "student_productivity.csv"
INFO_PATH = BASE_DIR / "model_info.json"


# ============================================================
# CUSTOM CSS
# IMPORTANT:
# The card HTML below intentionally has NO blank lines between
# nested HTML elements. This prevents Streamlit Markdown from
# displaying inner HTML tags as literal text.
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.16), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(168,85,247,0.14), transparent 25%),
        #070914;
    color: #f8fafc;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 4rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: rgba(8, 10, 24, 0.98);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -1.3px;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.gradient-text {
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
    margin-bottom: 1.2rem;
}

.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1.5rem;
    backdrop-filter: blur(15px);
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
    height: 100%;
    box-sizing: border-box;
}

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

.section-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 1.8rem;
    margin-bottom: 1rem;
}

.prediction-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.17), rgba(168,85,247,0.10));
    border: 1px solid rgba(129,140,248,0.35);
    border-radius: 26px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 20px 70px rgba(79,70,229,0.14);
    min-height: 290px;
    box-sizing: border-box;
}

.prediction-value {
    font-size: 4.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.insight-card {
    background: rgba(129,140,248,0.07);
    border: 1px solid rgba(129,140,248,0.20);
    border-radius: 18px;
    padding: 1.25rem;
    margin-bottom: 0.7rem;
}

.improvement-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.05));
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

.stButton > button {
    width: 100%;
    border-radius: 13px;
    border: 1px solid rgba(129,140,248,0.3);
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 700;
    padding: 0.75rem;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

.footer {
    text-align: center;
    color: #64748b;
    padding-top: 3rem;
    font-size: 0.8rem;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.8rem;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 1.8rem;
    }

    .prediction-value {
        font-size: 3.3rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SMALL HTML HELPERS
# ============================================================

def render_card(title, value, value_style=""):
    st.markdown(
        f'<div class="glass-card"><div class="kpi-label">{title}</div><div class="kpi-value" style="{value_style}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_flow_card(icon, title, description):
    st.markdown(
        f'<div class="glass-card"><div style="font-size:1.8rem;margin-bottom:.55rem">{icon}</div><div style="font-weight:750;font-size:1rem">{title}</div><p style="color:#94a3b8;line-height:1.55">{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_insight(text):
    st.markdown(
        f'<div class="insight-card">{text}</div>',
        unsafe_allow_html=True,
    )


def render_improvement(icon, title, text):
    st.markdown(
        f'<div class="improvement-card"><div class="improvement-title">{icon} {title}</div><div class="improvement-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "❌ focusguard_gpa_model.pkl was not found.\n\n"
            f"Expected location:\n{MODEL_PATH}"
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"❌ Unable to load the ML model.\n\n{exc}")
        st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "❌ student_productivity.csv was not found.\n\n"
            f"Expected location:\n{DATA_PATH}"
        )
        st.stop()

    try:
        return pd.read_csv(DATA_PATH)
    except Exception as exc:
        st.error(f"❌ Unable to load the dataset.\n\n{exc}")
        st.stop()


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

@st.cache_data
def load_model_info():
    default_info = {
        "model_name": "Unknown",
        "mae": 0,
        "rmse": 0,
        "r2": 0,
        "training_samples": 0,
        "testing_samples": 0,
        "target": "GPA",
    }

    if not INFO_PATH.exists():
        return default_info

    try:
        with open(INFO_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        return {**default_info, **data}
    except Exception:
        return default_info


model = load_model()
df = load_data()
model_info = load_model_info()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Stress_Level",
    "GPA",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(
        "❌ Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def total_daily_hours(study, extra, sleep, social, physical):
    return float(study + extra + sleep + social + physical)


def build_input_dataframe(study, extra, sleep, social, physical, stress):
    total_hours = total_daily_hours(
        study, extra, sleep, social, physical
    )

    productive_hours = study + extra
    free_time = 24 - total_hours

    return pd.DataFrame(
        {
            "Study_Hours_Per_Day": [study],
            "Extracurricular_Hours_Per_Day": [extra],
            "Sleep_Hours_Per_Day": [sleep],
            "Social_Hours_Per_Day": [social],
            "Physical_Activity_Hours_Per_Day": [physical],
            "Stress_Level": [stress],
            "Total_Tracked_Hours": [total_hours],
            "Productive_Hours_Per_Day": [productive_hours],
            "Free_Time_Per_Day": [free_time],
        }
    )


def predict_gpa(study, extra, sleep, social, physical, stress):
    total_hours = total_daily_hours(
        study, extra, sleep, social, physical
    )

    if total_hours > 24:
        raise ValueError(
            f"Daily routine exceeds 24 hours: {total_hours:.1f} hours."
        )

    if min(study, extra, sleep, social, physical) < 0:
        raise ValueError("Daily hours cannot be negative.")

    input_data = build_input_dataframe(
        study, extra, sleep, social, physical, stress
    )

    try:
        prediction = model.predict(input_data)
    except Exception as exc:
        raise RuntimeError(
            "The input features do not match the trained model.\n\n"
            "This app supplies these 9 features:\n"
            "Study_Hours_Per_Day, Extracurricular_Hours_Per_Day, "
            "Sleep_Hours_Per_Day, Social_Hours_Per_Day, "
            "Physical_Activity_Hours_Per_Day, Stress_Level, "
            "Total_Tracked_Hours, Productive_Hours_Per_Day, "
            "Free_Time_Per_Day.\n\n"
            f"Technical error: {exc}"
        ) from exc

    prediction = float(np.asarray(prediction).ravel()[0])
    return float(np.clip(prediction, 0, 10))


def get_status(gpa):
    if gpa >= 8.5:
        return "Excellent Academic Profile", "🟢"
    if gpa >= 7.0:
        return "Strong Academic Profile", "🔵"
    if gpa >= 5.5:
        return "Improvement Opportunity", "🟡"
    return "Needs Attention", "🔴"


def create_gauge(gpa):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gpa,
            number={"font": {"size": 48}, "suffix": " / 10"},
            title={"text": "Predicted GPA"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"thickness": 0.75},
                "steps": [
                    {"range": [0, 5.5]},
                    {"range": [5.5, 7]},
                    {"range": [7, 8.5]},
                    {"range": [8.5, 10]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": gpa,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=330,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def create_radar(study, sleep, social, physical, extra):
    values = [
        min(study / 12 * 100, 100),
        min(sleep / 12 * 100, 100),
        min(social / 10 * 100, 100),
        min(physical / 8 * 100, 100),
        min(extra / 8 * 100, 100),
    ]

    categories = [
        "Study",
        "Sleep",
        "Social",
        "Physical Activity",
        "Extracurricular",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Daily Routine",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
            }
        },
        showlegend=False,
        height=430,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def normalize_stress(stress):
    value = str(stress).strip().lower()

    if value in {"high", "3", "3.0"}:
        return "High"

    if value in {"medium", "moderate", "mid", "2", "2.0"}:
        return "Medium"

    if value in {"low", "1", "1.0"}:
        return "Low"

    return str(stress).title()


def generate_insights(
    study, sleep, social, physical, extra, stress, gpa
):
    insights = []
    display_stress = normalize_stress(stress)

    if study < 3:
        insights.append(
            "📚 Your reported study time is relatively low. "
            "Consider protecting at least one focused study block each day."
        )
    elif study >= 6:
        insights.append(
            "📚 Your reported study commitment is strong. "
            "Focus on consistency, revision and active practice."
        )

    if sleep < 6:
        insights.append(
            "😴 Your reported sleep duration is low. "
            "A consistent sleep routine may help you manage your academic workload."
        )
    elif sleep >= 7:
        insights.append(
            "😴 Your reported sleep duration is in a reasonable range. "
            "Try to maintain consistency."
        )

    if social > 5:
        insights.append(
            "👥 A large portion of your reported day is allocated to social activity. "
            "Protecting focused academic blocks may improve time management."
        )

    if physical < 1:
        insights.append(
            "🏃 Your reported physical activity is relatively low. "
            "Consider including regular movement in your routine."
        )

    if display_stress == "High":
        insights.append(
            "🧠 You reported high stress. "
            "Breaking large academic tasks into smaller steps may make your workload easier to manage."
        )
    elif display_stress == "Medium":
        insights.append(
            "🧠 Your reported stress is moderate. "
            "Planning tasks in smaller blocks may help you stay organized."
        )
    elif display_stress == "Low":
        insights.append("🧠 Your reported stress level is low.")

    if not insights:
        insights.append(
            "✨ Your routine does not trigger a major warning from "
            "the simple rule-based checks."
        )

    return insights


def generate_improvement_plan(
    study, sleep, social, physical, extra, stress, gpa
):
    tips = []
    display_stress = normalize_stress(stress)

    if study < 3:
        tips.append((
            "📚",
            "Study Focus",
            "Your current reported study time is relatively low. "
            "Try adding one focused study session to your daily routine.",
        ))
    elif study < 5:
        tips.append((
            "📚",
            "Study Consistency",
            "Build consistency with a fixed daily study schedule "
            "and gradually improve the quality of your focused sessions.",
        ))
    else:
        tips.append((
            "📚",
            "Study Strength",
            "Your reported study commitment is strong. "
            "Focus on consistency, revision and active practice.",
        ))

    if sleep < 6:
        tips.append((
            "😴",
            "Sleep Routine",
            "Your reported sleep duration is low. "
            "Try to maintain a consistent sleep schedule and allow enough time for rest.",
        ))
    elif sleep >= 7:
        tips.append((
            "😴",
            "Sleep Routine",
            "Your reported sleep duration is in a reasonable range. "
            "Try to keep your sleep schedule consistent.",
        ))

    if display_stress == "High":
        tips.append((
            "🧠",
            "Stress Management",
            "You reported high stress. Break difficult academic tasks "
            "into smaller goals and use short recovery breaks.",
        ))
    elif display_stress == "Medium":
        tips.append((
            "🧠",
            "Stress Management",
            "Your reported stress level is moderate. "
            "Planning your workload in smaller blocks may make studying easier to manage.",
        ))

    if physical < 1:
        tips.append((
            "🏃",
            "Physical Activity",
            "Your reported physical activity is relatively low. "
            "Consider adding some regular movement to your daily routine.",
        ))

    if social > 5:
        tips.append((
            "📱",
            "Time Management",
            "A large part of your reported day is allocated to social activity. "
            "Consider protecting specific hours for focused academic work.",
        ))

    if gpa < 5.5:
        tips.append((
            "🚀",
            "Recovery Strategy",
            "Do not try to change everything at once. Start with one or two "
            "manageable habits, stay consistent, and review your progress each week.",
        ))
    elif gpa < 7:
        tips.append((
            "🎯",
            "Improvement Strategy",
            "Your model-estimated profile shows room for improvement. "
            "Focus on consistency, planned study sessions and reducing avoidable distractions.",
        ))
    else:
        tips.append((
            "🏆",
            "Maintain Progress",
            "Your model-estimated performance is relatively strong. "
            "Focus on maintaining effective habits instead of overloading your routine.",
        ))

    return tips


def get_feature_importance():
    if not hasattr(model, "named_steps"):
        return None

    steps = model.named_steps

    # Prefer the step named "model", matching the original app.
    if "model" in steps:
        trained_model = steps["model"]
    else:
        # Fallback: use the last pipeline step.
        if not steps:
            return None
        trained_model = list(steps.values())[-1]

    feature_names = None

    if "preprocessor" in steps:
        try:
            feature_names = steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = None

    if hasattr(trained_model, "feature_importances_"):
        importance = np.asarray(trained_model.feature_importances_)
    elif hasattr(trained_model, "coef_"):
        importance = np.abs(np.asarray(trained_model.coef_))
    else:
        return None

    importance = np.ravel(importance)

    if feature_names is None:
        feature_names = [
            f"Feature {i + 1}" for i in range(len(importance))
        ]

    feature_names = list(feature_names)

    if len(importance) != len(feature_names):
        return None

    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div style="text-align:center"><div style="font-size:2.7rem;margin-bottom:.3rem">🧠</div><div style="font-size:1.35rem;font-weight:800">FocusGuard</div><div style="color:#94a3b8;font-size:.75rem">AI Student Intelligence</div></div>',
        unsafe_allow_html=True,
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
            "🤖 Model Intelligence",
        ],
    )

    st.divider()

    st.markdown("### 🤖 Current Model")
    st.caption(model_info.get("model_name", "Unknown"))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R²",
            f'{float(model_info.get("r2", 0)):.3f}',
        )

    with col2:
        st.metric(
            "RMSE",
            f'{float(model_info.get("rmse", 0)):.3f}',
        )

    st.divider()
    st.caption("FocusGuard AI")
    st.caption("Student Performance Intelligence")


# ============================================================
# GLOBAL HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">Focus<span class="gradient-text">Guard</span> AI</div><div class="hero-subtitle">Student Performance Intelligence &nbsp;•&nbsp; Predictive Analytics &nbsp;•&nbsp; Personalized Insights</div>',
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":
    st.markdown(
        '<div class="section-title">📡 System Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_card("Students", f"{len(df):,}")

    with col2:
        render_card("Average GPA", f'{df["GPA"].mean():.2f}')

    with col3:
        render_card(
            "Model R²",
            f'{float(model_info.get("r2", 0)):.3f}',
        )

    with col4:
        render_card(
            "ML Model",
            str(model_info.get("model_name", "Unknown")),
            "font-size:1.05rem;",
        )

    st.markdown(
        '<div class="section-title">🧠 How FocusGuard Works</div>',
        unsafe_allow_html=True,
    )

    flow_columns = st.columns(5)

    flow_items = [
        ("👤", "Student Habits", "Daily routine and lifestyle data"),
        ("🧹", "Data Processing", "Cleaning and feature engineering"),
        ("🤖", "ML Model", "Machine learning prediction"),
        ("🎯", "Prediction", "Estimated academic performance"),
        ("💡", "Insights", "Explore routine patterns"),
    ]

    for column, item in zip(flow_columns, flow_items):
        with column:
            render_flow_card(*item)

    st.markdown(
        '<div class="section-title">✨ Explore FocusGuard</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_flow_card(
            "🎯",
            "GPA Predictor",
            "Enter a student's routine and generate an ML-based GPA estimate.",
        )

    with col2:
        render_flow_card(
            "🧪",
            "What-If Lab",
            "Compare a current routine with an alternative simulated routine.",
        )

    with col3:
        render_flow_card(
            "🔬",
            "AI Explanation",
            "Explore which features the trained model relies on most.",
        )


# ============================================================
# GPA PREDICTOR
# ============================================================

elif page == "🎯 GPA Predictor":
    st.markdown(
        '<div class="section-title">🎯 GPA Predictor</div>',
        unsafe_allow_html=True,
    )
    st.caption("Enter a student's typical daily routine.")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "📚 Study Hours / Day", 0.0, 12.0, 5.0, 0.5
        )

        sleep_hours = st.slider(
            "😴 Sleep Hours / Day", 0.0, 12.0, 7.0, 0.5
        )

        extracurricular_hours = st.slider(
            "🎨 Extracurricular Hours / Day", 0.0, 8.0, 2.0, 0.5
        )

    with col2:
        social_hours = st.slider(
            "👥 Social Hours / Day", 0.0, 10.0, 2.0, 0.5
        )

        physical_hours = st.slider(
            "🏃 Physical Activity / Day", 0.0, 8.0, 1.0, 0.5
        )

        stress_options = (
            df["Stress_Level"]
            .dropna()
            .unique()
            .tolist()
        )

        if not stress_options:
            stress_options = ["Low", "Medium", "High"]

        stress_level = st.selectbox(
            "🧠 Stress Level",
            stress_options,
        )

    current_total = total_daily_hours(
        study_hours,
        extracurricular_hours,
        sleep_hours,
        social_hours,
        physical_hours,
    )

    if current_total > 24:
        st.error(
            f"⚠️ Your routine adds up to {current_total:.1f} hours/day. "
            "Please reduce the selected hours."
        )
    else:
        remaining = 24 - current_total
        st.caption(
            f"Daily tracked time: {current_total:.1f} / 24 hours "
            f"• Untracked time: {remaining:.1f} hours"
        )

    if st.button("🚀 ANALYZE MY ROUTINE"):
        if current_total > 24:
            st.error("Please create a valid 24-hour daily routine first.")
        else:
            try:
                prediction = predict_gpa(
                    study_hours,
                    extracurricular_hours,
                    sleep_hours,
                    social_hours,
                    physical_hours,
                    stress_level,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                status, icon = get_status(prediction)

                left, right = st.columns([1, 1])

                with left:
                    st.markdown(
                        f'<div class="prediction-box"><div class="kpi-label">ML-BASED GPA ESTIMATE</div><div class="prediction-value">{prediction:.2f}</div><h3>{icon} {status}</h3><p style="color:#94a3b8">This is a model estimate based on the provided routine and learned patterns in the training data.</p></div>',
                        unsafe_allow_html=True,
                    )

                with right:
                    st.plotly_chart(
                        create_gauge(prediction),
                        use_container_width=True,
                    )

                st.markdown(
                    '<div class="section-title">🧠 Smart Routine Insights</div>',
                    unsafe_allow_html=True,
                )

                for insight in generate_insights(
                    study_hours,
                    sleep_hours,
                    social_hours,
                    physical_hours,
                    extracurricular_hours,
                    stress_level,
                    prediction,
                ):
                    render_insight(insight)

                st.markdown(
                    '<div class="section-title">🎯 Personalized Improvement Plan</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Suggestions are based on the routine entered above. "
                    "They are general guidance, not guaranteed ways to increase GPA."
                )

                for icon_text, title, text in generate_improvement_plan(
                    study_hours,
                    sleep_hours,
                    social_hours,
                    physical_hours,
                    extracurricular_hours,
                    stress_level,
                    prediction,
                ):
                    render_improvement(icon_text, title, text)

                st.markdown(
                    '<div class="section-title">📅 7-Day Action Plan</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "A simple plan for turning the insights into consistent habits."
                )

                action_plan = [
                    ("Day 1", "📚", "Focused Study", "Complete one distraction-free study session."),
                    ("Day 2", "😴", "Sleep Routine", "Maintain a consistent sleep and wake schedule."),
                    ("Day 3", "🧠", "Stress Management", "Break one difficult academic task into smaller steps."),
                    ("Day 4", "🏃", "Physical Activity", "Include some regular movement in your day."),
                    ("Day 5", "📱", "Distraction Control", "Protect one study block from unnecessary distractions."),
                    ("Day 6", "🔄", "Weekly Review", "Review what worked and what made studying difficult."),
                    ("Day 7", "🎯", "Next Week Planning", "Set realistic study targets for the coming week."),
                ]

                for day, action_icon, title, description in action_plan:
                    st.markdown(
                        f'<div class="improvement-card"><div style="display:flex;align-items:center;gap:15px"><div class="day-number">{day}</div><div style="font-size:1.35rem">{action_icon}</div><div><div class="day-title">{title}</div><div class="day-description">{description}</div></div></div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="section-title">📊 Your Routine Distribution</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "This chart shows how your reported daily routine is distributed. "
                    "Higher values do not necessarily mean better performance."
                )

                st.plotly_chart(
                    create_radar(
                        study_hours,
                        sleep_hours,
                        social_hours,
                        physical_hours,
                        extracurricular_hours,
                    ),
                    use_container_width=True,
                )

                report = f"""FOCUSGUARD AI
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

Total Tracked Hours:
{current_total:.1f}

MODEL
-----

Model:
{model_info.get("model_name", "Unknown")}

R²:
{float(model_info.get("r2", 0)):.4f}

RMSE:
{float(model_info.get("rmse", 0)):.4f}

MAE:
{float(model_info.get("mae", 0)):.4f}

IMPORTANT
---------

This report contains an ML-based estimate.
It does not guarantee a student's future GPA.
Model associations should not be interpreted as causal effects.
"""

                st.download_button(
                    "📄 Download Prediction Report",
                    report,
                    file_name="focusguard_prediction_report.txt",
                    mime="text/plain",
                )


# ============================================================
# WHAT-IF LAB
# ============================================================

elif page == "🧪 What-If Lab":
    st.markdown(
        '<div class="section-title">🧪 Before → After What-If Lab</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Compare two hypothetical routines and see how the model's estimate changes."
    )

    st.markdown("### 👤 Current Routine")

    c1, c2 = st.columns(2)

    with c1:
        base_study = st.slider(
            "📚 Study Hours", 0.0, 12.0, 5.0, 0.5, key="base_study"
        )
        base_sleep = st.slider(
            "😴 Sleep Hours", 0.0, 12.0, 7.0, 0.5, key="base_sleep"
        )
        base_social = st.slider(
            "👥 Social Hours", 0.0, 10.0, 2.0, 0.5, key="base_social"
        )

    with c2:
        base_physical = st.slider(
            "🏃 Physical Activity", 0.0, 8.0, 1.0, 0.5, key="base_physical"
        )
        base_extra = st.slider(
            "🎨 Extracurricular", 0.0, 8.0, 2.0, 0.5, key="base_extra"
        )

        stress_options = (
            df["Stress_Level"]
            .dropna()
            .unique()
            .tolist()
        )
        if not stress_options:
            stress_options = ["Low", "Medium", "High"]

        base_stress = st.selectbox(
            "🧠 Stress Level",
            stress_options,
            key="base_stress",
        )

    baseline_total = total_daily_hours(
        base_study, base_extra, base_sleep, base_social, base_physical
    )

    st.divider()
    st.markdown("### 🚀 Alternative Routine")

    c3, c4 = st.columns(2)

    with c3:
        new_study = st.slider(
            "📚 New Study Hours", 0.0, 12.0, 6.0, 0.5, key="new_study"
        )
        new_sleep = st.slider(
            "😴 New Sleep Hours", 0.0, 12.0, 7.0, 0.5, key="new_sleep"
        )
        new_social = st.slider(
            "👥 New Social Hours", 0.0, 10.0, 2.0, 0.5, key="new_social"
        )

    with c4:
        new_physical = st.slider(
            "🏃 New Physical Activity", 0.0, 8.0, 1.5, 0.5, key="new_physical"
        )
        new_extra = st.slider(
            "🎨 New Extracurricular", 0.0, 8.0, 2.0, 0.5, key="new_extra"
        )
        new_stress = st.selectbox(
            "🧠 New Stress Level",
            stress_options,
            key="new_stress",
        )

    alternative_total = total_daily_hours(
        new_study, new_extra, new_sleep, new_social, new_physical
    )

    if baseline_total > 24:
        st.error(f"⚠️ Current routine totals {baseline_total:.1f} hours/day.")
    elif alternative_total > 24:
        st.error(
            f"⚠️ Alternative routine totals {alternative_total:.1f} hours/day."
        )
    else:
        try:
            baseline_gpa = predict_gpa(
                base_study,
                base_extra,
                base_sleep,
                base_social,
                base_physical,
                base_stress,
            )
            new_gpa = predict_gpa(
                new_study,
                new_extra,
                new_sleep,
                new_social,
                new_physical,
                new_stress,
            )

            difference = new_gpa - baseline_gpa

            st.markdown(
                '<div class="section-title">📈 Simulation Result</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Estimate", f"{baseline_gpa:.2f}")

            with col2:
                st.metric("Alternative Estimate", f"{new_gpa:.2f}")

            with col3:
                st.metric("Model Difference", f"{difference:+.2f}")

            comparison_df = pd.DataFrame(
                {
                    "Routine": ["Current", "Alternative"],
                    "Estimated GPA": [baseline_gpa, new_gpa],
                }
            )

            fig = go.Figure(
                go.Bar(
                    x=comparison_df["Routine"],
                    y=comparison_df["Estimated GPA"],
                    text=[f"{x:.2f}" for x in comparison_df["Estimated GPA"]],
                    textposition="outside",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                title="Current vs Alternative Routine",
                yaxis_title="Estimated GPA",
                yaxis_range=[0, 10],
                height=430,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "⚠️ This is a what-if model simulation. The difference is not "
                "proof that changing a single habit will causally change GPA."
            )

        except Exception as exc:
            st.error(f"❌ Prediction error: {exc}")


# ============================================================
# STUDENT ANALYTICS
# ============================================================

elif page == "📊 Student Analytics":
    st.markdown(
        '<div class="section-title">📊 Student Population Analytics</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(
            go.Histogram(
                x=df["GPA"],
                nbinsx=20,
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title="GPA Distribution",
            xaxis_title="GPA",
            yaxis_title="Students",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(
            go.Scatter(
                x=df["Study_Hours_Per_Day"],
                y=df["GPA"],
                mode="markers",
                opacity=0.65,
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title="Study Hours vs GPA",
            xaxis_title="Study Hours / Day",
            yaxis_title="GPA",
        )

        st.plotly_chart(fig, use_container_width=True)

    stress_gpa = (
        df.groupby("Stress_Level")["GPA"]
        .mean()
        .reset_index()
    )

    fig = go.Figure(
        go.Bar(
            x=stress_gpa["Stress_Level"],
            y=stress_gpa["GPA"],
            text=[f"{x:.2f}" for x in stress_gpa["GPA"]],
            textposition="outside",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Average GPA by Stress Level",
        xaxis_title="Stress Level",
        yaxis_title="Average GPA",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These charts show patterns in the dataset. "
        "They do not establish causal relationships."
    )

    st.markdown(
        '<div class="section-title">🔗 Feature Relationships</div>',
        unsafe_allow_html=True,
    )

    numerical_data = df.select_dtypes(include=np.number)
    correlation = numerical_data.corr()

    fig = go.Figure(
        go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Correlation Matrix",
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">🔎 Dataset Preview</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(25),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# AI EXPLANATION
# ============================================================

elif page == "🔬 AI Explanation":
    st.markdown(
        '<div class="section-title">🔬 Why Does the Model Predict This?</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Feature importance shows which signals the trained model relied on most."
    )

    importance_df = get_feature_importance()

    if importance_df is None:
        st.warning(
            "⚠️ Feature importance is not available for the current model configuration."
        )
        st.info(
            "This can happen when the saved model is not a Pipeline or "
            "the underlying estimator does not expose feature importance or coefficients."
        )
    else:
        importance_df = (
            importance_df
            .sort_values("Importance", ascending=True)
            .tail(10)
        )

        fig = go.Figure(
            go.Bar(
                x=importance_df["Importance"],
                y=importance_df["Feature"],
                orientation="h",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title="Top Predictive Signals",
            xaxis_title="Model Importance",
            yaxis_title="Feature",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        top_feature = importance_df.iloc[-1]["Feature"]

        st.success(f"💡 **Top model signal:** {top_feature}")

    st.info(
        "Important: feature importance describes how much the trained model "
        "relied on a feature. It does not prove that changing that feature "
        "will causally change GPA."
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "🤖 Model Intelligence":
    st.markdown(
        '<div class="section-title">🤖 Model Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Technical performance of the machine learning system."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            str(model_info.get("model_name", "Unknown")),
        )

    with col2:
        st.metric(
            "R² Score",
            f'{float(model_info.get("r2", 0)):.4f}',
        )

    with col3:
        st.metric(
            "RMSE",
            f'{float(model_info.get("rmse", 0)):.4f}',
        )

    with col4:
        st.metric(
            "MAE",
            f'{float(model_info.get("mae", 0)):.4f}',
        )

    st.markdown(
        '<div class="section-title">📚 Training Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_card(
            "Training Samples",
            f'{int(model_info.get("training_samples", 0)):,}',
        )

    with col2:
        render_card(
            "Testing Samples",
            f'{int(model_info.get("testing_samples", 0)):,}',
        )

    with col3:
        render_card(
            "Prediction Target",
            str(model_info.get("target", "GPA")),
        )

    st.markdown(
        '<div class="section-title">📖 How to Read These Metrics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="glass-card"><h4>R² Score</h4><p style="color:#94a3b8">Indicates how much variation in GPA is explained by the model. Higher is generally better.</p><h4>RMSE</h4><p style="color:#94a3b8">Measures prediction error while giving larger errors more weight. Lower is better.</p><h4>MAE</h4><p style="color:#94a3b8">Measures the average absolute prediction error. Lower is better.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">🔧 Model Structure</div>',
        unsafe_allow_html=True,
    )

    st.code(str(type(model)))

    if hasattr(model, "named_steps"):
        st.success("✅ The saved model is a Scikit-learn Pipeline.")
        st.write("Pipeline steps:", list(model.named_steps.keys()))
    else:
        st.warning("⚠️ The saved model is not a Pipeline.")

    st.markdown(
        '<div class="section-title">⚠️ Responsible Interpretation</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "FocusGuard is a predictive analytics project. Its predictions "
        "should not be treated as guaranteed future GPA values or causal conclusions."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer"><b>FocusGuard AI</b><br>Student Performance Intelligence<br><br>Python • Pandas • Scikit-learn • Plotly • Streamlit</div>',
    unsafe_allow_html=True,
)
