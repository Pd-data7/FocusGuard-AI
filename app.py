# ============================================================
# FOCUSGUARD AI
# Student Performance Intelligence System
# Stable Streamlit Version
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FocusGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "focusguard_gpa_model.pkl"
DATA_PATH = BASE_DIR / "student_productivity.csv"
INFO_PATH = BASE_DIR / "model_info.json"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(168,85,247,0.12),
                transparent 28%
            ),
            #070914;
    }

    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1450px;
    }

    section[data-testid="stSidebar"] {
        background: #090b18;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found:\n\n{MODEL_PATH}"
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        st.error(
            "Unable to load the ML model.\n\n"
            f"Error: {e}"
        )
        st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            f"Dataset file not found:\n\n{DATA_PATH}"
        )
        st.stop()

    try:
        return pd.read_csv(DATA_PATH)

    except Exception as e:
        st.error(
            "Unable to read dataset.\n\n"
            f"Error: {e}"
        )
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

        with open(
            INFO_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            info = json.load(file)

        return {
            **default_info,
            **info
        }

    except Exception:
        return default_info


# ============================================================
# INITIALIZE
# ============================================================

model = load_model()
df = load_data()
model_info = load_model_info()


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Stress_Level",
    "GPA",
]


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]


if missing_columns:

    st.error(
        "Dataset is missing these columns:\n\n"
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Stress_Level",
    "Total_Tracked_Hours",
    "Productive_Hours_Per_Day",
    "Free_Time_Per_Day",
]


# ============================================================
# GPA SCALE
# ============================================================

dataset_gpa_max = float(df["GPA"].max())

if dataset_gpa_max <= 4.5:
    GPA_SCALE_MAX = 4.0
else:
    GPA_SCALE_MAX = 10.0


# ============================================================
# STRESS OPTIONS
# IMPORTANT:
# The trained model uses Stress_Level as a categorical feature.
# Therefore we MUST pass text such as Low / Medium / High.
# ============================================================

def get_stress_options():

    try:

        if hasattr(model, "named_steps"):

            if "preprocessor" in model.named_steps:

                preprocessor = model.named_steps["preprocessor"]

                if hasattr(
                    preprocessor,
                    "transformers_"
                ):

                    for (
                        name,
                        transformer,
                        columns
                    ) in preprocessor.transformers_:

                        if name == "categorical":

                            if hasattr(
                                transformer,
                                "categories_"
                            ):

                                categories = (
                                    transformer.categories_[0]
                                )

                                options = [
                                    str(x)
                                    for x in categories
                                ]

                                if options:
                                    return options

    except Exception:
        pass

    return [
        "Low",
        "Medium",
        "High"
    ]


STRESS_OPTIONS = get_stress_options()


# ============================================================
# TOTAL DAILY HOURS
# ============================================================

def total_daily_hours(
    study,
    extra,
    sleep,
    social,
    physical
):

    return (
        float(study)
        + float(extra)
        + float(sleep)
        + float(social)
        + float(physical)
    )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_input_dataframe(
    study,
    extra,
    sleep,
    social,
    physical,
    stress
):

    study = float(study)
    extra = float(extra)
    sleep = float(sleep)
    social = float(social)
    physical = float(physical)

    total = total_daily_hours(
        study,
        extra,
        sleep,
        social,
        physical
    )

    productive = study + extra

    free_time = 24.0 - total

    data = {
        "Study_Hours_Per_Day": study,
        "Extracurricular_Hours_Per_Day": extra,
        "Sleep_Hours_Per_Day": sleep,
        "Social_Hours_Per_Day": social,
        "Physical_Activity_Hours_Per_Day": physical,

        # IMPORTANT:
        # Keep Stress_Level as STRING.
        "Stress_Level": str(stress).strip(),

        "Total_Tracked_Hours": total,
        "Productive_Hours_Per_Day": productive,
        "Free_Time_Per_Day": free_time,
    }

    return pd.DataFrame(
        [data],
        columns=FEATURE_COLUMNS
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_gpa(
    study,
    extra,
    sleep,
    social,
    physical,
    stress
):

    values = [
        study,
        extra,
        sleep,
        social,
        physical
    ]

    # --------------------------------------------------------
    # Validate negative values
    # --------------------------------------------------------

    if any(float(x) < 0 for x in values):

        raise ValueError(
            "Hours cannot be negative."
        )


    # --------------------------------------------------------
    # Validate 24-hour day
    # --------------------------------------------------------

    total_hours = total_daily_hours(
        study,
        extra,
        sleep,
        social,
        physical
    )

    if total_hours > 24:

        raise ValueError(
            f"Your daily hours are "
            f"{total_hours:.1f}. "
            "They cannot exceed 24 hours."
        )


    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT convert Low/Medium/High to 1/2/3.
    # The trained model uses OneHotEncoder.
    # --------------------------------------------------------

    input_data = build_input_dataframe(
        study,
        extra,
        sleep,
        social,
        physical,
        stress
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)

    except Exception as e:

        raise RuntimeError(
            "Model prediction failed.\n\n"
            f"Technical error: {e}\n\n"
            "Make sure the .pkl model and app.py "
            "belong to the same trained version."
        )


    prediction = float(
        np.asarray(prediction)
        .flatten()[0]
    )


    # --------------------------------------------------------
    # Keep prediction inside the valid GPA scale.
    # --------------------------------------------------------

    prediction = max(
        0.0,
        min(
            prediction,
            GPA_SCALE_MAX
        )
    )

    return prediction


# ============================================================
# GPA STATUS
# ============================================================

def get_status(gpa):

    if GPA_SCALE_MAX == 4.0:

        if gpa >= 3.5:
            return "Excellent Academic Profile", "🟢"

        elif gpa >= 3.0:
            return "Strong Academic Profile", "🔵"

        elif gpa >= 2.0:
            return "Improvement Opportunity", "🟡"

        else:
            return "Needs Attention", "🔴"

    else:

        if gpa >= 8.5:
            return "Excellent Academic Profile", "🟢"

        elif gpa >= 7.0:
            return "Strong Academic Profile", "🔵"

        elif gpa >= 5.5:
            return "Improvement Opportunity", "🟡"

        else:
            return "Needs Attention", "🔴"


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
                    "size": 42
                },
                "suffix": f" / {GPA_SCALE_MAX:g}"
            },

            title={
                "text": "Predicted GPA"
            },

            gauge={

                "axis": {
                    "range": [
                        0,
                        GPA_SCALE_MAX
                    ]
                },

                "bar": {
                    "thickness": 0.7
                },

                "steps": [

                    {
                        "range": [
                            0,
                            GPA_SCALE_MAX * 0.55
                        ]
                    },

                    {
                        "range": [
                            GPA_SCALE_MAX * 0.55,
                            GPA_SCALE_MAX * 0.70
                        ]
                    },

                    {
                        "range": [
                            GPA_SCALE_MAX * 0.70,
                            GPA_SCALE_MAX * 0.85
                        ]
                    },

                    {
                        "range": [
                            GPA_SCALE_MAX * 0.85,
                            GPA_SCALE_MAX
                        ]
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

        height=350,

        margin={
            "l": 20,
            "r": 20,
            "t": 70,
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
        "Physical Activity",
        "Extracurricular"
    ]

    fig = go.Figure()

    fig.add_trace(

        go.Scatterpolar(

            r=values + [values[0]],

            theta=categories + [
                categories[0]
            ],

            fill="toself",

            name="Daily Routine"
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

        height=430,

        margin={
            "l": 40,
            "r": 40,
            "t": 40,
            "b": 40
        }
    )

    return fig


# ============================================================
# INSIGHTS
# ============================================================

def generate_insights(
    study,
    sleep,
    social,
    physical,
    extra,
    stress
):

    insights = []


    if study < 3:

        insights.append(
            "📚 Your study time is relatively low. "
            "Consider adding one focused study block each day."
        )

    elif study >= 6:

        insights.append(
            "📚 Your study commitment is strong. "
            "Focus on consistency and active practice."
        )


    if sleep < 6:

        insights.append(
            "😴 Your sleep duration is relatively low. "
            "A consistent sleep routine can support academic performance."
        )

    elif sleep >= 7:

        insights.append(
            "😴 Your sleep duration is within a commonly recommended range. "
            "Try to maintain consistency."
        )


    if social > 5:

        insights.append(
            "👥 A significant amount of time is allocated "
            "to social activity. Protect your focused study blocks."
        )


    if physical < 1:

        insights.append(
            "🏃 Your physical activity is relatively low. "
            "Consider adding regular movement to your routine."
        )


    stress_lower = str(
        stress
    ).strip().lower()


    if stress_lower == "high":

        insights.append(
            "🧠 You reported high stress. "
            "Breaking large academic tasks into smaller goals may help."
        )

    elif stress_lower == "medium":

        insights.append(
            "🧠 You reported moderate stress. "
            "Planning tasks in smaller blocks may help."
        )

    elif stress_lower == "low":

        insights.append(
            "🧠 You reported low stress."
        )


    if not insights:

        insights.append(
            "✨ No major routine warning was detected "
            "by the current rule-based checks."
        )


    return insights


# ============================================================
# IMPROVEMENT PLAN
# ============================================================

def generate_plan(
    study,
    sleep,
    social,
    physical,
    extra,
    stress,
    gpa
):

    plan = []


    if study < 3:

        plan.append(
            "📚 Gradually increase focused study time."
        )

    elif study < 5:

        plan.append(
            "📚 Build a fixed daily study routine."
        )

    else:

        plan.append(
            "📚 Maintain your study consistency."
        )


    if sleep < 6:

        plan.append(
            "😴 Work toward a more consistent sleep schedule."
        )


    if physical < 1:

        plan.append(
            "🏃 Add regular physical activity to your routine."
        )


    if social > 5:

        plan.append(
            "📱 Review unnecessary social/distraction time."
        )


    if str(stress).strip().lower() == "high":

        plan.append(
            "🧠 Break large tasks into smaller goals "
            "and use short recovery breaks."
        )


    if gpa < GPA_SCALE_MAX * 0.55:

        plan.append(
            "🚀 Start with one or two manageable habits "
            "instead of changing everything at once."
        )

    elif gpa < GPA_SCALE_MAX * 0.70:

        plan.append(
            "🎯 Focus on consistency, revision "
            "and reducing unnecessary distractions."
        )

    else:

        plan.append(
            "🏆 Maintain the habits that are already working."
        )


    return plan


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    if not hasattr(
        model,
        "named_steps"
    ):

        return None


    steps = model.named_steps


    if "model" not in steps:

        return None


    trained_model = steps["model"]


    # --------------------------------------------------------
    # Tree models
    # --------------------------------------------------------

    if hasattr(
        trained_model,
        "feature_importances_"
    ):

        importance = np.asarray(
            trained_model.feature_importances_
        )


    # --------------------------------------------------------
    # Linear models
    # --------------------------------------------------------

    elif hasattr(
        trained_model,
        "coef_"
    ):

        importance = np.abs(
            np.asarray(
                trained_model.coef_
            )
        )


    else:

        return None


    importance = np.ravel(
        importance
    )


    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = None


    if "preprocessor" in steps:

        try:

            feature_names = (
                steps["preprocessor"]
                .get_feature_names_out()
            )

        except Exception:

            feature_names = None


    if feature_names is None:

        feature_names = [

            f"Feature {i + 1}"

            for i in range(
                len(importance)
            )

        ]


    if len(feature_names) != len(importance):

        return None


    result = pd.DataFrame(

        {
            "Feature": feature_names,

            "Importance": importance
        }
    )


    return result.sort_values(
        "Importance",
        ascending=False
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 FocusGuard")

    st.caption(
        "AI Student Intelligence"
    )

    st.divider()


    page = st.radio(

        "Navigation",

        [
            "🏠 Overview",
            "🎯 GPA Predictor",
            "🧪 What-If Lab",
            "📊 Student Analytics",
            "🤖 AI Explanation",
            "🧠 Model Intelligence",
        ]
    )


    st.divider()

    st.subheader(
        "Current Model"
    )

    st.write(
        model_info.get(
            "model_name",
            "Unknown"
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "R²",
            f"{float(model_info.get('r2', 0)):.3f}"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{float(model_info.get('rmse', 0)):.3f}"
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 FocusGuard AI"
)

st.markdown(
    "**Student Performance Intelligence** • "
    "Predictive Analytics • Personalized Insights"
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        "🎓 System Overview"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Students",
            f"{len(df):,}"
        )


    with col2:

        st.metric(
            "Average GPA",
            f"{df['GPA'].mean():.2f}"
        )


    with col3:

        st.metric(
            "Model R²",
            f"{float(model_info.get('r2', 0)):.3f}"
        )


    with col4:

        st.metric(
            "ML Model",
            model_info.get(
                "model_name",
                "ML Model"
            )
        )


    st.subheader(
        "⚙️ How FocusGuard Works"
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.info(
            "👤 **Student Habits**\n\n"
            "Daily routine and lifestyle data."
        )


    with c2:

        st.info(
            "🧹 **Data Processing**\n\n"
            "Cleaning and feature engineering."
        )


    with c3:

        st.info(
            "🤖 **ML Model**\n\n"
            "Machine learning prediction."
        )


    with c4:

        st.info(
            "🎯 **Prediction**\n\n"
            "Estimated academic performance."
        )


    with c5:

        st.info(
            "💡 **Insights**\n\n"
            "Explore routine patterns."
        )


    st.subheader(
        "✨ Explore FocusGuard"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.success(
            "🎯 **GPA Predictor**\n\n"
            "Estimate GPA from student habits."
        )


    with c2:

        st.warning(
            "🧪 **What-If Lab**\n\n"
            "Experiment with different routines."
        )


    with c3:

        st.info(
            "📊 **Student Analytics**\n\n"
            "Explore dataset patterns."
        )


# ============================================================
# GPA PREDICTOR
# ============================================================

elif page == "🎯 GPA Predictor":

    st.header(
        "🎯 GPA Predictor"
    )

    st.write(
        "Enter a daily routine and let the trained ML model "
        "estimate GPA."
    )


    left, right = st.columns(2)


    with left:

        study = st.slider(
            "📚 Study Hours / Day",
            0.0,
            12.0,
            4.0,
            0.5
        )


        extra = st.slider(
            "🎨 Extracurricular Hours / Day",
            0.0,
            8.0,
            1.0,
            0.5
        )


        sleep = st.slider(
            "😴 Sleep Hours / Day",
            0.0,
            12.0,
            7.0,
            0.5
        )


    with right:

        social = st.slider(
            "👥 Social Hours / Day",
            0.0,
            10.0,
            2.0,
            0.5
        )


        physical = st.slider(
            "🏃 Physical Activity Hours / Day",
            0.0,
            8.0,
            1.0,
            0.5
        )


        stress = st.selectbox(
            "🧠 Stress Level",
            STRESS_OPTIONS
        )


    total = total_daily_hours(
        study,
        extra,
        sleep,
        social,
        physical
    )


    st.caption(
        f"Total tracked time: **{total:.1f} / 24 hours**"
    )


    if total > 24:

        st.error(
            "Your routine exceeds 24 hours."
        )


    else:

        if st.button(
            "🚀 Predict GPA",
            type="primary"
        ):

            try:

                prediction = predict_gpa(
                    study,
                    extra,
                    sleep,
                    social,
                    physical,
                    stress
                )


                status, icon = get_status(
                    prediction
                )


                st.divider()


                c1, c2 = st.columns(2)


                with c1:

                    st.metric(
                        "Predicted GPA",
                        f"{prediction:.2f} / {GPA_SCALE_MAX:g}"
                    )


                    st.success(
                        f"{icon} {status}"
                    )


                with c2:

                    st.plotly_chart(
                        create_gauge(
                            prediction
                        ),
                        use_container_width=True
                    )


                st.subheader(
                    "💡 Personalized Insights"
                )


                insights = generate_insights(
                    study,
                    sleep,
                    social,
                    physical,
                    extra,
                    stress
                )


                for insight in insights:

                    st.info(
                        insight
                    )


                st.subheader(
                    "🚀 Improvement Plan"
                )


                plan = generate_plan(
                    study,
                    sleep,
                    social,
                    physical,
                    extra,
                    stress,
                    prediction
                )


                for item in plan:

                    st.write(
                        f"• {item}"
                    )


            except Exception as e:

                st.error(
                    f"Prediction failed:\n\n{e}"
                )


# ============================================================
# WHAT-IF LAB
# ============================================================

elif page == "🧪 What-If Lab":

    st.header(
        "🧪 What-If Lab"
    )

    st.write(
        "Compare two daily routines and see how "
        "the model prediction changes."
    )


    col1, col2 = st.columns(2)


    # ========================================================
    # ROUTINE A
    # ========================================================

    with col1:

        st.subheader(
            "Routine A"
        )


        a_study = st.slider(
            "Study A",
            0.0,
            12.0,
            3.0,
            0.5
        )


        a_sleep = st.slider(
            "Sleep A",
            0.0,
            12.0,
            6.0,
            0.5
        )


        a_social = st.slider(
            "Social A",
            0.0,
            10.0,
            3.0,
            0.5
        )


        a_physical = st.slider(
            "Physical A",
            0.0,
            8.0,
            1.0,
            0.5
        )


        a_extra = st.slider(
            "Extra A",
            0.0,
            8.0,
            1.0,
            0.5
        )


        a_stress = st.selectbox(
            "Stress A",
            STRESS_OPTIONS,
            key="stress_a"
        )


    # ========================================================
    # ROUTINE B
    # ========================================================

    with col2:

        st.subheader(
            "Routine B"
        )


        b_study = st.slider(
            "Study B",
            0.0,
            12.0,
            6.0,
            0.5
        )


        b_sleep = st.slider(
            "Sleep B",
            0.0,
            12.0,
            7.0,
            0.5
        )


        b_social = st.slider(
            "Social B",
            0.0,
            10.0,
            2.0,
            0.5
        )


        b_physical = st.slider(
            "Physical B",
            0.0,
            8.0,
            1.0,
            0.5
        )


        b_extra = st.slider(
            "Extra B",
            0.0,
            8.0,
            1.0,
            0.5
        )


        b_stress = st.selectbox(
            "Stress B",
            STRESS_OPTIONS,
            key="stress_b"
        )


    if st.button(
        "🔬 Compare Routines",
        type="primary"
    ):

        total_a = total_daily_hours(
            a_study,
            a_extra,
            a_sleep,
            a_social,
            a_physical
        )


        total_b = total_daily_hours(
            b_study,
            b_extra,
            b_sleep,
            b_social,
            b_physical
        )


        if total_a > 24 or total_b > 24:

            st.error(
                "One of the routines exceeds 24 hours."
            )


        else:

            try:

                gpa_a = predict_gpa(
                    a_study,
                    a_extra,
                    a_sleep,
                    a_social,
                    a_physical,
                    a_stress
                )


                gpa_b = predict_gpa(
                    b_study,
                    b_extra,
                    b_sleep,
                    b_social,
                    b_physical,
                    b_stress
                )


                c1, c2, c3 = st.columns(3)


                with c1:

                    st.metric(
                        "Routine A",
                        f"{gpa_a:.2f}"
                    )


                with c2:

                    st.metric(
                        "Routine B",
                        f"{gpa_b:.2f}"
                    )


                with c3:

                    difference = gpa_b - gpa_a

                    st.metric(
                        "Prediction Difference",
                        f"{difference:+.2f}"
                    )


                st.info(
                    "This comparison shows a difference in "
                    "model predictions. It does not prove that "
                    "changing a habit will cause the predicted GPA change."
                )


            except Exception as e:

                st.error(
                    f"Comparison failed:\n\n{e}"
                )


# ============================================================
# STUDENT ANALYTICS
# ============================================================

elif page == "📊 Student Analytics":

    st.header(
        "📊 Student Analytics"
    )

    st.write(
        "Explore relationships between student habits and GPA."
    )


    # --------------------------------------------------------
    # GPA Distribution
    # --------------------------------------------------------

    st.subheader(
        "📈 GPA Distribution"
    )


    fig_gpa = go.Figure()


    fig_gpa.add_trace(

        go.Histogram(

            x=df["GPA"],

            nbinsx=20,

            name="GPA"
        )
    )


    fig_gpa.update_layout(

        template="plotly_dark",

        xaxis_title="GPA",

        yaxis_title="Number of Students",

        height=400
    )


    st.plotly_chart(
        fig_gpa,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Study vs GPA
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        st.subheader(
            "Study Hours vs GPA"
        )


        fig1 = go.Figure()


        fig1.add_trace(

            go.Scatter(

                x=df["Study_Hours_Per_Day"],

                y=df["GPA"],

                mode="markers",

                name="Students"
            )
        )


        fig1.update_layout(

            template="plotly_dark",

            xaxis_title="Study Hours / Day",

            yaxis_title="GPA",

            height=400
        )


        st.plotly_chart(
            fig1,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Sleep vs GPA
    # --------------------------------------------------------

    with c2:

        st.subheader(
            "Sleep Hours vs GPA"
        )


        fig2 = go.Figure()


        fig2.add_trace(

            go.Scatter(

                x=df["Sleep_Hours_Per_Day"],

                y=df["GPA"],

                mode="markers",

                name="Students"
            )
        )


        fig2.update_layout(

            template="plotly_dark",

            xaxis_title="Sleep Hours / Day",

            yaxis_title="GPA",

            height=400
        )


        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    st.subheader(
        "🔗 Feature Correlation"
    )


    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns


    correlation = df[
        numeric_columns
    ].corr()


    fig_corr = go.Figure()


    fig_corr.add_trace(

        go.Heatmap(

            z=correlation.values,

            x=correlation.columns,

            y=correlation.columns,

            text=np.round(
                correlation.values,
                2
            ),

            texttemplate="%{text}",

            colorscale="Viridis"
        )
    )


    fig_corr.update_layout(

        template="plotly_dark",

        height=600
    )


    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------------

    st.subheader(
        "📋 Dataset Preview"
    )


    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# AI EXPLANATION
# ============================================================

elif page == "🤖 AI Explanation":

    st.header(
        "🤖 AI Explanation"
    )


    st.write(
        "FocusGuard combines machine-learning prediction "
        "with simple rule-based explanations."
    )


    st.info(
        """
        **How the system works**

        1. Student provides daily routine information.
        2. Input data is converted into model features.
        3. The trained ML model predicts GPA.
        4. Rule-based logic identifies possible improvement areas.
        5. The system provides personalized suggestions.
        """
    )


    st.subheader(
        "Model Information"
    )


    info_df = pd.DataFrame(

        {
            "Metric": [

                "Model",

                "Target",

                "Training Samples",

                "Testing Samples",

                "MAE",

                "RMSE",

                "R²"

            ],

            "Value": [

                model_info.get(
                    "model_name",
                    "Unknown"
                ),

                model_info.get(
                    "target",
                    "GPA"
                ),

                model_info.get(
                    "training_samples",
                    0
                ),

                model_info.get(
                    "testing_samples",
                    0
                ),

                model_info.get(
                    "mae",
                    0
                ),

                model_info.get(
                    "rmse",
                    0
                ),

                model_info.get(
                    "r2",
                    0
                )

            ]
        }
    )


    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "⚠️ Responsible Interpretation"
    )


    st.warning(
        "The GPA prediction is an estimate based on patterns "
        "learned from the training dataset. It does not establish "
        "that a specific lifestyle habit directly causes a change in GPA."
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    st.header(
        "🧠 Model Intelligence"
    )


    importance_df = get_feature_importance()


    if importance_df is None:

        st.warning(
            "Feature importance is not available for this model."
        )


    else:

        st.subheader(
            "Feature Importance"
        )


        fig = go.Figure()


        fig.add_trace(

            go.Bar(

                x=importance_df["Importance"],

                y=importance_df["Feature"],

                orientation="h"
            )
        )


        fig.update_layout(

            template="plotly_dark",

            height=500,

            xaxis_title="Importance",

            yaxis_title="Feature"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(

            importance_df,

            use_container_width=True,

            hide_index=True
        )


    # --------------------------------------------------------
    # Model Performance
    # --------------------------------------------------------

    st.subheader(
        "📌 Model Performance"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "R²",
            f"{float(model_info.get('r2', 0)):.3f}"
        )


    with c2:

        st.metric(
            "RMSE",
            f"{float(model_info.get('rmse', 0)):.3f}"
        )


    with c3:

        st.metric(
            "MAE",
            f"{float(model_info.get('mae', 0)):.3f}"
        )


    # --------------------------------------------------------
    # Pipeline Information
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Pipeline Structure"
    )


    if hasattr(model, "named_steps"):

        st.success(
            "The saved model is a Scikit-learn Pipeline."
        )


        st.write(
            "Pipeline steps:"
        )


        for step_name in model.named_steps:

            st.code(
                step_name
            )

    else:

        st.warning(
            "The saved model is not a Scikit-learn Pipeline."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 FocusGuard AI • Student Performance Intelligence System"
)
