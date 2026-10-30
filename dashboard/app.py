
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# =====================================
# PATHS AND CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "failure_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"

st.set_page_config(
    page_title="InfraSense-AI",
    page_icon="⚙️",
    layout="wide"
)

# =====================================
# PROFESSIONAL STYLING
# =====================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
    }
    .main-title {
        font-size: 36px;
        font-weight: 750;
    }
    .subtitle {
        color: #808080;
        font-size: 17px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">⚙️ InfraSense-AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-Powered Predictive Maintenance</div>',
    unsafe_allow_html=True
)

st.write(
    "Machine learning se failure risk predict karo, "
    "dataset analyze karo aur model performance dekho."
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

if not MODEL_PATH.exists():
    st.error(f"Model file nahi mili: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return (
        artifact["model"],
        artifact["threshold"],
        artifact["features"]
    )


try:
    model, threshold, features = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

# =====================================
# LOAD DATASET
# =====================================

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


dataset = None

if DATA_PATH.exists():
    try:
        dataset = load_dataset(DATA_PATH)
    except Exception as e:
        st.warning(f"Dataset load nahi hua: {e}")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Dashboard Section",
    [
        "Overview",
        "Failure Prediction",
        "Dataset Analysis",
        "Model Performance",
        "Prediction History"
    ]
)

st.sidebar.divider()
st.sidebar.write(f"Decision threshold: {threshold:.2%}")
st.sidebar.caption("InfraSense-AI | Predictive Maintenance")

# =====================================
# SESSION STATE
# =====================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =====================================
# OVERVIEW
# =====================================

if page == "Overview":

    st.header("📊 Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("ML Model", "Random Forest")
    c2.metric("Decision Threshold", f"{threshold:.2%}")
    c3.metric(
        "Predictions This Session",
        len(st.session_state.prediction_history)
    )

    if dataset is not None:
        st.subheader("Dataset Summary")

        if "Machine failure" in dataset.columns:
            failures = int(dataset["Machine failure"].sum())
            total = len(dataset)
            safe = total - failures

            a, b, c = st.columns(3)
            a.metric("Total Records", f"{total:,}")
            b.metric("No-Failure Records", f"{safe:,}")
            c.metric("Failure Records", f"{failures:,}")

            counts = pd.DataFrame({
                "Machine Status": ["No Failure", "Failure"],
                "Records": [safe, failures]
            })

            st.bar_chart(
                counts.set_index("Machine Status")
            )

            if total:
                st.caption(
                    f"Dataset failure rate: {failures / total:.2%}"
                )
        else:
            st.warning(
                "Dataset mein 'Machine failure' column nahi mila."
            )
    else:
        st.info(
            "Dataset summary ke liye cleaned_dataset.csv file "
            "data/processed folder mein honi chahiye."
        )

    st.subheader("Project Features")

    st.markdown("""
    - 🤖 Random Forest machine failure prediction
    - 🚦 Low, Medium aur High risk indicator
    - 📈 Machine readings chart
    - 🕘 Prediction history
    - 📁 CSV export
    - 📊 Dataset aur model performance analysis
    """)

# =====================================
# FAILURE PREDICTION
# =====================================

elif page == "Failure Prediction":

    st.header("🔍 Machine Failure Prediction")

    st.info(
        f"Model decision threshold: {threshold:.2%}"
    )

    with st.form("machine_readings"):

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        col1, col2 = st.columns(2)

        with col1:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=0.0,
                value=300.0,
                step=0.1
            )

            speed = st.number_input(
                "Rotational Speed (RPM)",
                min_value=0,
                value=1500
            )

            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,
                value=100
            )

        with col2:
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=0.0,
                value=310.0,
                step=0.1
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                value=40.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Predict Failure Risk",
            use_container_width=True
        )

    if submitted:

        if process_temp < air_temp:
            st.warning(
                "Process temperature air temperature se kam hai. "
                "Readings check karo."
            )

        else:
            readings = {
                "Type": machine_type,
                "Air temperature": air_temp,
                "Process temperature": process_temp,
                "Rotational speed": speed,
                "Torque": torque,
                "Tool wear": tool_wear
            }

            try:
                input_df = pd.DataFrame(
                    [readings],
                    columns=features
                )

                probability = float(
                    model.predict_proba(input_df)[0, 1]
                )

                if probability < 0.05:
                    risk = "LOW"
                elif probability < threshold:
                    risk = "MEDIUM"
                else:
                    risk = "HIGH"

                if probability >= threshold:
                    status = "FAILURE RISK ALERT"
                else:
                    status = "NO FAILURE ALERT"

                record = {
                    "Time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Machine Type": machine_type,
                    "Air Temp (K)": air_temp,
                    "Process Temp (K)": process_temp,
                    "Speed (RPM)": speed,
                    "Torque (Nm)": torque,
                    "Tool Wear (min)": tool_wear,
                    "Failure Probability (%)": round(
                        probability * 100, 2
                    ),
                    "Risk Level": risk,
                    "Status": status
                }

                st.session_state.prediction_history.insert(
                    0, record
                )
                st.session_state.latest_readings = readings

                st.divider()
                st.subheader("Prediction Result")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Failure Probability",
                    f"{probability:.1%}"
                )
                m2.metric(
                    "Decision Threshold",
                    f"{threshold:.1%}"
                )
                m3.metric("Risk Level", risk)

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                if probability >= threshold:
                    st.error("⚠️ FAILURE RISK ALERT")
                    st.write(
                        "Recommendation: Machine inspect karo "
                        "aur operating conditions check karo."
                    )
                else:
                    st.success("✅ NO FAILURE ALERT")
                    st.write(
                        "Recommendation: Routine monitoring "
                        "jaari rakho."
                    )

                st.subheader("Machine Readings Chart")

                chart_df = pd.DataFrame({
                    "Reading": [
                        air_temp,
                        process_temp,
                        speed,
                        torque,
                        tool_wear
                    ]
                }, index=[
                    "Air Temp (K)",
                    "Process Temp (K)",
                    "Speed (RPM)",
                    "Torque (Nm)",
                    "Tool Wear (min)"
                ])

                st.bar_chart(chart_df)

                st.caption(
                    "Readings ki units alag hain, isliye bars "
                    "ko direct same-scale comparison na samjhein."
                )

                st.caption(
                    "Prediction guarantee nahi hai. False alarms "
                    "aur missed failures possible hain."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# =====================================
# DATASET ANALYSIS
# =====================================

elif page == "Dataset Analysis":

    st.header("📁 Dataset Analysis")

    if dataset is None:
        st.error(
            "Dataset nahi mila. Pehle preprocessing script run "
            "karke cleaned_dataset.csv generate karo."
        )

    else:
        st.write(f"**Total rows:** {len(dataset):,}")
        st.write(f"**Total columns:** {len(dataset.columns)}")

        st.subheader("Dataset Preview")
        st.dataframe(
            dataset.head(20),
            use_container_width=True
        )

        st.subheader("Missing Values")

        missing_df = pd.DataFrame({
            "Column": dataset.columns,
            "Missing Values": [
                int(dataset[col].isna().sum())
                for col in dataset.columns
            ]
        })

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

        if "Machine failure" in dataset.columns:

            failures = int(dataset["Machine failure"].sum())
            no_failure = len(dataset) - failures

            a, b, c = st.columns(3)

            a.metric("Total Machines/Records", f"{len(dataset):,}")
            b.metric("No Failure", f"{no_failure:,}")
            c.metric("Failure", f"{failures:,}")

            st.subheader("Failure Distribution")

            distribution = pd.DataFrame({
                "Status": ["No Failure", "Failure"],
                "Count": [no_failure, failures]
            })

            st.bar_chart(
                distribution.set_index("Status")
            )

            if len(dataset) > 0:
                st.write(
                    f"**Failure rate:** "
                    f"{failures / len(dataset):.2%}"
                )

        st.subheader("Numeric Feature Summary")
        st.dataframe(
            dataset.describe().round(2),
            use_container_width=True
        )

# =====================================
# MODEL PERFORMANCE
# =====================================

elif page == "Model Performance":

    st.header("🎯 Model Performance Dashboard")

    st.write(
        "Neeche diye gaye metrics trained model ke held-out "
        "test set ke evaluation results hain."
    )

    # Saved test results from the model evaluation.
    # Update these values if the model is retrained.
    accuracy = 93.55
    precision = 32.96
    recall = 86.76
    f1_score = 47.77
    roc_auc = 97.21
    pr_auc = 70.18

    st.subheader("Test Set Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.2f}%")
    c2.metric("Precision", f"{precision:.2f}%")
    c3.metric("Recall", f"{recall:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1-Score", f"{f1_score:.2f}%")
    c5.metric("ROC-AUC", f"{roc_auc:.2f}%")
    c6.metric("PR-AUC", f"{pr_auc:.2f}%")

    st.subheader("Metric Comparison")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Score (%)": [
            accuracy,
            precision,
            recall,
            f1_score,
            roc_auc,
            pr_auc
        ]
    })

    st.bar_chart(
        metrics_df.set_index("Metric")
    )

    st.subheader("Test Set Confusion Matrix")

    confusion = pd.DataFrame(
        [[1812, 120], [9, 59]],
        index=["Actual No Failure", "Actual Failure"],
        columns=["Predicted No Failure", "Predicted Failure"]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.markdown("""
    **Metrics ka matlab**

    - **Accuracy:** Total predictions mein kitne correct the.
    - **Precision:** Failure alerts mein kitne actual failures the.
    - **Recall:** Actual failures mein se kitne model ne pakde.
    - **F1-score:** Precision aur recall ka combined measure.
    - **ROC-AUC:** Failure aur no-failure cases ko rank karne ki ability.
    - **PR-AUC:** Imbalanced dataset mein precision-recall performance.

    **Important:** Test results mein false alerts bhi aaye the.
    Is model ko real machinery ke safety decisions ke liye
    independent validation ke bina use nahi karna chahiye.
    """)

    st.warning(
        "Ye metrics current saved evaluation ke results hain. "
        "Agar model retrain hota hai, to in numbers ko naye "
        "test results se update karna hoga."
    )

# =====================================
# PREDICTION HISTORY
# =====================================

elif page == "Prediction History":

    st.header("🕘 Prediction History")

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="infrasense_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.session_state.pop("latest_readings", None)
            st.rerun()

    else:
        st.info(
            "Abhi history empty hai. Failure Prediction page "
            "par jaakar pehli prediction submit karo."
        )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "InfraSense-AI | Random Forest | "
    "ML-Based Predictive Maintenance"
)
