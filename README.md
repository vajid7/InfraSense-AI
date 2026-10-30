# ⚙️ InfraSense-AI
### AI-Powered Predictive Maintenance System

InfraSense-AI is a Machine Learning-based predictive maintenance application designed to predict the risk of machine failure using operational parameters. It helps identify potential failures early so that maintenance can be planned before unexpected breakdowns occur.

## 🎯 Project Objective

The main objective of InfraSense-AI is to:
- Predict machine failure risk using machine operating data.
- Identify machines that may require maintenance.
- Reduce unexpected machine downtime.
- Support better maintenance decisions through data-driven predictions.

## 📊 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**.

- Total records: 10,000
- Total columns: 12
- Target variable: `Machine failure`
- Failure cases: 339
- Non-failure cases: 9,661

### Input Features
- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** to predict machine failure.

Model development includes:
- Data preprocessing
- Feature selection
- Categorical feature encoding
- Numerical feature scaling
- Training and validation
- Model comparison
- Decision threshold optimization

The selected model configuration is **Baseline Random Forest**.

- Selected decision threshold: 0.2067
- Validation F2-score: 0.7479
- Test failure precision: 0.60
- Test failure recall: 0.76

*Note: These metrics are from the current model experiment and may change after retraining.*

## 🖥️ Dashboard

The project includes an interactive Streamlit dashboard where users can:
- Enter machine operating readings.
- Select machine type (L, M, H).
- Predict machine failure probability.
- View failure alerts.
- Get maintenance recommendations.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- Streamlit
- Matplotlib / Seaborn (if used in analysis)

## 📁 Project Structure

```text
InfraSense-AI/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── failure_model.joblib
│
├── src/
│   ├── load_data.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── .gitignore
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vajid7/InfraSense-AI.git
```

### 2. Open the project folder

```bash
cd InfraSense-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install required libraries

```bash
pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn
```

### 6. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

## ⚠️ Disclaimer

InfraSense-AI provides machine learning-based predictions. Its output is not a guarantee of machine failure and should be used as decision-support information alongside regular machine inspections.

## 👨‍💻 Author

**Vajid Khan**

GitHub: https://github.com/vajid7

---

⭐ If you find this project interesting, consider giving it a star!# InfraSense-AI
Explainable AI for Predictive Infrastructure Maintenance and Risk Assessment
