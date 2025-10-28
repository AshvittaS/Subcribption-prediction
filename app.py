import gradio as gr
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- Load model and preprocessing ----------------
model = load_model("ann_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pt.pkl", "rb") as f:
    pt = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---------------- Label → Numeric Mappings ----------------
edu_map = {
    "Illiterate": -2,
    "Basic (4y/6y/9y)": -1,
    "High School": 1,
    "Professional Course": 1,
    "University Degree": 2,
    "Unknown": 0
}

bool_map = {"No": -1, "Yes": 1, "Unknown": 0}
poutcome_map = {"Failure": -1, "Nonexistent": 0, "Success": 1}
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# ---------------- Prediction Function ----------------
def predict_subscription(age, job, marital, education, default, housing, loan, contact,
                         month, day_of_week, duration, campaign, pdays, previous,
                         poutcome, emp_var_rate, cons_price_idx, cons_conf_idx,
                         euribor3m, nr_employed):

    # Convert dropdown text to encoded values
    education = edu_map[education]
    default = bool_map[default]
    housing = bool_map[housing]
    loan = bool_map[loan]
    month = month_map[month]
    poutcome = poutcome_map[poutcome]

    # Base columns
    input_dict = {
        'age': [age],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed]
    }

    # Add dummy columns (0 by default)
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = [0]

    # One-hot encode job, marital, contact
    job_col = f"job_{job}"
    marital_col = f"marital_{marital}"
    contact_col = f"contact_{contact}"

    if job_col in input_dict:
        input_dict[job_col] = [1]
    if marital_col in input_dict:
        input_dict[marital_col] = [1]
    if contact_col in input_dict:
        input_dict[contact_col] = [1]

    # Create and reorder DataFrame
    input_df = pd.DataFrame(input_dict)
    input_df = input_df[feature_columns]

    # Apply same preprocessing
    num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    input_df[num_cols] = pt.transform(input_df[num_cols])
    input_df[input_df.columns] = scaler.transform(input_df[input_df.columns])

    # Predict
    pred = model.predict(input_df)
    result = "✅ Will Subscribe" if pred[0][0] > 0.5 else "❌ Will Not Subscribe"

    return f"{result}\nPrediction Probability: {pred[0][0]:.2f}"


# ---------------- Gradio Interface ----------------
interface = gr.Interface(
    fn=predict_subscription,
    inputs=[
        gr.Number(label="Age (years)"),
        gr.Dropdown(
            ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
             'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
            label="Job Type"
        ),
        gr.Dropdown(['married', 'single', 'divorced', 'unknown'], label="Marital Status"),
        gr.Dropdown(list(edu_map.keys()), label="Education Level"),
        gr.Dropdown(list(bool_map.keys()), label="Has Credit in Default?"),
        gr.Dropdown(list(bool_map.keys()), label="Has Housing Loan?"),
        gr.Dropdown(list(bool_map.keys()), label="Has Personal Loan?"),
        gr.Dropdown(['cellular', 'telephone'], label="Contact Type"),
        gr.Dropdown(list(month_map.keys()), label="Last Contact Month"),
        gr.Dropdown(['mon', 'tue', 'wed', 'thu', 'fri'], label="Day of Week"),
        gr.Number(label="Last Contact Duration (seconds)"),
        gr.Number(label="Number of Contacts in Campaign"),
        gr.Number(label="Days Since Last Contact (pdays)"),
        gr.Number(label="Number of Previous Contacts"),
        gr.Dropdown(list(poutcome_map.keys()), label="Previous Campaign Outcome"),
        gr.Number(label="Employment Variation Rate"),
        gr.Number(label="Consumer Price Index"),
        gr.Number(label="Consumer Confidence Index"),
        gr.Number(label="Euribor 3-month Rate"),
        gr.Number(label="Number of Employees")
    ],
    outputs="text",
    title="💼 Bank Subscription Prediction",
    description="Predict whether a client will subscribe to a term deposit based on personal and contact details."
)

if __name__ == "__main__":
    interface.launch()
