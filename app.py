import gradio as gr
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model and preprocessing objects
model = load_model("ann_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pt.pkl", "rb") as f:
    pt = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Define prediction function
def predict_subscription(age, education, default, housing, loan, month, duration,
                         campaign, pdays, previous, poutcome, emp_var_rate,
                         cons_price_idx, cons_conf_idx, euribor3m, nr_employed,
                         job, marital, contact):
    
    # Start with base columns
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
    
    # Add encoded columns (default all 0)
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = [0]

    # Set 1 for chosen job, marital, contact
    job_col = f"job_{job}"
    marital_col = f"marital_{marital}"
    contact_col = f"contact_{contact}"

    if job_col in input_dict:
        input_dict[job_col] = [1]
    if marital_col in input_dict:
        input_dict[marital_col] = [1]
    if contact_col in input_dict:
        input_dict[contact_col] = [1]

    # Create DataFrame and align with training columns
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

# Gradio interface
interface = gr.Interface(
    fn=predict_subscription,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Education (Encoded)"),
        gr.Number(label="Default (-1/0/1)"),
        gr.Number(label="Housing (-1/0/1)"),
        gr.Number(label="Loan (-1/0/1)"),
        gr.Number(label="Month (1-12)"),
        gr.Number(label="Duration"),
        gr.Number(label="Campaign"),
        gr.Number(label="Pdays"),
        gr.Number(label="Previous"),
        gr.Number(label="Poutcome (-1/0/1)"),
        gr.Number(label="Emp. Var Rate"),
        gr.Number(label="Cons. Price Idx"),
        gr.Number(label="Cons. Conf Idx"),
        gr.Number(label="Euribor3m"),
        gr.Number(label="Nr. Employed"),
        gr.Dropdown(
            ['blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
             'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
            label="Job"
        ),
        gr.Dropdown(['married', 'single', 'unknown'], label="Marital Status"),
        gr.Dropdown(['telephone', 'cellular'], label="Contact Type")
    ],
    outputs="text",
    title="💼 Bank Subscription Prediction",
    description="Enter customer details to predict whether they will subscribe to a term deposit."
)

if __name__ == "__main__":
    interface.launch()
