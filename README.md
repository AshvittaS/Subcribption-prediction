# Bank Term-Deposit Subscription Prediction

This project predicts whether a client will subscribe to a bank term deposit using an Artificial Neural Network (ANN). It includes a complete data-to-deployment workflow: data preparation, model development with hyperparameter tuning, artifact exporting, and a Gradio app deployed on Hugging Face Spaces.

- Live demo: [`Hugging Face Space`](https://huggingface.co/spaces/Ashvitta07/bank-addition)

## What the notebook does
- Cleans and prepares the Bank Marketing dataset (`bank-additional-full.csv`).
- Encodes categorical variables using a consistent feature schema saved to `columns.pkl`.
- Applies numerical transforms: PowerTransformer (`pt.pkl`) followed by scaling (`scaler.pkl`).
- Builds and trains an ANN classifier (`ann_model.h5`).
- Tunes hyperparameters with Keras Tuner (results in `tuner_results/`).
- Evaluates performance and sets a probability threshold of 0.5 for classification.
- Exports all preprocessing and modeling artifacts for reproducible inference in the app.

## Unique aspects in this work
- Consistent feature alignment: inference aligns inputs to the exact `feature_columns` order, preventing training–serving skew.
- Hybrid preprocessing: Power transformation for numerics to stabilize variance, then global scaling for model-friendly distributions.
- Human-friendly encodings: ordinal mappings for education and boolean fields centered around 0, aiding network convergence.
- Robust one-hot handling: unseen categories default to 0 via pre-created dummy columns to avoid runtime errors.
- Hyperparameter tuning artifacts retained: full tuner trials saved for traceability and future model selection.
- Clear probability output: app displays both class decision and calibrated probability for transparency.

## Artifacts produced
- `ann_model.h5`: Trained ANN classifier.
- `pt.pkl`: Fitted PowerTransformer for numeric features.
- `scaler.pkl`: Fitted scaler applied after power transform.
- `columns.pkl`: Ordered feature list used by both training and inference.
- `tuner_results/`: Hyperparameter search logs and top models.

## App
- Frontend built with Gradio for interactive predictions.
- Encodes inputs, applies the exact preprocessing pipeline, and returns the decision plus probability.
- Try it here: [`Hugging Face Space`](https://huggingface.co/spaces/Ashvitta07/bank-addition)
