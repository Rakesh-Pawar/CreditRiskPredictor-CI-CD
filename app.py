import json
import logging
import os
import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)
root_path = os.path.abspath("./.")
print(("root path", root_path))

# Load the model pipeline
with open("./models/final_model.pkl", 'rb') as file:
    model_pipeline = pickle.load(file)

# Load selected features
with open("./models/selected_features.json", 'r') as file:
    selected_features = json.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    input_data = {
        'Primary_applicant_age_in_years': request.form.get('Primary_applicant_age_in_years'),
        'Gender': request.form.get('Gender'),
        'Marital_status': request.form.get('Marital_status'),
        'Number_of_dependents': request.form.get('Number_of_dependents'),
        'Housing': request.form.get('Housing'),
        'Years_at_current_residence': request.form.get('Years_at_current_residence'),
        'Employment_status': request.form.get('Employment_status'),
        'Foreign_worker': request.form.get('Foreign_worker'),
        'Savings_account_balance': request.form.get('Savings_account_balance'),
        'Has_been_employed_for_at_least (year)': request.form.get('Has_been_employed_for_at_least (year)'),
        'Months_loan_taken_for': request.form.get('Months_loan_taken_for'),
        'Purpose': request.form.get('Purpose'),
        'Principal_loan_amount': request.form.get('Principal_loan_amount'),
        'EMI_rate_in_percentage_of_disposable_income': request.form.get('EMI_rate_in_percentage_of_disposable_income'),
        'Property': request.form.get('Property'),
        'Has_coapplicant': request.form.get('Has_coapplicant'),
        'Has_guarantor': request.form.get('Has_guarantor'),
        'Number_of_existing_loans_at_this_bank': request.form.get('Number_of_existing_loans_at_this_bank'),
        'Loan_history': request.form.get('Loan_history')
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data using the pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    X_new_preprocessed = preprocessor.transform(input_df)

    # Get feature names after preprocessing
    numerical_features = preprocessor.named_transformers_['num'].feature_names_in_
    categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    preprocessed_feature_names = np.concatenate([numerical_features, categorical_features])

    # Ensure all selected features are present in the preprocessed features
    missing_features = [name for name in selected_features if name not in preprocessed_feature_names]
    if missing_features:
        logging.info(f"Missing features from preprocessed data: {missing_features}")
        raise ValueError("Some selected features are not present in the preprocessed data.")

    # Find indices of the selected features in the preprocessed feature names
    selected_feature_indices = [preprocessed_feature_names.tolist().index(name) for name in selected_features]

    # Ensure the new data contains only the selected features
    X_new_selected = X_new_preprocessed[:, selected_feature_indices]

    # Make predictions
    predictions = model_pipeline.named_steps['model'].predict(X_new_selected)
    output = predictions[0]

    # Prediction Result
    result = "Applicant has low credit risk" if output == 0 else "Applicant has high credit risk"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

