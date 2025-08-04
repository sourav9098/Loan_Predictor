from flask import Blueprint, render_template, request
import numpy as np

from .model import load_model,predict_crop
main=Blueprint('main',__name__)
model=load_model()

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        gross_income = float(request.form['gross_income'])
        tenure = int(request.form['tenure'])
        interest_rate = float(request.form['interest_rate'])
        other_emis = float(request.form['other_emis'])
        credit_score = int(request.form['credit_score'])
        age = int(request.form['age'])
        employment_type = int(request.form['employment_type'])
        loan_amount = float(request.form['loan_amount'])

        # Create feature array
        features = np.array([[gross_income, tenure, interest_rate, other_emis,
                              credit_score, age, employment_type, loan_amount]])

        # Make prediction
        prediction = predict_crop(model,features)
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

        return render_template('index.html', prediction_result=result)

    except Exception as e:
        return f"Error: {str(e)}"

