from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from flask_cors import CORS

loan_Prediction = Flask(__name__)
CORS(loan_Prediction)

with open("Loan_approval_pred_model.pkl", 'rb') as lp:
    model = pickle.load(lp)

@loan_Prediction.route('/')
def home():
    return render_template('loan_eligibility.html')

@loan_Prediction.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    required_fields = ['age', 'income', 'credit_score', 'years_employed', 'debt_ratio', 'num_accounts']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400
    try:
        input_data = pd.DataFrame({
            'age': [data['age']],
            'income': [data['income']],
            'credit_score': [data['credit_score']],
            'years_employed': [data['years_employed']],
            'debt_ratio': [data['debt_ratio']],
            'num_accounts': [data['num_accounts']]
        })
        prediction = int(model.predict(input_data)[0])
        eligibility = "Eligible" if prediction == 1 else "Not Eligible"
        return jsonify({'prediction': prediction, 'eligibility': eligibility})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    loan_Prediction.run(debug=True, port=5000)
