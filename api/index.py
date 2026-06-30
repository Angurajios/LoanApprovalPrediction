from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os
import traceback

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

CORS(app)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "Loan_approval_pred_model.pkl")

print("=" * 60)
print("BASE_DIR :", BASE_DIR)
print("MODEL_PATH :", MODEL_PATH)
print("MODEL EXISTS :", os.path.exists(MODEL_PATH))
print("=" * 60)

model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("✅ Model Loaded Successfully")
    print("Model Type :", type(model))

except Exception as e:
    print("❌ MODEL LOADING FAILED")
    traceback.print_exc()
    model = None


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("loan_eligibility.html")


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded. Check Vercel Runtime Logs."
        }), 500

    try:

        data = request.get_json()

        required = [
            "age",
            "income",
            "credit_score",
            "years_employed",
            "debt_ratio",
            "num_accounts"
        ]

        for field in required:
            if field not in data:
                return jsonify({
                    "error": f"Missing field: {field}"
                }), 400

        input_df = pd.DataFrame([{
            "age": data["age"],
            "income": data["income"],
            "credit_score": data["credit_score"],
            "years_employed": data["years_employed"],
            "debt_ratio": data["debt_ratio"],
            "num_accounts": data["num_accounts"]
        }])

        prediction = model.predict(input_df)

        result = int(prediction[0])

        return jsonify({
            "prediction": result,
            "eligibility": "Eligible" if result == 1 else "Not Eligible"
        })

    except Exception:
        traceback.print_exc()
        return jsonify({
            "error": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=True)