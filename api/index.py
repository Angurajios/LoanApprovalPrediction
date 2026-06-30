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

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "Loan_approval_pred_model.pkl")

print("=" * 60)
print("Python Version :", os.sys.version)
print("Base Directory :", BASE_DIR)
print("Model Path     :", MODEL_PATH)
print("Model Exists   :", os.path.exists(MODEL_PATH))
print("=" * 60)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("✅ Model Loaded Successfully")
    print("Model Type :", type(model))

except Exception:
    print("❌ Error Loading Model")
    traceback.print_exc()
    raise


# ----------------------------------------------------
# Home Page
# ----------------------------------------------------
@app.route("/")
def home():
    return render_template("loan_eligibility.html")


# ----------------------------------------------------
# Prediction API
# ----------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        required_fields = [
            "age",
            "income",
            "credit_score",
            "years_employed",
            "debt_ratio",
            "num_accounts"
        ]

        missing = [field for field in required_fields if field not in data]

        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
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

        prediction = int(prediction[0])

        return jsonify({
            "prediction": prediction,
            "eligibility": "Eligible" if prediction == 1 else "Not Eligible"
        })

    except Exception:
        traceback.print_exc()

        return jsonify({
            "error": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=True)