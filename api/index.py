from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import onnxruntime as rt
import os
import json
import traceback

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

CORS(app)

# ----------------------------------------------------
# Load Model (ONNX — no native .so / OpenMP dependency)
# ----------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.onnx")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_order.json")

print("=" * 60)
print("Python Version :", os.sys.version)
print("Base Directory :", BASE_DIR)
print("Model Path     :", MODEL_PATH)
print("Model Exists   :", os.path.exists(MODEL_PATH))
print("=" * 60)

try:
    session = rt.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    with open(FEATURE_ORDER_PATH, "r") as f:
        FEATURE_ORDER = json.load(f)

    print("✅ ONNX Model Loaded Successfully")
    print("Input name   :", input_name)
    print("Output names :", output_names)
    print("Feature order:", FEATURE_ORDER)

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

        # Build the feature vector in the exact order the model was trained on
        row = [float(data[name]) for name in FEATURE_ORDER]
        input_arr = np.array([row], dtype=np.float32)

        results = session.run(output_names, {input_name: input_arr})

        # LightGBM->ONNX classifiers typically output [label, probability_map]
        # label is usually the first output, shape (1,)
        label_output = results[0]
        prediction = int(np.array(label_output).reshape(-1)[0])

        response = {
            "prediction": prediction,
            "eligibility": "Eligible" if prediction == 1 else "Not Eligible"
        }

        # If a probability output exists, include it (best-effort, won't break if absent)
        if len(results) > 1:
            try:
                probs = results[1]
                # probs may be a list of dicts (zipmap) or an array
                if isinstance(probs, list) and isinstance(probs[0], dict):
                    response["probability"] = probs[0]
                else:
                    response["probability"] = np.array(probs).reshape(-1).tolist()
            except Exception:
                pass

        return jsonify(response)

    except Exception:
        traceback.print_exc()

        return jsonify({
            "error": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=True)