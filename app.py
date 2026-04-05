import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnClassifier.components.model_prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

import os

MODEL_PATH = os.path.join(os.getcwd(), "artifacts", "training", "kidney_savedmodel")
predictor  = PredictionPipeline(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files["file"]
    img_path = os.path.join("temp_upload.jpg")
    file.save(img_path)

    result = predictor.predict(img_path)
    os.remove(img_path)
    return jsonify(result)

@app.route("/train", methods=["GET"])
def train_route():
    os.system("python main.py")
    return jsonify({"status": "Training completed!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)