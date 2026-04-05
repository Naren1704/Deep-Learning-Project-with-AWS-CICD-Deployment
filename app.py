import os
import boto3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnClassifier.components.model_prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

# ✅ FIXED: Use ONE consistent path
MODEL_DIR = "/app/artifacts/training/kidney_savedmodel"

def download_model_from_s3():
    if os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        print("Model already exists. Skipping download.")
        return

    print("Downloading model from S3...")

    s3 = boto3.client("s3")
    bucket = "kidney-model-bucket"
    prefix = "kidney_savedmodel/"

    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        key = obj["Key"]

        if key.endswith("/"):
            continue

        local_path = os.path.join(MODEL_DIR, key.replace(prefix, ""))

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

    print("Model downloaded successfully!")

# ✅ Ensure model exists
os.makedirs(MODEL_DIR, exist_ok=True)
download_model_from_s3()

# ✅ Use SAME path
predictor = PredictionPipeline(MODEL_DIR)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_path = "temp_upload.jpg"
    file.save(img_path)

    result = predictor.predict(img_path)
    os.remove(img_path)

    return jsonify(result)


@app.route("/train", methods=["GET"])
def train_route():
    os.system("python main.py")
    return jsonify({"status": "Training completed!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)