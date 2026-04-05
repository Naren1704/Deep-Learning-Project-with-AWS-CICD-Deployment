import os
import boto3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnClassifier.components.model_prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

MODEL_DIR = "/app/artifacts/training/kidney_savedmodel"
predictor = None  # will initialize after validation


def download_model_from_s3():
    print("🔍 Checking S3 bucket...")

    s3 = boto3.client("s3")
    bucket = "kidney-model-bucket"
    prefix = "kidney_savedmodel/"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        raise Exception("❌ No files found in S3 bucket. Check bucket/prefix/permissions.")

    print(f"✅ Found {len(response['Contents'])} objects in S3")

    for obj in response["Contents"]:
        key = obj["Key"]

        if key.endswith("/"):
            continue

        local_path = os.path.join(MODEL_DIR, key.replace(prefix, ""))

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"⬇️ Downloading {key} → {local_path}")
        s3.download_file(bucket, key, local_path)

    print("✅ Model download complete!")


def verify_model_exists():
    required_file = os.path.join(MODEL_DIR, "saved_model.pb")

    if not os.path.exists(required_file):
        raise Exception(f"❌ Model not found after download: {required_file}")

    print("✅ Model files verified!")


# ========================
# BOOTSTRAP (CRITICAL FLOW)
# ========================
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    download_model_from_s3()
    verify_model_exists()

    predictor = PredictionPipeline(MODEL_DIR)

except Exception as e:
    print("❌ Startup failed:", str(e))
    predictor = None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500

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
    print("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=8080)