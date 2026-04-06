# 🧠 Kidney Disease Classification — Deep Learning + AWS CI/CD

> An end-to-end deep learning project that classifies kidney CT scans into 4 categories — Cyst, Tumor, Stone, and Normal — deployed live on AWS with a fully automated CI/CD pipeline.

🌐 **Live Demo:** http://13.234.162.95:8080/

---

## 📌 Overview

This project builds a **production-grade image classification pipeline** using transfer learning on CT scan images to detect kidney conditions. It goes beyond a simple notebook — the trained model is containerized, pushed to AWS ECR, and automatically deployed to EC2 every time code is pushed to GitHub.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.21, Keras 3.14, EfficientNetB0 |
| Web Framework | Flask, Flask-CORS |
| Containerization | Docker |
| Cloud | AWS EC2, AWS ECR, AWS IAM, Elastic IP |
| CI/CD | GitHub Actions (self-hosted runner on EC2) |
| Dataset | [CT Kidney Dataset — Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | EfficientNetB0 + Custom Head |
| Training Strategy | 2-Phase (Frozen → Fine-tuned) |
| Total Images | 12,446 CT scans |
| Classes | Cyst, Tumor, Stone, Normal |
| Test Accuracy | **99.36%** |
| Tumor Recall | **100.00%** |
| Stone Recall | **100.00%** |

---

## 📂 Project Structure
├── .github/workflows/
│   └── main.yaml                  # CI/CD pipeline
├── artifacts/
│   └── training/
│       └── kidney_savedmodel/     # Trained model (TF SavedModel)
├── config/
│   └── config.yaml
├── src/cnnClassifier/
│   ├── components/
│   │   └── model_prediction.py
│   └── ...
├── templates/
│   └── index.html                 # Frontend UI
├── app.py                         # Flask app
├── Dockerfile
├── params.yaml
└── requirements.txt

---

## ⚙️ Local Setup
```bash
# Clone the repo
git clone https://github.com/Naren1704/Deep-Learning-Project-with-AWS-CICD-Deployment.git
cd Deep-Learning-Project-with-AWS-CICD-Deployment

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Open `http://127.0.0.1:8080` in your browser.

---

## 🚀 AWS CI/CD Deployment

### Architecture
GitHub Push → GitHub Actions → Build Docker Image
→ Push to AWS ECR → Pull on EC2 → Run Container → Live at Elastic IP:8080

### AWS Services Used

| Service | Purpose |
|---------|---------|
| **IAM** | Created a user with `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess` policies. Generated access keys for GitHub Secrets. |
| **ECR (Elastic Container Registry)** | Private Docker registry to store and version the application image. |
| **EC2 (t2.medium, Ubuntu 24.04)** | Hosts the Docker container and runs the GitHub Actions self-hosted runner as a background service. |
| **Elastic IP** | Fixed public IP (`13.234.162.95`) attached to EC2 so the URL never changes across restarts. |

---

### GitHub Secrets Required

Go to your repo → **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `ap-south-1` |
| `AWS_ECR_LOGIN_URI` | e.g. `050752616680.dkr.ecr.ap-south-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | e.g. `kidney-classifier` |

---

### EC2 Self-Hosted Runner Setup

Connect to EC2 and run these **once**:
```bash
# Install Docker
sudo apt-get update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

Then go to GitHub → **Settings → Actions → Runners → New self-hosted runner → Linux** and run each command shown. Finally install as a persistent service:
```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

The runner now survives terminal disconnects and EC2 reboots.

---

### ⚠️ First Deployment Note

On your **very first deployment**, the `continuous-deployment` job will fail because no image exists in ECR yet. This is expected. Simply comment out that job in `.github/workflows/main.yaml` for the first push:
```yaml
# continuous-deployment:     # ← comment this entire job out for first push
#   name: Continuous Deployment
#   ...
```

Push once to build and push the image to ECR. Then **uncomment** the job and push again. From that point on, the full pipeline runs automatically end to end.

---

### CI/CD Pipeline Flow

The `.github/workflows/main.yaml` runs 3 jobs on every push to `main`:

1. **Continuous Integration** — linting and unit test checks
2. **Continuous Delivery** — builds Docker image and pushes to ECR (runs on GitHub's servers)
3. **Continuous Deployment** — pulls image from ECR and runs container on EC2 (runs on self-hosted EC2 runner)

---

## 👤 Author

- **Krish Naik** (Original Project Guide)
- **Narendren S V** (Implementation — model, pipeline, AWS deployment)
