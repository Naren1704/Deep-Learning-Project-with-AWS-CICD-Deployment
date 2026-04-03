# 🧠 Kidney Disease Classification using Deep Learning

> A deep learning-based end-to-end project for detecting Chronic Kidney Disease (CKD) from medical data.

---

## 📌 Overview

This project focuses on building a **machine learning/deep learning pipeline** to classify whether a patient has **Chronic Kidney Disease (CKD)** based on clinical parameters.

The project follows a complete workflow including:

* Data ingestion
* Data validation
* Data transformation
* Model training
* Model evaluation
* Model deployment

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras
* **Visualization:** Matplotlib, Seaborn
* **Framework:** Flask (for deployment)
* **Tools:** Git, Docker (optional), Jupyter Notebook

---

## 📂 Project Structure

```id="3x2b1k"
Kidney-Disease-Classification/
│── .github/workflows/        # CI/CD pipelines
│── artifacts/                # Saved models & outputs
│── config/                   # Configuration files
│── logs/                     # Logging files
│── notebooks/                # Jupyter notebooks (EDA & experiments)
│── src/                      # Source code (modular pipeline)
│   │── components/
│   │── pipeline/
│   │── utils/
│── templates/                # HTML templates (for Flask app)
│── static/                   # Static files (CSS, JS)
│── app.py                    # Flask application
│── main.py                   # Pipeline execution entry point
│── requirements.txt
│── setup.py
│── README.md
```

---

## ⚙️ Installation

```bash id="7dlj2m"
# Clone the repository
git clone https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project.git

# Navigate to the project folder
cd Kidney-Disease-Classification-Deep-Learning-Project

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Training Pipeline

```bash id="q2xv1a"
python main.py
```

### Run Flask App

```bash id="m1kz9p"
python app.py
```

Then open in browser:

```
http://127.0.0.1:5000/
```

---

## 📊 Dataset

* **Domain:** Healthcare
* **Type:** Structured clinical dataset
* **Target Variable:** CKD (Yes/No)
* **Features Include:**

  * Blood Pressure
  * Sugar Levels
  * Serum Creatinine
  * Hemoglobin
  * Age

---

## 🧠 Methodology

1. **Data Ingestion**

   * Load dataset from source

2. **Data Validation**

   * Check schema, missing values

3. **Data Transformation**

   * Encoding categorical variables
   * Scaling numerical features

4. **Model Training**

   * Deep Learning model (Neural Network)

5. **Model Evaluation**

   * Accuracy
   * Confusion Matrix
   * Precision / Recall

6. **Deployment**

   * Flask-based web interface

---

## 📈 Model Used

* Artificial Neural Network (ANN)
* Alternative baseline: Logistic Regression (optional)

---

## 📊 Results

* Achieved high classification accuracy on CKD prediction
* Model successfully distinguishes between CKD and non-CKD patients
* Visualizations include:

  * Confusion Matrix
  * Feature distribution plots

---

## ✨ Features

* End-to-end ML pipeline
* Modular code structure
* Logging & exception handling
* Flask deployment
* Scalable architecture

---


## 🚀 Future Improvements

* Deploy on cloud (AWS / Azure)
* Add real-time prediction API
* Improve model accuracy with advanced architectures

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and submit PRs.

---

## 📜 License

This project is for educational purposes.

---

## 👤 Author

* **Krish Naik (Original Project Guide)**
* **Narendren S V (Implementation)**

---
