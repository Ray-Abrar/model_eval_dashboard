# **README.md**

```markdown
# 📊 Model Evaluation Dashboard (Streamlit + Scikit‑Learn + Docker)

A production‑style machine learning evaluation dashboard that trains a model, computes performance metrics, and visualizes results through an interactive Streamlit UI.  
Fully containerized with Docker for reproducible deployment.

---

## 🚀 Overview

This project demonstrates a clean, end‑to‑end ML evaluation workflow:

- Load dataset  
- Train a classification model  
- Generate predictions + probabilities  
- Compute evaluation metrics  
- Visualize results in a dashboard  
- Package everything in Docker  

It’s designed to reflect real ML engineering practices: modular code, reproducibility, and clear separation of training, evaluation, and visualization.

---

## 🧠 Features

- ✔ Logistic Regression model (Breast Cancer dataset)  
- ✔ Train/test split  
- ✔ Metrics: Accuracy, Precision, Recall, F1  
- ✔ Confusion Matrix heatmap  
- ✔ ROC Curve + AUC  
- ✔ Streamlit dashboard  
- ✔ Dockerized for deployment  
- ✔ Clean project structure  

---

## 📂 Project Structure

```
model_eval_dashboard/
│
├── src/
│   ├── train.py          # Train model + save predictions
│   ├── evaluate.py       # Compute metrics + save JSON
│   └── dashboard.py      # Streamlit UI
│
├── data/                 # Generated after running train/evaluate
│   ├── X_test.csv
│   ├── y_test.csv
│   ├── preds.csv
│   └── metrics.json
│
├── models/
│   └── model.pkl         # Saved trained model
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Usage

### 1. Create a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

### 4. Generate evaluation metrics

```bash
python src/evaluate.py
```

### 5. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

Open in your browser:

```
http://localhost:8501
```

---

## 🐳 Run with Docker

### Build the Docker image

```bash
docker build -t model-dashboard .
```

### Run the container

```bash
docker run -p 8501:8501 model-dashboard
```

Then open:

```
http://localhost:8501
```

---

## 📈 Dashboard Preview

The dashboard includes:

- Performance metrics  
- Confusion matrix heatmap  
- ROC curve with AUC  
- Clean, interactive UI  


## 🔮 Future Enhancements

- Add Random Forest + model comparison  
- Add feature importance visualization  
- Add CI/CD pipeline (GitHub Actions)  
- Deploy to AWS / Azure / Render  
- Add experiment tracking (MLflow)  

---

## 🧑‍💻 Author

**Rahmat Abrar Mohammed**  
ML & AI Engineering | Data Systems | Python | MLOps  
GitHub: https://github.com/ray-abrar
