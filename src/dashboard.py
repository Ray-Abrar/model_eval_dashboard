import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Model Evaluation Dashboard")

# Load metrics
with open("data/metrics.json", "r") as f:
    metrics = json.load(f)

# Display metrics
st.header("Performance Metrics")
st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
st.write(f"**Precision:** {metrics['precision']:.4f}")
st.write(f"**Recall:** {metrics['recall']:.4f}")
st.write(f"**F1 Score:** {metrics['f1']:.4f}")

# Confusion matrix
st.header("Confusion Matrix")
cm = metrics["confusion_matrix"]
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot()

# ROC curve
st.header("ROC Curve")
roc = metrics["roc"]
plt.figure(figsize=(5, 4))
plt.plot(roc["fpr"], roc["tpr"], label=f"AUC = {roc['auc']:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot()