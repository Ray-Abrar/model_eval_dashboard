import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

def main():
    # Load saved data
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv", header=None).iloc[:, 0]
    preds_df = pd.read_csv("data/preds.csv")

    preds = preds_df["preds"]
    probs = preds_df["probs"]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, preds).tolist()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    metrics["confusion_matrix"] = cm
    metrics["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    # Save metrics
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to data/metrics.json")

if __name__ == "__main__":
    main()