




import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from src.config import REPORTS_DIR


def _save(file_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / file_name, dpi=150)
    plt.close()
    print("Saved:", REPORTS_DIR / file_name)


def evaluate(model, X_test, y_test, threshold):
    """Final evaluation on the test set at a fixed threshold."""
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    print(f"Threshold: {threshold}")
    print(classification_report(
        y_test, y_pred, target_names=["ham", "spam"], digits=4
    ))
    print("PR-AUC:", round(average_precision_score(y_test, y_score), 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["ham", "spam"]
    )
    plt.title(f"Confusion matrix (test set, threshold {threshold})")
    _save("confusion_matrix.png")

    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("Precision-Recall curve (test set)")
    _save("pr_curve.png")


