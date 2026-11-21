




import numpy as np
import pandas as pd


def get_errors(model, X_test, y_test, threshold):
    """Return the false positives and false negatives on the test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = pd.DataFrame({
        "text": X_test.values,
        "actual": y_test.values,
        "predicted": pred,
        "spam_probability": proba.round(3),
    })
    false_positives = out[(out["actual"] == 0) & (out["predicted"] == 1)]
    false_negatives = out[(out["actual"] == 1) & (out["predicted"] == 0)]
    return false_positives, false_negatives


def show_errors(errors, title, n_chars=300):
    print(f"\n=== {title}: {len(errors)} emails ===")
    for _, row in errors.iterrows():
        print(f"\n[p(spam)={row['spam_probability']}] {row['text'][:n_chars]!r}")


def top_weights(model, n=20):
    """Words with the strongest weights in the Logistic Regression model."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["model"]
    words = np.array(vectorizer.get_feature_names_out())
    weights = classifier.coef_[0]
    order = np.argsort(weights)

    print(f"\n=== Top {n} words pushing towards HAM ===")
    for i in order[:n]:
        print(f"{words[i]:<15} {weights[i]:.2f}")

    print(f"\n=== Top {n} words pushing towards SPAM ===")
    for i in order[::-1][:n]:
        print(f"{words[i]:<15} {weights[i]:.2f}")




