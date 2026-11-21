

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict

from src.config import RANDOM_STATE, ARTIFACT_WORDS, MODELS_DIR, MODEL_PATH

from src.config import RANDOM_STATE, ARTIFACT_WORDS
from src.preprocess import build_pipeline


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "pr_auc": "average_precision",
}


def get_models():
    """The baseline and the candidate models."""
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }


def compare_models(X_train, y_train):
    """5-fold stratified CV on the training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        pipe = build_pipeline(model)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
        rows.append(row)
        print("Done:", name)
    return pd.DataFrame(rows).set_index("model").round(4)





PARAM_GRIDS = {
    "Linear SVM": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "model__C": [0.1, 0.5, 1, 5],
    },
    "Logistic Regression": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "model__C": [10, 50, 100, 200, 500],
    },
}


def tune_model(name, X_train, y_train):
    """Grid search with 5-fold CV on the training data only."""
    model = get_models()[name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(model),
        PARAM_GRIDS[name],
        scoring="f1",
        cv=cv,
    )
    search.fit(X_train, y_train)
    print(f"\n{name}")
    print("Best CV F1:", round(search.best_score_, 4))
    std = search.cv_results_["std_test_score"][search.best_index_]
    print("CV F1 std:", round(std, 4))
    print("Best params:", search.best_params_)
    return search




def train_final(X_train, y_train):
    """Fit the chosen final model on the full training set."""
    model = LogisticRegression(C=100, max_iter=1000, random_state=RANDOM_STATE)
    pipe = build_pipeline(model)
    pipe.fit(X_train, y_train)
    return pipe




def threshold_table(X_train, y_train, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Out-of-fold results at different thresholds, using the training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(
        LogisticRegression(C=100, max_iter=1000, random_state=RANDOM_STATE)
    )
    proba = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    y = np.asarray(y_train)

    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        rows.append({
            "threshold": t,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0,
            "recall": tp / (tp + fn),
        })
    return pd.DataFrame(rows).set_index("threshold").round(4)





def ablation_cv(X_train, y_train):
    """CV on the training data with and without the Enron/year words."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for label, extra in [("All words", None),
                         ("Without Enron/year words", ARTIFACT_WORDS)]:
        model = LogisticRegression(C=100, max_iter=1000, random_state=RANDOM_STATE)
        pipe = build_pipeline(model, extra_stop_words=extra)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"setup": label}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("setup").round(4)



def save_model(model):
    """Save the fitted pipeline to the models folder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved:", MODEL_PATH)



    