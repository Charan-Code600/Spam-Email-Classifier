





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import TEXT_COLUMN, LABEL_COLUMN, REPORTS_DIR


def audit(df):
    print("Rows, columns:", df.shape)

    print("\n--- Class ratio ---")
    counts = df[LABEL_COLUMN].value_counts()
    percent = df[LABEL_COLUMN].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "percent": percent.round(2)}))

    print("\n--- label vs label_num agreement ---")
    print(pd.crosstab(df["label"], df[LABEL_COLUMN]))

    print("\n--- Duplicates ---")
    dup_count = df.duplicated(subset=[TEXT_COLUMN]).sum()
    print("Rows with duplicate text:", dup_count)
    conflict = df.groupby(TEXT_COLUMN)[LABEL_COLUMN].nunique()
    print("Same text, different labels:", (conflict > 1).sum())

    print("\n--- Email length (words) ---")
    words = df[TEXT_COLUMN].str.split().str.len()
    print(words.groupby(df["label"]).describe().round(1))




def duplicates_by_class(df):
    print("\n--- Duplicates by class ---")
    dup_mask = df.duplicated(subset=[TEXT_COLUMN])
    print(df.loc[dup_mask, "label"].value_counts())
    print("Rows after dropping duplicates:", len(df) - dup_mask.sum())





def _top_words_series(df, label_value, n):
    texts = df.loc[df[LABEL_COLUMN] == label_value, TEXT_COLUMN]
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names_out()
    return pd.Series(counts, index=words).sort_values(ascending=False).head(n)


def top_words(df, n=15):
    for label_value, name in [(0, "ham"), (1, "spam")]:
        print(f"\n--- Top {n} words in {name} ---")
        print(_top_words_series(df, label_value, n))


def _save(file_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / file_name, dpi=150)
    plt.close()
    print("Saved:", REPORTS_DIR / file_name)


def plot_class_distribution(df):
    counts = df["label"].value_counts().reindex(["ham", "spam"])
    total = counts.sum()
    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
    for i, value in enumerate(counts.values):
        plt.text(i, value, f"{value} ({value / total * 100:.1f}%)",
                 ha="center", va="bottom")
    plt.title("Class distribution")
    plt.ylabel("Number of emails")
    _save("class_distribution.png")


def plot_length_distribution(df, clip=1000):
    words = df[TEXT_COLUMN].str.split().str.len()
    plt.figure(figsize=(7, 4))
    for name in ["ham", "spam"]:
        plt.hist(words[df["label"] == name].clip(upper=clip),
                 bins=50, alpha=0.6, label=name)
    plt.xlabel(f"Words per email (clipped at {clip})")
    plt.ylabel("Number of emails")
    plt.title("Email length by class")
    plt.legend()
    _save("email_length.png")


def plot_top_words(df, n=15):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label_value, name) in zip(axes, [(0, "ham"), (1, "spam")]):
        series = _top_words_series(df, label_value, n)
        series.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Top {n} words in {name}")
        ax.set_xlabel("Count")
    _save("top_words.png")


