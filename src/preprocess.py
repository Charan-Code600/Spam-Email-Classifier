



from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from src.config import TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE


def drop_duplicates(df):
    """Keep one copy of every email text."""
    before = len(df)
    df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows, {len(df)} remain.")
    return df


def split_data(df):
    """Stratified train/test split. Nothing is fitted here."""
    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test




def build_vectorizer(extra_stop_words=None):
    """TF-IDF vectorizer (not fitted yet)."""
    stop_words = set(ENGLISH_STOP_WORDS)
    if extra_stop_words:
        stop_words |= set(extra_stop_words)
    return TfidfVectorizer(
        lowercase=True,
        stop_words=list(stop_words),
        min_df=2,
        sublinear_tf=True,
    )


def build_pipeline(model, extra_stop_words=None):
    """TF-IDF + a classifier in one pipeline (not fitted yet)."""
    return Pipeline([
        ("tfidf", build_vectorizer(extra_stop_words)),
        ("model", model),
    ])






