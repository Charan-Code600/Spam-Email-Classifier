




import pandas as pd

from src.config import DATA_PATH, TEXT_COLUMN, LABEL_COLUMN


def load_raw_data(path=DATA_PATH):
    """Read the CSV as it is."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place spam_ham_dataset.csv inside the data folder."
        )
    return df


def clean_data(df):
    """Drop the unnamed index column, drop empty rows, cast label to int."""
    df = df.copy()


    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)


    needed = {"label", TEXT_COLUMN, LABEL_COLUMN}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df = df.reset_index(drop=True)
    return df


