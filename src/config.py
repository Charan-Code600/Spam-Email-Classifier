




from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spam_ham_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.7

TEXT_COLUMN = "text"
LABEL_COLUMN = "label_num"


ARTIFACT_WORDS = [
    "enron", "hpl", "hou", "ect", "daren", "deal", "meter", "gas",
    "nom", "mmbtu", "xls", "2000", "2001", "2004", "2005",
]



