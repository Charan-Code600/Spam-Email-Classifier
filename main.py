




from src.config import THRESHOLD, MODEL_PATH
from src.data_loader import load_raw_data, clean_data
from src.preprocess import drop_duplicates, split_data
from src.train import train_final, save_model
from src.evaluate import evaluate
from src.predict import load_model, predict_email

MENU = """
==============================
  Spam Email Classifier
==============================
1. Train model and show test results
2. Check an email
3. Show model info
4. Exit
"""


def train_and_evaluate():
    df = drop_duplicates(clean_data(load_raw_data()))
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_final(X_train, y_train)
    save_model(model)
    evaluate(model, X_test, y_test, THRESHOLD)


def read_email():
    print("Paste the email text. Finish with a line containing only END:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def check_email():
    model = load_model()
    text = read_email()
    if len(text) < 10:
        print("Email text is too short. Please paste the full email.")
        return
    label, probability = predict_email(model, text)
    print(f"\nResult: {label.upper()}  (spam score: {probability:.1%})")


def show_model_info():
    model = load_model()
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["model"]
    print("Model file:", MODEL_PATH)
    print("Classifier:", type(classifier).__name__, "| C =", classifier.C)
    print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
    print("Decision threshold:", THRESHOLD)


def main():
    actions = {"1": train_and_evaluate, "2": check_email, "3": show_model_info}
    while True:
        print(MENU)
        choice = input("Choose 1-4: ").strip()
        if choice == "4":
            print("Goodbye.")
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid choice. Enter a number from 1 to 4.")
            continue
        try:
            action()
        except FileNotFoundError as error:
            print("Error:", error)
        except Exception as error:
            print("Something went wrong:", error)


if __name__ == "__main__":
    main()





