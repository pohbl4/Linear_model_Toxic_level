import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_data
from src.vectorizer import get_vectorizer, vectorize_text
from src.evaluation import evaluate_model, show_top_toxic_words

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def split_data(comments, target, test_size=0.3, random_state=42):
    return train_test_split(comments, target, test_size=test_size, random_state=random_state)

def train_and_save_model(data_path="data/data.csv"):
    comments, target = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(comments, target)

    vectorizer = get_vectorizer()
    X_train_vect, X_test_vect = vectorize_text(vectorizer, X_train, X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vect, y_train)

    accuracy = evaluate_model(model, X_test_vect, y_test)
    print(f"Accuracy on test set: {accuracy:.4f}")

    print("Top toxic words")
    for word, coeff in show_top_toxic_words(vectorizer, model):
        print(f"{word}: {coeff:.4f}")

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print("Модель и векторизатор сохранены")

    return model, vectorizer

if __name__ == "__main__":
    train_and_save_model()
