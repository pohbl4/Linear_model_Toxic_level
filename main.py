import os
import pickle
from src.predict import predict_toxicity

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Модель не найдена. Сначала запустите train.py для обучения и сохранения модели."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("Загружена сохранённая модель и векторизатор.")
    return model, vectorizer

def main():
    model, vectorizer = load_model_and_vectorizer()

    print("Введите комментарий для проверки токсичности 'ENG' (или 'exit' для выхода):")
    while True:
        user_input = input("Ваш комментарий: ").strip()
        if user_input.lower() == "exit":
            break
        prob = predict_toxicity(model, vectorizer, user_input)
        print(f"Toxicity score: {prob:.4f}")
        print("Toxic" if prob > 0.5 else "Not toxic")

if __name__ == "__main__":
    main()
