from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test_vect, y_test):
    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def show_top_toxic_words(vectorizer, model, top_n=10):
    vocab = vectorizer.vocabulary_
    coefficients = model.coef_[0]
    most_toxic_words = sorted(vocab, key=lambda x: coefficients[vocab[x]], reverse=True)
    return [(word, coefficients[vocab[word]]) for word in most_toxic_words[:top_n]]
