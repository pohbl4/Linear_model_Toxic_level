from sklearn.feature_extraction.text import CountVectorizer

def get_vectorizer():
    return CountVectorizer()

def vectorize_text(vectorizer, train_texts, test_texts):
    X_train_vect = vectorizer.fit_transform(train_texts)
    X_test_vect = vectorizer.transform(test_texts)
    return X_train_vect, X_test_vect
