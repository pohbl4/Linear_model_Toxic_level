def predict_toxicity(model, vectorizer, comment: str):
    comment_vect = vectorizer.transform([comment])
    prob = model.predict_proba(comment_vect)[0][1]
    return prob