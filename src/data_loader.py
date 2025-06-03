import pandas as pd

def load_data(path: str):
    data = pd.read_csv(path)
    comments = data["comment_text"]
    target = (data["target"] > 0.7).astype(int)
    return comments, target
