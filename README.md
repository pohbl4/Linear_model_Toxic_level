# Toxic Comment Classifier

This project is a simple command-line toxicity classifier for user comments using traditional machine learning techniques. It is based on logistic regression — a **linear model** — trained on text data transformed using a `CountVectorizer`.

---

## Prerequisites

To run this project, you need the following tools installed on your machine:

- Python 3.8+
- Git

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/pohbl4/Linear_model_Toxic_level.git
   cd Linear_model_Toxic_level
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the training dataset in the `data/` folder as `data.csv`.

   The CSV file must contain at least:
   - `comment_text`: the comment content
   - `target`: toxicity score (float between 0 and 1)

4. Train the model and save it:
   ```bash
   python -m src.train
   ```

5. Run the interactive toxicity checker:
   ```bash
   python main.py
   ```

---

## How It Works

### Model

- The classifier is based on **logistic regression**, a linear model for binary classification.
- Text is transformed into feature vectors using `CountVectorizer` (bag-of-words).
- Comments are labeled as toxic if `target > 0.7`.

### Modules

- `src/data_loader.py`: Loads and prepares the dataset
- `src/vectorizer.py`: Creates and applies a CountVectorizer
- `src/train.py`: Trains and saves the model and vectorizer
- `src/evaluation.py`: Evaluates the model and lists top toxic words
- `src/predict.py`: Provides a prediction function for new comments
- `main.py`: Interactive command-line interface

---

## Saved Artifacts

Trained components are saved to the `models/` folder:

- `model.pkl`: Trained logistic regression model
- `vectorizer.pkl`: Fitted vectorizer

These are used in `main.py` without the need to retrain.

---

## Example

```
💬 Enter a comment to check for toxicity (or 'exit' to quit):
Your comment: You are disgusting.
🧪 Toxicity score: 0.8745
⚠️ Toxic
```
