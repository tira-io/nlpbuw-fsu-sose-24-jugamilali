from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
<<<<<<< HEAD
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
=======
        "nlpbuw-fsu-sose-24", "language-identification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240408-training"
>>>>>>> a2daf3776a3fbf72a6f0b53259bff5d19f1711af
    )
    df = text.join(labels.set_index("id"))

    # Train the model
    model = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())]
    )
    model.fit(df["text"], df["lang"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")