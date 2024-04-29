from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    merged_train = pd.merge(text_train, targets_train, on='id')
    merged_validation = pd.merge(text_validation, targets_validation, on='id')

    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    x_train_tfidf = tfidf_vectorizer.fit_transform(merged_train['text']).toarray()
    x_validation_tfidf = tfidf_vectorizer.fit_transform(merged_validation['text']).toarray()

    y_train = merged_train['generated']
    y_validation = merged_validation['generated']

    model = RandomForestClassifier()
    model.fit(x_train_tfidf, y_train)
    prediction = model.predict(x_validation_tfidf)
        
    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
