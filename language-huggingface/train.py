import fasttext
from huggingface_hub import hf_hub_download
from pathlib import Path
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tira.rest_api_client import Client

class FastTextTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, model_path, label_mapping):
        self.model_path = model_path
        self.label_mapping = label_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = fasttext.load_model(self.model_path)
        predictions = [model.predict(text)[0][0] for text in X]
        mapped_predictions = [self.label_mapping[label] for label in predictions]
        return mapped_predictions

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    df = text.join(labels.set_index("id"))

    # Define language IDs and labels mapping
    lang_ids = [
        "af", "az", "bg", "cs", "da", "de", "el", "en", "es", "fi", "fr", "hr", "it", "ko", "nl", "no", "pl", "ru", "ur", "zh"
    ]
    lang_labels = [
        "__label__af", "__label__az", "__label__bg", "__label__cs", "__label__da", "__label__de", "__label__el",
        "__label__en", "__label__es", "__label__fi", "__label__fr", "__label__hr", "__label__it", "__label__ko",
        "__label__nl", "__label__no", "__label__pl", "__label__ru", "__label__ur", "__label__zh"
    ]
    label_mapping = dict(zip(lang_labels, lang_ids))

    # Download and save FastText model
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model_save_path = "fasttext_model.bin"
    model = fasttext.load_model(model_path)
    model.save_model(model_save_path)

    # Define the pipeline
    model_pipeline = Pipeline([
        ("fasttext", FastTextTransformer(model_save_path, label_mapping))
    ])

    # Train the model
    model_pipeline.fit(df["text"])

    # Save the model
    dump(model_pipeline, Path(__file__).parent / "model.joblib")

