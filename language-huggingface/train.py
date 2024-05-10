import fasttext
from huggingface_hub import hf_hub_download
from pathlib import Path
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tira.rest_api_client import Client

class FastTextTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, model_path):
        self.model_path = model_path

    def fit(self, X, y=None):
        return self
    
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

    # Download and save FastText model
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model_save_path = "fasttext_model.bin"
    model = fasttext.load_model(model_path)
    model.save_model(model_save_path)

    # Define the pipeline
    model_pipeline = Pipeline([
        ("fasttext", FastTextTransformer(model_save_path))
    ])

    # Train the model
    model_pipeline.fit(df["text"])

    # Save the model
    dump(model_pipeline, Path(__file__).parent / "model.joblib")

