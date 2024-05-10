from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    )

    # Load the model
    model = load(Path(__file__).parent / "model.joblib")

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

    predictions = model.predict(df["text"])
    mapped_predictions = [label_mapping[label] for label in predictions]
    df["lang"] = mapped_predictions
    df = df[["id", "lang"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
