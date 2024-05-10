from pathlib import Path

import fasttext
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def convert_labels_to_lang(predictions):
    lang_labels = {
        '__label__af': 'af',
        '__label__az': 'az',
        '__label__bg': 'bg',
        '__label__cs': 'cs',
        '__label__da': 'da',
        '__label__de': 'de',
        '__label__el': 'el',
        '__label__en': 'en',
        '__label__es': 'es',
        '__label__fi': 'fi',
        '__label__fr': 'fr',
        '__label__hr': 'hr',
        '__label__it': 'it',
        '__label__ko': 'ko',
        '__label__nl': 'nl',
        '__label__no': 'no',
        '__label__pl': 'pl',
        '__label__ru': 'ru',
        '__label__ur': 'ur',
        '__label__zh': 'zh'
    }
    converted_predictions = []
    for prediction in predictions:
        if prediction in lang_labels:
            converted_predictions.append(lang_labels[prediction])

    return converted_predictions

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    )

    # Load the model
    model_path = Path(__file__).parent / "model.bin"
    model = fasttext.load_model(str(model_path))
    
    predictions = [model.predict(text)[0][0] for text in df["text"]]
    preds = convert_labels_to_lang(predictions)
    df["lang"] = preds
    df = df[["id", "lang"]]


    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
