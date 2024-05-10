import fasttext
from tira.rest_api_client import Client
from pathlib import Path
    
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

    with open("train_data.txt", "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            line = row['text'] + " __label__" + row['lang'] + "\n"
            f.write(line)

    model = fasttext.train_supervised(input="train_data.txt", epoch=10, lr=0.1)

    # Save the model
    model_path = Path(__file__).parent / "model.bin"
    model.save_model(str(model_path))

