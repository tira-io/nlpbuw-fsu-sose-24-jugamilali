from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

from transformers import pipeline, BartForConditionalGeneration
import torch

def summarize_texts(texts, summarizer, max_length=150):
    summaries = []
    for text in texts:
        summary = summarizer(text, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=5, early_stopping=True)
        summaries.append(summary[0]['summary_text'])
    return summaries

if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    # Determine device
    device = 0 if torch.cuda.is_available() else -1

    model = BartForConditionalGeneration.from_pretrained(str(Path(__file__).parent / "summarizer_model"))

    # Load the summarization pipeline
    summarizer = pipeline("summarization", model=model, device=device)

    # Summarize texts
    df['summary'] = summarize_texts(df['story'].tolist(), summarizer)

    # Prepare the DataFrame for output
    summaries = df[["id", "summary"]].reset_index()

    # Save the summaries
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_path = Path(output_directory) / "df.jsonl"
    summaries.to_json(output_path, orient="records", lines=True)