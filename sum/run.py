from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Function to generate summaries
def generate_summary(batch):
    inputs = tokenizer(batch['story'], max_length=1024, return_tensors='pt', truncation=True, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    summaries = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    batch['summary'] = tokenizer.batch_decode(summaries, skip_special_tokens=True)[0]
    return batch

if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = str(Path(__file__).parent / "sumbart")
    tokenizer_path = str(Path(__file__).parent / "sumbart")

    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

    # Move model to the appropriate device
    model.to(device)

    # Generate summaries
    df = df.apply(lambda row: generate_summary(row), axis=1, result_type='expand')

    # Prepare the DataFrame for output
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_file = Path(output_directory) / "predictions.jsonl"
    df.to_json(output_file, orient="records", lines=True)
