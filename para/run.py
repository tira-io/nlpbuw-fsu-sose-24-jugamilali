from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import torch
from torch.utils.data import DataLoader, Dataset

class ParaphraseDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        sentence1, sentence2 = self.texts[item]
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
        }

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")

    # Convert text DataFrame to list of tuples
    texts = df[['sentence1', 'sentence2']].values.tolist()

    # Set model path
    model_path = Path(__file__).parent

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    dataset = ParaphraseDataset(texts, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch['input_ids']
            b_attention_mask = batch['attention_mask']
            b_token_type_ids = batch['token_type_ids']
            
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_mask
            )
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())

    # Save the predictions
    df['label'] = predictions
    df = df.drop(columns=["sentence1", "sentence2"]).reset_index()
    
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )