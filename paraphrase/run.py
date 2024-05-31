from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import torch
from torch.utils.data import DataLoader, Dataset

class ParaphraseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"paraphrase-identification-validation-20240515-training"
    )

    texts, labels= df[['sentence1', 'sentence2']].values.tolist(), df['label'].values

    # Lade den Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(".")

    dataset = ParaphraseDataset(texts, labels, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Lade das Modell
    model = AutoModel.from_pretrained(".")
    model = model.to('cuda')

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch['input_ids'].to('cuda')
            b_attention_mask = batch['attention_mask'].to('cuda')
            b_token_type_ids = batch['token_type_ids'].to('cuda')
            
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_mask
            )
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )