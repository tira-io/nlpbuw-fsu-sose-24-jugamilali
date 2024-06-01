from tira.rest_api_client import Client
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
    
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
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    df = text.join(labels.set_index("id"))

    texts, labels= df[['sentence1', 'sentence2']].values.tolist(), df['label'].values

    # Tokenizer laden
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = ParaphraseDataset(texts, labels, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Modell laden
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to('cuda')

    # Optimizer und Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(dataloader) * 3  # 3 Epochen
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in dataloader:
            b_input_ids = batch['input_ids'].to('cuda')
            b_attention_mask = batch['attention_mask'].to('cuda')
            b_token_type_ids = batch['token_type_ids'].to('cuda')
            b_labels = batch['label'].to('cuda')
            
            model.zero_grad()
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f"Epoch {epoch+1}, Loss: {total_train_loss / len(dataloader)}")

    # Speichere das Modell und den Tokenizer
    model.save_pretrained(".")
    tokenizer.save_pretrained(".")