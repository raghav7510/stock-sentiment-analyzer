from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Load data with error handling
try:
    df = pd.read_csv("combined_financial_news.csv")
    print(f"✓ Loaded data: {len(df)} records")
except FileNotFoundError:
    print("❌ Error: combined_financial_news.csv not found. Run data_prepare.py first.")
    exit()

# Remove rows with missing values
df = df.dropna(subset=['Text', 'Sentiment'])
print(f"✓ After cleaning: {len(df)} records")

MODEL_NAME = "ProsusAI/finbert"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)

class FinancialDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]), return_tensors="pt",
            padding="max_length", truncation=True, max_length=256
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'labels': label
        }

label_map = {"Negative":0, "Neutral":1, "Positive":2}
df["Label"] = df["Sentiment"].map(label_map)

# Check for unmapped labels
if df["Label"].isna().any():
    print("⚠ Warning: Some sentiment labels could not be mapped")
    df = df.dropna(subset=['Label'])

print(f"✓ Label distribution:\n{df['Sentiment'].value_counts()}")

dataset = FinancialDataset(df["Text"].tolist(), df["Label"].tolist())
loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("\n" + "="*50)
print("Starting Training...")
print("="*50)

model.train()
total_loss = 0
total_batches = 0

for epoch in range(1):
    print(f"\nEpoch {epoch + 1}")
    for batch_idx, batch in enumerate(loader):
        try:
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / total_batches
                print(f"  Batch {batch_idx + 1}/{len(loader)} - Loss: {avg_loss:.4f}")
        
        except Exception as e:
            print(f"  ❌ Error processing batch {batch_idx}: {str(e)}")
            continue
    
    avg_epoch_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"\n✓ Epoch Complete - Average Loss: {avg_epoch_loss:.4f}")

print("\n" + "="*50)
print("✓ Training complete")
print("="*50)
model.save_pretrained("finbert_sentiment_model")
tokenizer.save_pretrained("finbert_sentiment_model")
print("✓ Model saved to: finbert_sentiment_model")
print("="*50)
