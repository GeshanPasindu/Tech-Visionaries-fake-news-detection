import torch
from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from torch.utils.data import Dataset

# Load pre-trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fake_news_model")
tokenizer = DistilBertTokenizer.from_pretrained("./fake_news_model")

# Load dataset
df = pd.read_csv("sentiment_fake_or_real_news.csv")

# Remove NaN (missing) values
df = df.dropna(subset=['cleaned_text'])

# Ensure all values in 'cleaned_text' are strings
df['cleaned_text'] = df['cleaned_text'].astype(str)

# Check for empty strings and remove them
df = df[df['cleaned_text'].str.strip() != ""]

# Ensure 'label' column exists
if 'label' not in df.columns:
    raise ValueError("Error: 'label' column is missing in the dataset.")

# Convert categorical labels to integers if necessary (e.g., "fake" → 0, "real" → 1)
if df['label'].dtype == object:
    label_mapping = {"fake": 0, "real": 1}
    df['label'] = df['label'].map(label_mapping)

# Define the 'texts' variable as a list of 'cleaned_text' values
texts = df['cleaned_text'].tolist()

# Tokenize the text using batch_encode_plus
encodings = tokenizer.batch_encode_plus(
    texts, 
    truncation=True, 
    padding=True, 
    max_length=512, 
    return_tensors="pt"
)

# Convert to PyTorch tensors
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings['input_ids'][idx]),
            "attention_mask": torch.tensor(self.encodings['attention_mask'][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

# Create dataset
eval_dataset = FakeNewsDataset(encodings, df['label'].tolist())

# Function to compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Define training arguments
training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=8)

# Evaluate model
trainer = Trainer(model=model, args=training_args, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

eval_results = trainer.evaluate()
print(eval_results)
