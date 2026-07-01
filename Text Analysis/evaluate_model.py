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

# Drop missing or empty text
df = df.dropna(subset=['cleaned_text'])
df['cleaned_text'] = df['cleaned_text'].astype(str)
df = df[df['cleaned_text'].str.strip() != ""]

# --- FIX: Clean label column ---
if 'label' not in df.columns:
    raise ValueError("Error: 'label' column is missing in the dataset.")

# Drop missing labels
df = df.dropna(subset=['label'])

# Map string labels to integers if needed
if df['label'].dtype == object:
    label_mapping = {"fake": 0, "real": 1}
    df['label'] = df['label'].map(label_mapping)

# Drop any remaining NaNs after mapping
df = df.dropna(subset=['label'])

# Convert label column to integers
df['label'] = df['label'].astype(int)

# Optional: Debug print to confirm label values
print("Unique label values after cleaning:", df['label'].unique())

# Tokenize the text
texts = df['cleaned_text'].tolist()
encodings = tokenizer.batch_encode_plus(
    texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)

# Dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings['input_ids'][idx],
            "attention_mask": self.encodings['attention_mask'][idx],
            "labels": torch.tensor(self.labels[idx])
        }

# Create dataset
eval_dataset = FakeNewsDataset(encodings, df['label'].tolist())

# Metric computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Evaluation config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:")
print(eval_results)
