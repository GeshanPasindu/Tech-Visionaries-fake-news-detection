import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import transformers

print("Transformers version:", transformers.__version__)

# Read CSV safely with low_memory disabled
df = pd.read_csv("processed_fake_or_real_news.csv", low_memory=False)

# Ensure required columns exist
if 'cleaned_text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'cleaned_text' and 'label' columns.")

# Drop rows where 'label' or 'cleaned_text' is missing
df = df.dropna(subset=['label', 'cleaned_text'])

# Handle string labels if needed
if df['label'].dtype == object or isinstance(df['label'].iloc[0], str):
    label_mapping = {'fake': 0, 'real': 1}
    df['label'] = df['label'].map(label_mapping)

# Drop rows where mapping failed (i.e., labels were not 'fake' or 'real')
df = df.dropna(subset=['label'])

# Convert label to integer
df['label'] = df['label'].astype(int)

# Ensure text is string type
df['cleaned_text'] = df['cleaned_text'].astype(str)

# Optional: Print value counts for sanity check
print("\nLabel distribution:")
print(df['label'].value_counts())

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_dict({
    'cleaned_text': train_df['cleaned_text'].tolist(),
    'label': train_df['label'].tolist()
})
test_dataset = Dataset.from_dict({
    'cleaned_text': test_df['cleaned_text'].tolist(),
    'label': test_df['label'].tolist()
})

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("\nClass weights:", class_weights)

# Tokenization function
def tokenize(batch):
    return tokenizer(batch['cleaned_text'], padding=True, truncation=True, max_length=512)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Format datasets for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments
training_args = TrainingArguments(
    output_dir="./fake_news_model",
    evaluation_strategy="epoch",              
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    warmup_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# Metrics function
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

train_output = trainer.train()

print("\n✅ Model training completed successfully!")
print(f"Final training loss: {train_output.training_loss:.4f}")
print(f"Trained for {training_args.num_train_epochs} epochs")
print(f"Model saved at: {training_args.output_dir}")
