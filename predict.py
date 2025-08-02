import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load the model and tokenizer
model_path = "./fake_news_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Define class names
class_names = ["FAKE", "REAL"]

# Define LIME prediction wrapper
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    return probs

# Generate a natural justification paragraph
def generate_justification(label, top_features):
    influential_words = [f"'{word}'" for word, weight in top_features[:4]]
    direction = "support" if label == "REAL" else "indicate falsehood in"
    justification = (
        f"The model predicted this news as **{label}**. "
        f"This decision was mainly influenced by the presence of words like {', '.join(influential_words)}. "
        f"These words {direction} the news article based on patterns learned from real-world data. "
        f"Such terms often appear in {'genuine' if label == 'REAL' else 'misleading'} news, which shaped the prediction."
    )
    return justification

# Predict with explanation and paragraph
def predict_news_with_explanation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    prediction_label = class_names[predicted_class]

    # Generate explanation
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text, predict_proba, num_features=6)
    top_features = explanation.as_list()

    # Get paragraph-style justification
    justification_paragraph = generate_justification(prediction_label, top_features)

    return prediction_label, top_features, justification_paragraph

# Run
if __name__ == "__main__":
    news_article = input("Enter a news article: ")
    prediction, features, paragraph = predict_news_with_explanation(news_article)

    print(f"\nPrediction: {prediction}")
    print("\nTop Contributing Words:")
    for word, weight in features:
        print(f"  {word}: {weight:.4f}")
    print("\nJustification:\n", paragraph)
