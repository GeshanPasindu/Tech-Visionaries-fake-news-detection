import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
import emoji

# --- Load Model and Tokenizer ---
model_path = "./fake_news_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

class_names = ["FAKE", "REAL"]

# --- Load Emoji Sentiment Lexicon ---
emoji_sentiment = {}
try:
    emoji_df = pd.read_csv("data/EmoTag1200-scores-details.csv")
    for _, row in emoji_df.iterrows():
        try:
            emoji_char = str(row['emoji']).strip()  # <- Use emoji character, not name
            joy = float(row.get('joy', 0))
            trust = float(row.get('trust', 0))
            sadness = float(row.get('sadness', 0))
            anger = float(row.get('anger', 0))
            fear = float(row.get('fear', 0))
            disgust = float(row.get('disgust', 0))
            polarity_score = (joy + trust) - (sadness + anger + fear + disgust)
            emoji_sentiment[emoji_char] = polarity_score
        except:
            continue
except FileNotFoundError:
    print("Warning: Emoji sentiment lexicon not found.")
    emoji_sentiment = {}

# --- Utility Functions ---
def convert_emoji_to_text(text):
    if not isinstance(text, str):
        return text
    try:
        emoji_text = emoji.demojize(text, delimiters=(" [", "] "), language="en")
        emoji_text = emoji_text.replace('_', ' ')
        emoji_text = re.sub(r'\[\s*\]', '', emoji_text)
        return emoji_text
    except Exception:
        return text

def preprocess_text_for_prediction(text):
    text = str(text)
    text = convert_emoji_to_text(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\[\]]', '', text)
    return text

def extract_raw_emojis(text):
    return [char for char in text if emoji.is_emoji(char)]

def get_emoji_sentiment_from_raw_emojis(raw_emojis):
    if not raw_emojis:
        return 0.0, []

    scores = []
    emoji_names_with_scores = []
    for e_char in raw_emojis:
        score = emoji_sentiment.get(e_char, 0.0)
        emoji_names_with_scores.append((emoji.demojize(e_char).strip(':'), score))
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, emoji_names_with_scores

def get_text_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

def get_hybrid_sentiment(text_score, emoji_score, w_text=0.4, w_emoji=0.6):
    # If there are no meaningful emoji contributions, fallback to pure text
    if emoji_score == 0.0 or emoji_score is None:
        return text_score
    return (w_text * text_score) + (w_emoji * emoji_score)


# --- Prediction Function for LIME ---
def predict_proba(texts):
    processed_texts = [preprocess_text_for_prediction(t) for t in texts]
    inputs = tokenizer(processed_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

# --- Justification Generation ---
def generate_justification(label, top_features, raw_emojis_in_text, emoji_sentiment_info, text_sentiment, hybrid_sentiment):
    justification_parts = []
    justification_parts.append(f"The model predicted this news as **{label}**.")

    if top_features:
        words = [f"'{word}'" for word, weight in top_features if not (word.startswith('[') and word.endswith(']'))]
        if words:
            justification_parts.append(f"Key influential words: {', '.join(words)}.")

    if raw_emojis_in_text and emoji_sentiment_info:
        emoji_names = [e[0] for e in emoji_sentiment_info[1]]
        emoji_scores = [f"{e[0]} (score: {e[1]:.2f})" for e in emoji_sentiment_info[1]]
        avg_score = emoji_sentiment_info[0]
        sentiment_desc = "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral")

        emoji_sentence = (
            f"{len(raw_emojis_in_text)} emoji(s) found: {', '.join(raw_emojis_in_text)} "
            f"(interpreted as {', '.join(emoji_names)}), showing an overall {sentiment_desc} sentiment "
            f"with a score of {avg_score:.2f}."
        )
        justification_parts.append(emoji_sentence)

    justification_parts.append(
        f"Text Sentiment: {text_sentiment:.2f}, Hybrid Sentiment (40% Text + 60% Emoji): {hybrid_sentiment:.2f}."
    )

    if label == "FAKE" and hybrid_sentiment < 0:
        justification_parts.append("Negative emotional tone and sensational language often correlate with fake news.")
    elif label == "REAL" and hybrid_sentiment > 0:
        justification_parts.append("Positive sentiment and neutral tone often reflect real content.")

    justification_parts.append("These elements collectively guided the prediction decision.")
    return " ".join(justification_parts)

# --- Prediction Wrapper ---
def predict_news_with_explanation(text):
    raw_emojis_in_text = extract_raw_emojis(text)
    avg_emoji_sentiment, emoji_sentiment_details = get_emoji_sentiment_from_raw_emojis(raw_emojis_in_text)
    text_sentiment = get_text_sentiment(text)
    hybrid_sentiment = get_hybrid_sentiment(text_sentiment, avg_emoji_sentiment)

    processed_text_for_lime = preprocess_text_for_prediction(text)
    inputs = tokenizer(processed_text_for_lime, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    prediction_label = class_names[predicted_class]

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(processed_text_for_lime, predict_proba, num_features=8)
    top_features = explanation.as_list()

    justification_paragraph = generate_justification(
        prediction_label, top_features, raw_emojis_in_text,
        (avg_emoji_sentiment, emoji_sentiment_details),
        text_sentiment, hybrid_sentiment
    )

    return prediction_label, top_features, justification_paragraph, raw_emojis_in_text, text_sentiment, avg_emoji_sentiment, hybrid_sentiment

# --- Run Main ---
if __name__ == "__main__":
    print("\U0001F4E3 Welcome to the Fake News Predictor with Emoji Sentiment Integration!")
    print("\U0001F4DD Enter a social media news post (with or without emojis):")
    news_article = input(">>> ")

    prediction, features, paragraph, raw_emojis, text_sentiment, emoji_sentiment, hybrid_sentiment = predict_news_with_explanation(news_article)

    print(f"\n\U0001F52E Prediction: {prediction}")

    print("\n\U0001F4CA Top Contributing Words (LIME):")
    for word, weight in features:
        print(f"  '{word}': {weight:.4f}")

    print(f"\n\U0001F9E0 Text Sentiment: {text_sentiment:.4f}")
    print(f"🎮 Emoji Sentiment: {emoji_sentiment:.4f}")
    print(f"\u2696\uFE0F Hybrid Sentiment (0.4 Text + 0.6 Emoji): {hybrid_sentiment:.4f}")

    if raw_emojis:
        print(f"\n\U0001F50D Detected Emojis: {', '.join(raw_emojis)}")

    print("\n\U0001F4DD Justification:")
    print(paragraph)
