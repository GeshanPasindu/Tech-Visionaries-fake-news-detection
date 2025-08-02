import pandas as pd
import re
from textblob import TextBlob

# --- Load Preprocessed News Data ---
df = pd.read_csv("processed_fake_or_real_news.csv", dtype=str, low_memory=False)

# Check required columns
if 'label' not in df.columns:
    raise ValueError("Missing 'label' column.")
if 'cleaned_text' not in df.columns:
    raise ValueError("Missing 'cleaned_text' column.")

# --- Load Emoji Sentiment Lexicon ---
emoji_sentiment = {}
emoji_df = pd.read_csv("data/EmoTag1200-scores-details.csv")

# Calculate a polarity score: joy + trust - sadness - anger - fear - disgust
for _, row in emoji_df.iterrows():
    try:
        name = str(row['name']).lower().strip()
        joy = float(row.get('joy', 0))
        trust = float(row.get('trust', 0))
        sadness = float(row.get('sadness', 0))
        anger = float(row.get('anger', 0))
        fear = float(row.get('fear', 0))
        disgust = float(row.get('disgust', 0))

        polarity_score = (joy + trust) - (sadness + anger + fear + disgust)
        emoji_sentiment[name] = polarity_score
    except:
        continue  

# --- Sentiment Functions ---

# Text sentiment from TextBlob
def get_text_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Extract [emoji] tokens from text
def extract_emojis(text):
    return re.findall(r'\[([^\[\]]+)\]', str(text))

# Emoji sentiment using extracted emoji tags
def get_emoji_sentiment(text):
    emojis = extract_emojis(text)
    if not emojis:
        return 0.0

    scores = []
    for e in emojis:
        name = e.lower().strip()
        score = emoji_sentiment.get(name)
        if score is not None:
            scores.append(score)
        else:
            scores.append(0.0)  

    return sum(scores) / len(scores) if scores else 0.0

# Hybrid sentiment: fallback to text if emoji is 0.0
def get_hybrid_sentiment(row):
    text_score = row['text_sentiment']
    emoji_score = row['emoji_sentiment']
    return text_score if emoji_score == 0.0 else (0.4 * text_score) + (0.6 * emoji_score)

# --- Apply Sentiment Analysis ---
print("\n⚙️ Calculating text and emoji sentiment...")
df['text_sentiment'] = df['cleaned_text'].apply(get_text_sentiment)
df['emoji_sentiment'] = df['cleaned_text'].apply(get_emoji_sentiment)
df['hybrid_sentiment'] = df.apply(get_hybrid_sentiment, axis=1)

# Save results
output_path = "sentiment_fake_or_real_news.csv"
df.to_csv(output_path, index=False)

# Preview
print("\n✅ Sentiment analysis complete. Sample output:")
print(df[['cleaned_text', 'text_sentiment', 'emoji_sentiment', 'hybrid_sentiment', 'label']].head(125))
