import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the preprocessed file
csv_path = "processed_fake_or_real_news.csv" 
df = pd.read_csv(csv_path)

# Check for cleaned_text
if 'cleaned_text' not in df.columns:
    raise ValueError("Missing 'cleaned_text' column")

df['cleaned_text'].fillna("", inplace=True)

# Custom tokenizer to keep [emoji] as a full token
def custom_tokenizer(text):
    # Match [emoji], or regular words — but ignore empty [ ]
    return re.findall(r'\[(?!\s*\])[^]]+\]|\w+', text)

# TF-IDF with custom tokenizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    tokenizer=custom_tokenizer
)
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print(f"\n✅ TF-IDF matrix shape: {tfidf_df.shape}")
print(tfidf_df.head())

# Detect emojis
emoji_features = [f for f in tfidf_df.columns if f.startswith('[') and f.endswith(']')]
if emoji_features:
    print("\n🧠 Emoji-related features detected:")
    print(emoji_features)
else:
    print("\nℹ️ No emoji-related features detected. Proceeding with text-only features.")
