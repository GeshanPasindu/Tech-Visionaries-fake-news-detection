import re
import emoji
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Ask user which dataset to load
print("\nWhich dataset do you want to load?")
print("1: Dataset with emojis (real_fake_dataset_with_emojies.json)")
print("2: Dataset without emojis (fake_or_real_news.csv)")
choice = input("Enter 1 or 2: ")

if choice == "1":
    file_path = "data/real_fake_dataset_with_emojies.json"
    is_emoji = True
elif choice == "2":
    file_path = "data/fake_or_real_news.csv"
    is_emoji = False
else:
    print("❌ Invalid choice.")
    exit()

# Load the dataset
try:
    if file_path.endswith(".json"):
        df = pd.read_json(file_path, orient='records')
    else:
        df = pd.read_csv(file_path)

    print(f"\n✅ Loaded {file_path} successfully. Shape: {df.shape}")
except Exception as e:
    print("❌ Failed to load dataset:", e)
    exit()

# Show raw examples
print("\n🔍 Sample rows from dataset:")
print(df[['text']].head())

# Emoji conversion
def convert_emoji_to_text(text):
    emoji_text = emoji.demojize(text, delimiters=(" [", "] "))
    emoji_text = emoji_text.replace('_', ' ')
    # Remove any leftover empty brackets
    emoji_text = re.sub(r'\[\s*\]', '', emoji_text)
    return emoji_text

# Preprocessing function
def preprocess_text(text, convert_emojis=False):
    text = str(text)

    if convert_emojis:
        text = convert_emoji_to_text(text)

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\[\]]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words or w.startswith('[')]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) if not w.startswith('[') else w for w in tokens]

    return ' '.join(tokens)

# Preprocess the text
print("\n⚙️ Preprocessing started...")
df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, convert_emojis=is_emoji))

# Normalize labels if available
if 'label' in df.columns:
    df['label'] = df['label'].astype(str).str.upper().map({'FAKE': 0, 'REAL': 1})

# Drop rows with missing cleaned text
df.dropna(subset=['cleaned_text'], inplace=True)

# Save to output CSV
df.to_csv("processed_fake_or_real_news.csv", index=False)

# Display output preview
print("\n✅ Final Sample Output Preview:")
preview_cols = ['text', 'cleaned_text']
if 'label' in df.columns:
    preview_cols.append('label')

if is_emoji:
    print(df[preview_cols].head(50).to_json(orient='records', indent=2, force_ascii=False))
else:
    print(df[preview_cols].head(200))
