import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load reviews dataset
df = pd.read_csv("20191226-reviews.csv")

print("Columns:", df.columns)

# Correct columns based on your dataset
review_col = "body"
rating_col = "rating"

# Remove empty reviews
df = df.dropna(subset=[review_col])

# Convert rating → sentiment
def get_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df[rating_col].apply(get_sentiment)

# Train model
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df[review_col])
y = df["sentiment"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")