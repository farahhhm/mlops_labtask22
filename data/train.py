import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('data/train.csv')
X = df['description']
y = df['label']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # smaller matrix
X_tfidf = vectorizer.fit_transform(X)

# Train model (faster)
model = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
model.fit(X_tfidf, y)

# Save model
joblib.dump((model, vectorizer), 'movie_model.pkl')
print("Model trained and saved successfully")
