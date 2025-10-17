import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load training data
df = pd.read_csv('data/train.csv')

# 2. Select features and labels (match your preprocess.py)
X = df['description']   # Movie descriptions (text)
y = df['label']         # Labels (genres/categories)

# 3. Convert text to numeric features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# 4. Train model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_tfidf, y)

# 5. Save trained model and vectorizer
joblib.dump((model, vectorizer), 'movie_model.pkl')

print("Model trained and saved successfully as 'movie_model.pkl'")
