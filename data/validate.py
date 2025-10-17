import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json

# 1. Load the test dataset
df = pd.read_csv('data/test.csv')

# 2. Select features and labels
X_test = df['description']
y_test = df['label']

# 3. Load the trained model and vectorizer
model, vectorizer = joblib.load('movie_model.pkl')

# 4. Transform test data (convert text → TF-IDF)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Make predictions
preds = model.predict(X_test_tfidf)

# 6. Calculate accuracy
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")

# 7. Save accuracy to metrics.json
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# 8. Generate and save confusion matrix plot
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("Validation complete! Metrics and confusion matrix saved.")
