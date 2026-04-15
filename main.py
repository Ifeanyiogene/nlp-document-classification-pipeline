# NLP Text Classification Pipeline
# Simple ML project for text classification

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("🚀 NLP Text Classification Project")

# TODO: Replace this with your actual dataset
# Example placeholder:
# df = pd.read_csv('data/your_dataset.csv')

# Sample data (you can delete this later and load real data)
texts = ["I love this product", "This is terrible service", "Amazing experience", "Worst purchase ever"]
labels = ["positive", "negative", "positive", "negative"]

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\n✅ NLP pipeline ready! Add your real dataset next.")
