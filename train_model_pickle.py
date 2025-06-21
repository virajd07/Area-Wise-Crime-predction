# 📓 train_model.ipynb (Python script version for GitHub)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

# ============================
# 🔹 1. Load and Prepare Historical Crime Data
# ============================
df = pd.read_csv("cleaned_crime_data.csv")

# Create 'Major_Crime' column if not exists
if "Major_Crime" not in df.columns:
    df["Major_Crime"] = df.iloc[:, 2:].idxmax(axis=1)

X = df.iloc[:, 2:-1]  # All features except Year/State and label
y = df["Major_Crime"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
with open("crime_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Crime Prediction Model and Label Encoder Saved.")

# ============================
# 🔹 2. Train News Classification Model (NLP)
# ============================
news_df = pd.read_csv("labeled_crime_news.csv")  # Columns: text, label

X_text = news_df["text"].fillna("")
y_label = news_df["label"]  # should be: Murder, Robbery, etc.

vec = TfidfVectorizer(max_features=3000, stop_words='english')
X_vec = vec.fit_transform(X_text)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, y_label)

joblib.dump(clf, "crime_news_model.pkl")
joblib.dump(vec, "crime_vectorizer.pkl")

print("✅ Crime News Classifier and Vectorizer Saved.")
