from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def train_crime_classifier():
    crimes = ["Murder", "Robbery", "Kidnapping", "Assault", "Cyber Crime"]
    examples = [
        "A man was stabbed and killed",
        "The bank was looted by armed men",
        "A child was abducted from home",
        "She was attacked by two men",
        "Online fraud and phishing attack"
    ]
    labels = [0, 1, 2, 3, 4]

    vec = CountVectorizer()
    X = vec.fit_transform(examples)

    model = MultinomialNB()
    model.fit(X, labels)

    joblib.dump(model, "crime_news_model.pkl")
    joblib.dump(vec, "crime_vectorizer.pkl")
    print("✅ NLP model and vectorizer saved!")

if __name__ == "__main__":
    train_crime_classifier()
