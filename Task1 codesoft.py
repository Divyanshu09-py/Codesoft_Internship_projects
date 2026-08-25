import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------

data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

print(data.head())
print(data["label"].value_counts())


# ---------------------------------------------------
# 2. Convert labels
# ---------------------------------------------------

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# ---------------------------------------------------
# 3. Text Cleaning
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


data["message"] = data["message"].apply(clean_text)


# ---------------------------------------------------
# 4. Split Dataset
# ---------------------------------------------------

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# 5. TF-IDF Feature Extraction
# ---------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# ---------------------------------------------------
# 6. Train Models
# ---------------------------------------------------

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}

for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    prediction = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    results[name] = [accuracy, precision, recall, f1]

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        prediction,
        target_names=["Legitimate", "Spam"]
    ))


# ---------------------------------------------------
# 7. Compare Models
# ---------------------------------------------------

result_df = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nModel Comparison:")
print(result_df)


# ---------------------------------------------------
# 8. Select Best Model
# ---------------------------------------------------

best_model_name = result_df.loc["F1 Score"].idxmax()

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# ---------------------------------------------------
# 9. Test New SMS
# ---------------------------------------------------

def predict_sms(message):

    cleaned = clean_text(message)

    message_tfidf = tfidf.transform([cleaned])

    prediction = best_model.predict(message_tfidf)[0]

    if prediction == 1:
        return "SPAM"
    else:
        return "LEGITIMATE"


# ---------------------------------------------------
# 10. Test Examples
# ---------------------------------------------------

messages = [
    "Congratulations! You have won a $1000 cash prize. Call now!",
    "Hey, are you coming to college tomorrow?",
    "URGENT! You have won a free lottery ticket.",
    "Can you send me the notes from today's class?"
]

for message in messages:

    result = predict_sms(message)

    print("\nSMS:", message)
    print("Prediction:", result)