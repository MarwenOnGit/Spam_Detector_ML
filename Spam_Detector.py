import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.add("subject")


# --- Preprocessing function ---
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'\\[ntvfrb]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


# --- Load and combine datasets ---
try:
    df1 = pd.read_csv(r".\spam_ham_dataset.csv", encoding='latin-1')[['label', 'text']]
    df1['label'] = df1['label'].apply(lambda x: 1 if x == 'spam' else 0)

    df2 = pd.read_csv(r".\emails.csv", encoding='latin-1')[['spam', 'ttext']]
    df2.columns = ['label', 'text']

    df = pd.concat([df1, df2], ignore_index=True)
except FileNotFoundError as e:
    print(
        f"Error loading dataset: {e}. Make sure 'spam_ham_dataset.csv' and 'emails.csv' are in the correct directory.")
    exit()

# --- Count ham and spam labels ---
initial_label_distribution = df['label'].value_counts().to_dict()
print("Initial Label Distribution (0=Ham, 1=Spam):", initial_label_distribution)

# --- Clean dataset ---
df["processedT"] = df["text"].apply(preprocessing)
df["label"] = df["label"].apply(lambda x: 1 if x == 'spam' or x == 1 else 0)

# --- Vectorize ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processedT"])
Y = df["label"]

# --- Train-test split ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Train model ---
print("\nTraining the RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, Y_train)
print("Model training complete.")

# --- Evaluate model ---
print("\nEvaluating model performance:")
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))


# --- Prediction function ---
def predict_email_type(message: str) -> str:
    cleaned_message = preprocessing(message)
    if not cleaned_message:
        return "Cannot classify empty message after cleaning. Please enter some text."
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction_result = model.predict(vectorized_message)
    return "Spam" if prediction_result[0] == 1 else "Ham"


# --- CLI ---
if __name__ == "__main__":
    print("-" * 50)
    print("Welcome to the Email Spam Classifier!")
    print("Type 'exit' to quit at any prompt.")
    print(
        "To finish entering an email, type 'END_EMAIL' on a new line and press Enter, or just press Enter twice for a short email.")
    print("-" * 50)

    while True:
        print("\nEnter your email message:")
        print("(Type 'END_EMAIL' or press Enter twice to finish input for this email.)")

        lines = []
        empty_line_entered = False
        while True:
            line = input()
            if line.strip().lower() == 'exit':
                lines = []
                break
            if line.strip().lower() == 'end_email':
                break
            if not line.strip():
                if empty_line_entered:
                    break
                else:
                    empty_line_entered = True
            else:
                empty_line_entered = False

            lines.append(line)

        # If 'exit' was typed, break the outer loop
        if not lines and line.strip().lower() == 'exit':
            break

        user_message = "\n".join(lines).strip()

        if not user_message:
            print("No email content was entered for classification. Please try again.")
            continue

        prediction = predict_email_type(user_message)
        print(f"\n--- Prediction Result ---")
        print(f"Your email is classified as: **{prediction}**")
        print(f"-------------------------")

    print("\nThank you for using the Email Spam Classifier. Goodbye!")