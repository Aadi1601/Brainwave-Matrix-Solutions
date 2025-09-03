# main.py

import pandas as pd

# Load fake and real datasets
fake_df = pd.read_csv("dataset/Fake.csv")
real_df = pd.read_csv("dataset/True.csv")

# Add labels
fake_df["label"] = 0   # FAKE = 0
real_df["label"] = 1   # REAL = 1

# Combine the datasets
df = pd.concat([fake_df, real_df], axis=0)

# Shuffle the combined dataset
df = df.sample(frac=1).reset_index(drop=True)

import string
from nltk.corpus import stopwords

# Download stopwords (if not already downloaded)
import nltk
#nltk.download('stopwords')

# Define a text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Rejoin the cleaned words
    return ' '.join(words)

# Apply cleaning only on the "text" column
df['text'] = df['text'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Define features (X) and labels (y)
X = df['text']     # Cleaned news text
y = df['label']    # 0 for FAKE, 1 for REAL

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = LogisticRegression()

from sklearn.feature_extraction.text import TfidfVectorizer

# Create the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_vec = vectorizer.transform(X_test)


# Train the model on training data
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Test the model on your own custom news input
def predict_news(news_text):
    # Clean the input text using the same function
    cleaned_text = clean_text(news_text)

    # Vectorize using the same TF-IDF vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vectorized_text)[0]
    label = "REAL" if prediction == 1 else "FAKE"

    print(f"\n📰 Prediction for Custom Input:\n\"{news_text[:100]}...\"")
    print(f"→ This news is predicted to be: **{label}**")

# Function to predict if news is real or fake
def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    label = "REAL" if prediction == 1 else "FAKE"

    print("\n📰 Prediction Result:")
    print(f"→ The news you entered is: **{label}**")

import re
import string
from nltk.corpus import stopwords

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# ---- Get Input from User ----
print("\nType a news article or headline below (or type 'exit' to quit):")

while True:
    user_input = input("🧾 News Text: ")
    if user_input.lower() == 'exit':
        print("👋 Exiting. Stay aware, not scared.")
        break
    else:
        cleaned_input = preprocess_text(user_input)  # Use your existing pre-processing function
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 0:
            print("🔴 Prediction: FAKE NEWS\n")
        else:
            print("🟢 Prediction: REAL NEWS\n")


    predict_news(user_input)



# Fit and transform on training data, transform on test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)




