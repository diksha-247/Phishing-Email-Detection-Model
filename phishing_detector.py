from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Verify your bank account immediately",
    "Claim your free prize now",
    "Meeting scheduled tomorrow",
    "Project report submitted"
]

labels = [
    "Phishing",
    "Phishing",
    "Safe",
    "Safe"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(emails)

model = MultinomialNB()

model.fit(X, labels)

new_email = ["Click here to verify your account"]

new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("Prediction:", prediction[0])
