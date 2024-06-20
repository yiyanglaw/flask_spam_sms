from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import mysql.connector
from datetime import datetime

app = Flask(__name__)

# MySQL database connection
db = mysql.connector.connect(
    host="sql12.freesqldatabase.com",
    user="sql12714674",
    password="15cCYtDhUC",
    database="sql12714674"
)
cursor = db.cursor()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the data for training
data = pd.read_csv('sms_s.csv')

# Fill missing values with an empty string
data['Message'] = data['Message'].fillna('')

data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# Advanced text preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words less than 3 characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['Message'] = data['Message'].apply(preprocess_text)

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Hyperparameter tuning using GridSearchCV
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__max_df': [0.75, 0.85, 1.0],
    'vectorizer__min_df': [1, 5, 10],
    'nb__alpha': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the results
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

@app.route('/sms/predict_spam', methods=['POST'])
def predict_spam_sms():
    text = request.form['text']
    processed_text = preprocess_text(text)
    prediction = best_model.predict([processed_text])[0]
    if prediction == 0:
        result = 'Ham (Not Spam)'
    else:
        result = 'Spam'
    
    # Store prediction in database
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = "INSERT INTO sms_spam_predictions (text, result, created_at) VALUES (%s, %s, %s)"
    values = (text[:255], result, current_time)  # Limit text to 255 characters
    cursor.execute(query, values)
    db.commit()
    
    return result

@app.route('/sms/history', methods=['GET'])
def get_sms_spam_history():
    query = "SELECT text, result, created_at FROM sms_spam_predictions ORDER BY created_at DESC LIMIT 50"
    cursor.execute(query)
    predictions = cursor.fetchall()
    predictions_list = []
    for prediction in predictions:
        prediction_data = {
            'text': prediction[0],
            'result': prediction[1],
            'created_at': prediction[2].strftime("%Y-%m-%d %H:%M:%S")
        }
        predictions_list.append(prediction_data)
    return jsonify(predictions_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10002, debug=False)
