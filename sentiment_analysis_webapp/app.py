from flask import Flask, render_template, request, redirect, url_for
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import re

app = Flask(__name__)

# Download NLTK data (run this only once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the pipeline with vectorizer, SMOTE, and model
loaded_pipeline_bow_lr_smote = joblib.load('models/pipeline_bow_lr_smote.pkl')

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation and token not in stop_words]

    # Join tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('index.html', prediction=None, text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_text = request.form['text']

        # Preprocess the text
        processed_text = preprocess_text(raw_text)

        # Make prediction using the loaded pipeline
        prediction = loaded_pipeline_bow_lr_smote.predict([processed_text])[0]

        return render_template('index.html', prediction=prediction, text=raw_text)
    
@app.route('/clear')
def clear():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
