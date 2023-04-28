import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import gdown
import zipfile
import os

# Download the model from Google Drive
# Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1TxaMckkdOZ64XWs6RUBsDi5NSuZhBN1L"
model_zip_path = "balaji.zip"
gdown.download(model_url, model_zip_path, quiet=False)

model_folder = "fine_tuned_model"
with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
    zip_ref.extractall(model_folder)


nltk.download('stopwords')


# Load your fine-tuned model for sentiment analysis
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('fine_tuned_model/IMDB_model_bert')
tokenizer_fine_tuned = AutoTokenizer.from_pretrained('fine_tuned_model/IMDB_model_bert')
sentiment_model_fine_tuned = pipeline('sentiment-analysis', model=fine_tuned_model, tokenizer=tokenizer_fine_tuned)

# Load pre-trained model for emotional word highlighting
highlight_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',model_max_length=512)

stop_words = set(stopwords.words('english'))

def get_emotional_words(text, sentiment_label, top_n=5):
    # Truncate the text to fit within the model's limit
    truncated_text = tokenizer.encode(text.lower(), max_length=512, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(truncated_text)

    emotional_words = []

    for token in tokens:
        if token not in stop_words and token.isalpha():
            token_sentiment_result = highlight_model(token)[0]
            token_sentiment_score = token_sentiment_result['score']
            token_sentiment_label = token_sentiment_result['label']

            if token_sentiment_label == sentiment_label:
                emotional_words.append((token, token_sentiment_score))

    # Sort emotional words by sentiment score and keep the top N
    emotional_words.sort(key=lambda x: x[1], reverse=True)
    top_emotional_words = emotional_words[:top_n]

    return top_emotional_words
# Streamlit app
image = Image.open('Zocket-Logo-1024x1024-transparent.png')
st.image(image)
st.title("Movie Reviewer")

input_text = st.text_area("Try Zocket Movie Reviewer:")
submit_button = st.button("Submit")

if input_text and submit_button:
    # Truncate the text to fit within the model's limit
    input_text_encoded = tokenizer_fine_tuned.encode(input_text, max_length=512, truncation=True)
    input_text_truncated = tokenizer_fine_tuned.decode(input_text_encoded,skip_special_tokens=True)

    # Check the overall sentiment using your fine-tuned model
    overall_sentiment = sentiment_model_fine_tuned(input_text_truncated)[0]['label']
    
    if overall_sentiment == "LABEL_1":
        overall_sentiment = "POSITIVE"
        top_emotional_words = get_emotional_words(input_text_truncated, "POSITIVE")
        color = "green"
        emoji = "😊"
    else:
        overall_sentiment="NEGATIVE"
        top_emotional_words = get_emotional_words(input_text_truncated, "NEGATIVE")
        color = "red"
        emoji = "☹️"

    

    # Highlight top emotional words with NER-style boxes
    highlighted_text = input_text_truncated
    for word, _ in top_emotional_words:
        highlighted_text = highlighted_text.replace(word, f"<mark style='background-color: {color}; padding: 2px; border-radius: 3px;'>{word}</mark>")


    st.write(f"Predicted Movie Review<span style='color:{color};'>{overall_sentiment} {emoji} </span>", unsafe_allow_html=True)
    st.write(highlighted_text, unsafe_allow_html=True)
