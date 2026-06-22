# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}

# Set up the special tokens in the reverse index for decoding
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = '<UNUSED>'

# Load the pre-trained model
model = load_model('imdb_lstm_rnn.h5')

# Step 2: Helper Functions
# Function to decode reviews back to text for verification
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Function to preprocess user input accurately
def preprocess_text(text):
    # Remove punctuation so words like "good!" or "bad." are recognized properly
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    
    # Keras IMDB dataset sequences always start with the <START> token (index 1)
    encoded_review = [1]
    
    for word in words:
        if word in word_index:
            index = word_index[word] + 3
            # Use index 2 (<UNK>) if the word index exceeds the model's vocabulary limit
            # (Assuming standard top 10,000 words; adjust if your model uses a different limit)
            if index < 10000:
                encoded_review.append(index)
            else:
                encoded_review.append(2)
        else:
            encoded_review.append(2) # Map unknown words to <UNK> token
            
    # Pad the sequence to match training length (500 tokens)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit UI
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter some text before classifying.")
    else:
        # Preprocess the text
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]
        
        # Determine sentiment classification
        sentiment = 'Positive' if score > 0.5 else 'Negative'

        # Display the result
        st.subheader(f'Sentiment: {sentiment}')
        st.write(f'**Prediction Confidence Score:** {score:.4f}')
        
else:
    st.write('Please enter a movie review.')
