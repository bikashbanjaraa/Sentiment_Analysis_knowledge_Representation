import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the saved model
model = tf.keras.models.load_model('sentiment_lstm_model.h5')

# Load the tokenizer
tokenizer = joblib.load('tokenizer.pkl')

# Define padding parameters
max_length = 100

# Streamlit app title
st.title("Sentiment Analysis with LSTM")

# Text input for user review
user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=max_length)
        
        # Predict the sentiment
        prediction = model.predict(input_padded)
        sentiment = np.argmax(prediction, axis=1)[0]  # 0 for Negative, 1 for Positive

        # Display result
        if sentiment == 1:
            st.success("The sentiment is **Positive**! 😊")
        else:
            st.error("The sentiment is **Negative**. 😟")
    else:
        st.warning("Please enter a review to analyze.")

