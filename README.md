# Sentiment Analysis Web App

This repository contains a sentiment analysis project that uses a variety of techniques to classify text as positive, negative, or neutral. The goal is to provide a user-friendly web application where users can input text and instantly receive sentiment predictions.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Approaches](#approaches)
  - [LSTM (Long Short-Term Memory)](#lstm-long-short-term-memory)
  - [Traditional Method: Vader Sentiment Analysis](#traditional-method-vader-sentiment-analysis)
  - [Hugging Face RoBERTa Model](#hugging-face-roberta-model)
  - [Simple Method: Transformer Pipeline](#simple-method-transformer-pipeline)
- [Model Performance](#model-performance)
- [Web App](#web-app)
- [How to Run](#how-to-run)
- [License](#license)

## Overview

In this project, I explore multiple methods for sentiment analysis and implement them through a Streamlit web app. I have used **LSTM** (Long Short-Term Memory) networks for text classification, Hugging Face's **RoBERTa** model for transfer learning, and **NLTK's Vader** for a more traditional sentiment analysis approach.

## Technologies Used

- **LSTM** (Long Short-Term Memory) – Deep learning model for sequence prediction.
- **RoBERTa** – Pre-trained transformer-based model from Hugging Face.
- **VADER** (Valence Aware Dictionary and sEntiment Reasoner) – Traditional sentiment analysis model from the NLTK library.
- **Hugging Face Transformers** – Library to access pre-trained NLP models.
- **Streamlit** – Framework to quickly build interactive web applications.


## Approaches

### LSTM (Long Short-Term Memory)

LSTM is a type of recurrent neural network (RNN) that is particularly effective for sequential data, such as text. It learns long-term dependencies and is capable of capturing complex relationships between words in a sequence.

- **Why LSTM?**  
  LSTMs are great for handling sequences of text, as they are designed to remember information over time, which is crucial in tasks like sentiment analysis where context matters.
  
- **Performance:**  
  I achieved an accuracy of **97%** using an LSTM model. The model is able to effectively predict sentiment from input text.

### Traditional Method: Vader Sentiment Analysis

I have also implemented a traditional sentiment analysis approach using **VADER**, which is based on a lexicon and rules. VADER is very fast and works well for social media text or short sentences, making it a solid choice for basic sentiment tasks.

### Hugging Face RoBERTa Model

For more complex sentiment analysis, I used the pre-trained **RoBERTa** model from Hugging Face. RoBERTa is a robust transformer model known for its high performance in natural language understanding tasks.

- **Easy to Use:** Hugging Face’s `transformers` library provides an easy-to-use pipeline that allows you to run sentiment analysis with just **two lines of code**.
- **Performance:** RoBERTa outperforms traditional models and achieves superior accuracy on complex sentiment classification tasks.

### Simple Method: Transformer Pipeline

For those who prefer a simpler approach, I have included a **two-line code solution** using Hugging Face's transformer pipeline. This provides an easy and effective way to perform sentiment analysis with minimal code.

```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I love this product!")

##Model Performance

    LSTM Accuracy: 97%

    The LSTM model effectively handles complex sequences and delivers impressive results with text classification tasks.

    VADER Sentiment Analysis: A traditional approach that works well with shorter texts but might not be as accurate on more nuanced data.

    RoBERTa: The pre-trained RoBERTa model performs exceptionally well, and I used it for tasks that require deeper understanding and context. It delivers highly accurate predictions.
