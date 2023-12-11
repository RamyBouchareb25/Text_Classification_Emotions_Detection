# Import NumPy for numerical operations
import numpy as np
# Import pandas for data manipulation and analysis
import pandas as pd

# Import TensorFlow library for machine learning tasks
import tensorflow as tf
# Import Tokenizer and pad_sequences from Keras for text preprocessing
from tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Import EarlyStopping callback from Keras for early stopping during model training
from tensorflow.keras.callbacks import EarlyStopping

# Import Matplotlib for creating visualizations
import matplotlib.pyplot as plt


# Load training, validation, and test datasets from CSV files
train_df = pd.read_csv("training.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

# Extract tweets and labels from the training dataset
train_tweets = train_df['text']
train_labels = train_df['label']

# Extract tweets and labels from the validation dataset
val_tweets = val_df['text']
val_labels = val_df['label']

# Extract tweets and labels from the test dataset
test_tweets = test_df['text']
test_labels = test_df['label']


# Mapping of numerical indices to corresponding emotion classes
index_to_class = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}


# Create a Tokenizer with a vocabulary size of 10,000 and an out-of-vocabulary token
tokenizer = Tokenizer(num_words=10000, oov_token='<UKN>')

# Fit the Tokenizer on the training tweets to build the vocabulary
tokenizer.fit_on_texts(train_tweets)

# Function to convert text data into padded sequences using a tokenizer
def get_sequences(tokenizer, tweets):
    # Convert text to sequences of integers using the tokenizer
    sequences = tokenizer.texts_to_sequences(tweets)

    # Pad sequences to a maximum length of 50, truncating or padding as needed
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=50)

    # Return the padded sequences
    return padded


# Obtain padded sequences for the training tweets using the tokenizer
padded_train = get_sequences(tokenizer, train_tweets)

# Obtain padded sequences for the validation tweets using the tokenizer
padded_val = get_sequences(tokenizer, val_tweets)

# Obtain padded sequences for the test tweets using the tokenizer
padded_test = get_sequences(tokenizer, test_tweets)