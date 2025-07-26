# (D:\Udemy\Complete_DSMLDLNLP_Bootcamp\Python\58-LSTM & GRU-TSTM End to End DL Project\venv) 
# D:\Udemy\Complete_DSMLDLNLP_Bootcamp\Python\58-LSTM & GRU-TSTM End to End DL Project>streamlit run 3-app.py

import streamlit as st # type:ignore
import numpy as np # type:ignore
import pickle
from tensorflow.keras.models import load_model # type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

# Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words") # input_text=st.text_input("Enter the sequence of Words","Barn. Last night of all,When yond same")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
