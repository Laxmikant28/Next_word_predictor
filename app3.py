import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

model = load_model('next_word_prediction.keras')
word_prediction = pickle.load(open('word_prediction.pickle','rb'))

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)> max_sequence_len:
        token_list = token_list[-(max_sequence_len):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len,padding='pre')
    prediction = model.predict(token_list,verbose=0)
    prediction_word_index = np.argmax(prediction,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == prediction_word_index:
            return word
    return None

st.title("Next Word Prediction")
input = st.text_input("Enter the sentence","to be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]
    next_word = predict_next_word(model,word_prediction,input.lower(),max_sequence_len)
    st.write(f"Next Word:{next_word}")