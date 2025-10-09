# Importing Dependencies
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Model
model = load_model('data/gru_rnn.h5')

# Load Tokenizer
with open('data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len - 1):]
    token_list = pad_sequences([token_list], maxlen = max_seq_len - 1, padding = 'pre')
    predict = model.predict(token_list, verbose = 0)
    predicted_next_index = np.argmax(predict, axis = 1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_next_index:
            return word
    return None

# Streamlit
st.title('NEXT WORD PREDICTION USING GRU RNN')
text = st.text_input('Enter the sequence of words : ', 'To be or not to be')
if st.button('Predict'):
    max_seq_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, text, max_seq_len)
    st.write('Next Word:', next_word)
