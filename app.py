import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the LSTM model
model = load_model('Dense_Spam_Detection.h5')

# Define the function to preprocess the user's input message
def preprocess_message(message):
    # Convert the message to lowercase
    message = message.lower()
    # Tokenize the message
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([message])
    # Convert the message to a sequence of integers
    sequence = tokenizer.texts_to_sequences([message])
    # Pad the sequence with zeros so that it has the same length as the sequences used to train the model
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=50)
    return padded_sequence

# Define the Streamlit app
def app():
    st.title("Spam Detector")
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/2387793/pexels-photo-2387793.jpeg?cs=srgb&dl=pexels-adrien-olichon-2387793.jpg&fm=jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    # Ask the user to input a message
    message = st.text_input("Enter a message:")
    # Preprocess the message and make a prediction
    if message:
        processed_message = preprocess_message(message)
        prediction = model.predict(processed_message)
        # Display the prediction
        if prediction > 0.5:
            st.write("This message is spam with a probability of {:.2f}%.".format(prediction[0][0] * 100))
        else:
            st.write("This message is ham with a probability of {:.2f}%.".format((1-prediction[0][0]) * 100))
            
if __name__ == '__main__':
    app()
