import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Modeli yükleme
model = tf.keras.models.load_model('chat.h5')

# Tokenizer nesnesini yükleme
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Tahmin yapmak için kullanılan fonksiyon
def generate_text(input_text):
    # Metni dönüştürme ve diziye dönüştürme
    sequence = tokenizer.texts_to_sequences([input_text])
    sequence = pad_sequences(sequence, maxlen=10, padding='post')

    # Tahmin yapma
    predicted_sequence = model.predict(sequence)
    predicted_index = tf.argmax(predicted_sequence, axis=-1).numpy()[0]
    predicted_words = []
    for text in predicted_index:
        if text != 0:
            predicted_words.append(tokenizer.index_word[text])

    predicted_sentence = ' '.join(predicted_words)
    return predicted_sentence

input_text = "Merhaba"
response = generate_text(input_text)
print(response)