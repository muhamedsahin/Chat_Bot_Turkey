import json
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Veri setini yükleme
with open('./data/chat.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

kullanici = np.array([], dtype='<U')
bot = np.array([], dtype='<U')

for veri in data:
    kullanici = np.append(kullanici, [veri["kullanici_girisi"]])
    bot = np.append(bot, [veri["bot_yaniti"]])

# Tokenizer oluşturma ve eğitme
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(kullanici) + list(bot))

# Kelime dağarcığını alma
word_index = tokenizer.word_index

#text to number
kullanici_sequences = tokenizer.texts_to_sequences(kullanici)
bot_sequences = tokenizer.texts_to_sequences(bot)

# Ayırma işlemi
X = kullanici_sequences#[:-1]
y = bot_sequences#[1:]

# Giriş dizilerini doldurma
max_sequence_length = 10  # Belirli bir maksimum dizgi uzunluğu seçin
X = pad_sequences(X, maxlen=max_sequence_length, padding='post')
y = pad_sequences(y, maxlen=max_sequence_length, padding='post')

# Modeli oluşturma
vocab_size = len(word_index) + 1
embedding_dim = 50
hidden_units = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(hidden_units, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

# Modeli derleme
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
epochs = 3000
batch_size = 32
model.fit(X, y, epochs=epochs, batch_size=batch_size)
model.save("chat.h5")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Modeli değerlendirme
test_loss, test_accuracy = model.evaluate(X, y, verbose=2)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)