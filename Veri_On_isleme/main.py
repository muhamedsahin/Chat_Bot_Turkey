import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json

def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()
    
    # Noktalama işaretlerini kaldırma
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Rakamları kaldırma
    text = re.sub(r'\d+', '', text)
    
    # Tokenizasyon (kelimeleri ayırma)
    tokens = word_tokenize(text)
    
    # Stop kelimelerini kaldırma
    stop_words = set(stopwords.words('turkish'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Kök bulma (stemming)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Önişleme sonucunu birleştirme
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text


# kullanım
with open('./data/chat_islenmemis.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

preprocessed_data = []

for veri in data:
    kullanıcı = veri["kullanici_girisi"]
    bot = veri["bot_yaniti"]

    cıktı_kullanıcı = preprocess_text(kullanıcı)
    cıktı_bot = preprocess_text(bot)

    preprocessed_data.append({
        "kullanici_girisi":cıktı_kullanıcı,
        "bot_yaniti":cıktı_bot
    })

# cıktı json 
with open("./data/chat.json", "w", encoding="utf-8") as file:
    json.dump(preprocessed_data, file, ensure_ascii=False, indent=4)