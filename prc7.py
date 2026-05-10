# Practical No. 07

import nltk
import re
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')

# -------------------------------
# Tokenization
# -------------------------------

text = "Tokenization is the first step in text analytics."

tokenized_text = sent_tokenize(text)
print("Sentence Tokenization:")
print(tokenized_text)

tokenized_word = word_tokenize(text)
print("\nWord Tokenization:")
print(tokenized_word)

# -------------------------------
# Stop Words Removal
# -------------------------------

stop_words = set(stopwords.words("english"))

text = "How to remove stop words with NLTK library in Python?"

text = re.sub('[^a-zA-Z]', ' ', text)

tokens = word_tokenize(text.lower())

filtered_text = []

for w in tokens:
    if w not in stop_words:
        filtered_text.append(w)

print("\nTokenized Sentence:")
print(tokens)

print("\nFiltered Sentence:")
print(filtered_text)

# -------------------------------
# Stemming
# -------------------------------

e_words = ["wait", "waiting", "waited", "waits"]

ps = PorterStemmer()

print("\nStemming:")

for w in e_words:
    rootWord = ps.stem(w)
    print(rootWord)

# -------------------------------
# Lemmatization
# -------------------------------

wordnet_lemmatizer = WordNetLemmatizer()

text = "studies studying cries cry"

tokenization = nltk.word_tokenize(text)

print("\nLemmatization:")

for w in tokenization:
    print("Lemma for {} is {}".format(
        w,
        wordnet_lemmatizer.lemmatize(w)
    ))

# -------------------------------
# POS Tagging
# -------------------------------

data = "The pink sweater fit her perfectly"

words = word_tokenize(data)

print("\nPOS Tagging:")

for word in words:
    print(nltk.pos_tag([word]))

# -------------------------------
# TF-IDF
# -------------------------------

d0 = 'Jupiter is the largest Planet'
d1 = 'Mars is the fourth planet from the sun'

string = [d0, d1]

tfidf = TfidfVectorizer()

result = tfidf.fit_transform(string)

print("\nWord Indices:")
print(tfidf.vocabulary_)

print("\nTF-IDF Values:")
print(result.toarray())
