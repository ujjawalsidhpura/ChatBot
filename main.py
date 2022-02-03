import json
import tensorflow
import random
import numpy
import nltk


from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('commands.json') as file:
    data = json.load(file)

words = []
labels = []
docs = []

# Loop through commands to filter out sentences and then use nltk to convert those sentences into array of words
for command in data['commands']:
    for pattern in command['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)  # Add this list to words array
        docs.append(pattern)  # Also add pattern to docs

    if command['tag'] not in labels:
        labels.append(command['tag'])
