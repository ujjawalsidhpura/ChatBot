import json
import tensorflow
import random
import numpy
import nltk


from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('commands.json') as file:
    data = json.load(file)

print(data)
