from helpers import basket_of_words
from model import model_maker
import json
import random
import numpy
import nltk

import pickle

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('commands.json') as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as f:
        words, training, labels, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_a = []
    docs_b = []

    # Loop through commands to filter out sentences and then use nltk to convert those sentences into array of words
    for command in data['commands']:
        for pattern in command['patterns']:
            words_list = nltk.word_tokenize(pattern)
            words.extend(words_list)  # Add this list to words array
            docs_a.append(pattern)  # Also add pattern to docs
            # Corresponding tag to a pattern in docs_a
            docs_b.append(command['tag'])

        if command['tag'] not in labels:
            labels.append(command['tag'])

    # Stem it, also filter out '?' from data model
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    # Remove duplicates from stemmed set, convert back to list and sort it
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # Creating lists to train the function

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for a, doc in enumerate(docs_a):
        basket = []

        words_list = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in words_list:
                basket.append(1)
            else:
                basket.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_b[a])] = 1

        training.append(basket)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, training, labels, output), f)

model_maker(training, output)


def chat():
    print('Start chatting with the bot. Type "quit" to exit')

    while True:
        i = input('You: ')
        if i.lower() == 'quit':
            break

    # This gives probabality of each tag
    results = model.predict([basket_of_words(i, words)])
    # This gives the index of that tag with max probability
    results_indexof_highest = numpy.argmax(results)

    tag = labels[results_indexof_highest]

    # Response
    # Now we find the json with that specific tag,and find random response from that list

    for each_tag in data['commands']:
        if each_tag['tag'] == tag:
            responses = each_tag['responses']
            # This gives a list of responsed from the JSON

    print(random.choice(responses))


chat()
