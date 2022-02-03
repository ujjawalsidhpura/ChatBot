import tensorflow
import json
import random
import numpy
import nltk
import tflearn

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('commands.json') as file:
    data = json.load(file)

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


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')
