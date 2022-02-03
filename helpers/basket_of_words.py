import nltk
import numpy

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


def basket_of_words(s, words):
    basket = [0 for _ in range(len(words))]

    set_of_words = nltk.word_tokenize(s)
    set_of_words = [stemmer.stem(w.lower()) for w in set_of_words]

    for set in set_of_words:
        for i, wrd in enumerate(words):
            if wrd == set:
                basket[i].append(1)

    return numpy.array(basket)
