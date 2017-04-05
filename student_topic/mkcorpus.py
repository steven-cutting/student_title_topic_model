# coding: utf-8

# In[51]:

import csv

import itertools as itls


# In[75]:

import cytoolz as tlz
from cytoolz import curried as tlzc

from text2math import raw2text as r2t
from text2math import text2tokens as t2t

from gensim import corpora


# In[62]:

head = lambda l: list(tlzc.take(4, l))


# ---

# # Loading Data

# In[4]:

def read_student_title():
    with open("../student_title.csv") as f:
        return tlz.drop(1, [line for line in csv.reader(f)])


# ---

# # Initial Cleaning and Prepping

# In[58]:

def uni_and_bigram_cleanup(txts):
    return tlz.pipe(txts,
                    tlzc.map(r2t.decode_and_fix),
                    tlzc.map(lambda title: itls.chain(t2t.unigram(title), t2t.bigram(title))),
                    tlzc.map(tuple),
                    list)


# In[67]:

# CLEAN_TOKENS_SETS = uni_and_bigram_cleanup(STUDENT_TITLES)


# ---

# # Creating Dictionary and Bag of Words Corpus

# In[72]:

def make_dict_and_corpus(tokensets):

    dictionary = corpora.Dictionary(tokensets)

    dictionary.filter_extremes(no_below=5,
                               no_above=0.5,
                               keep_n=100000)
    dictionary.compactify()

    corpus = list(tlz.map(dictionary.doc2bow, tokensets))

    return (dictionary, corpus)
