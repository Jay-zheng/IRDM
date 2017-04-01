from nltk.corpus import stopwords
from math import log
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from nltk.stem.porter import *





# count how many words in list1 appears in list2
def counter_appearance(list1, list2):
    count = 0
    for i in list1:
        if i in list2:
            count += 1

    return count

#count how many times a word in list1 matches a word in list2
def counter_appear_times(list1, list2):
    count = 0
    for i in list1:
        for j in list2:
            if i == j:
                count += 1

    return count

#this calculates the term frequency,inverse document frequency,document length and average length for BM25 score
def tfidf(df, column):

    docs_tf = {}
    idf = {}
    doc_length = {}
    vocab = set()

    count = 0
    total_length = 0
    c = 0

    for index, row in df.iterrows():
        dd = {}
        total_words = 0

        words = str(row[column]).split() # split words

        for word in words: # select words in document
            vocab.add(word) # add to vacab
            dd.setdefault(word, 0)
            dd[word] += 1
            total_words += 1

        for k, v in dd.items():
            dd[k] = 1. * v / total_words #term frequency

        docs_tf[row['product_uid']] = dd # total number of documents in a collection
        doc_length[row['product_uid']] = total_words

        count += 1
        c += 1

        total_length += total_words

        if c % 1000 == 0:
            print('processing ', c, 'th documents(tf)',)

    co = 0

    for w in list(vocab):
        docs_with_w = 0
        for path, doc_tf in docs_tf.items():
            if w in doc_tf:
                docs_with_w += 1 # number of documents containing query
        idf[w] = log((len(docs_tf) - docs_with_w + 0.5)/(docs_with_w + 0.5)) # this tfidf is used for BM25 later

        co += 1
        if co % 1000 == 0:
            print('processing ', co, 'th word(idf)',)

    ave_length = total_length/count

    return docs_tf, idf, doc_length, ave_length


def BM25_score(df, docs_tf, idf, doc_length, ave_length):

    score = 0
    query = df['search_term']
    prod_uid = df['product_uid']
    words = query.split()
    for word in words:
        if word in docs_tf[prod_uid]:
            # chosen parameters for k = 1.5 and b = 0.75
            score += idf[word]*((docs_tf[prod_uid][word] * 2.5)/(docs_tf[prod_uid][word] + 1.5 *
                                                           (1 - 0.75 + 0.75 * doc_length[prod_uid]/ave_length)))

    return score


def tf(doc_tf, search_term):
    tf = 0
    for word in search_term:
        if word in doc_tf:
            tf += doc_tf[word]

    return tf


def idf(idf, search_term):
    idf_score = 0
    for word in search_term:
        if word in idf:
            idf_score += idf[word]

    return idf_score


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


