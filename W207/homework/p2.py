# This tells matplotlib not to try opening a new window for each plot.
# %matplotlib inline

# General libraries.
import re
import numpy as np
import matplotlib.pyplot as plt

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn library for importing the newsgroup data.
from sklearn.datasets import fetch_20newsgroups

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)

num_test = len(newsgroups_test.target)
test_data, test_labels = newsgroups_test.data[int(num_test/2):], newsgroups_test.target[int(num_test/2):]
dev_data, dev_labels = newsgroups_test.data[:int(num_test/2)], newsgroups_test.target[:int(num_test/2)]
train_data, train_labels = newsgroups_train.data, newsgroups_train.target

print('training label shape:', train_labels.shape)
print('test label shape:', test_labels.shape)
print('dev label shape:', dev_labels.shape)
print('labels names:', newsgroups_train.target_names)

# -------------------------------------------------------------------------------- Question 1 -----
"""
def head_training_data(num_examples = 5):
    '''
    Prints the first specified number of newsgroup training data labels and text.

    Arguments
    ---------
    num_examples: int
    
    Number of newsgroup training data labels and text to print (default: 5)
    '''

    assert isinstance(num_examples, int), "num_examples must be of type int"
    assert num_examples >= 0, "num_examples must be greater than or equal to 0"

    for i in range(num_examples):
        print("+++++ newsgroups_train[", i, "]", sep = "")
        print("Label:", newsgroups_train.target_names[newsgroups_train.target[i]])
        print("Text:\n-----\n", newsgroups_train.data[i], "\n-----\n")

head_training_data()
"""

# -------------------------------------------------------------------------------- Question 2 -----

# Learn the training vocabulary and create the training term matrix
train_vectorizer = CountVectorizer()
train_term_matrix = train_vectorizer.fit_transform(newsgroups_train.data)

# What is the size of the vocabulary?
print("Vocabulary size:", len(train_vectorizer.get_feature_names()), "features")

# What is the average number of non-zero features per example?
average_nonzero_features = train_term_matrix.data.sum() / len(newsgroups_train.data)
print("Average number of non-zero features per example: %.4f" % average_nonzero_features, "features")

# What fraction of the entries in the matrix are non-zero?
fraction_nonzero_entries = \
    train_term_matrix.nnz / (train_term_matrix.shape[0] * train_term_matrix.shape[1])
print("Fraction of entries in matrix that are non-zero: %.4f" % fraction_nonzero_entries)

q2c_vocabulary = ["atheism", "graphics", "space", "religion"]
q2c_vectorizer = CountVectorizer(vocabulary =  q2c_vocabulary)
q2c_term_matrix = q2c_vectorizer.fit_transform(newsgroups_train.data)
fraction_nonzero_entries = \
    q2c_term_matrix.nnz / (q2c_term_matrix.shape[0] * q2c_term_matrix.shape[1])
print(q2c_term_matrix.shape)
print("Fraction of entries in matrix that are non-zero: %.4f" % fraction_nonzero_entries)

# Initialize CountVectorizers with character ngram min and max lengths of (2, 2), (3, 3) and 
# (2, 3), and identify the size of the resulting vocabularies once trained on the training data
# for t in [(2, 2), (3, 3), (2, 3)]:
for t in [(2, 2), (3, 3), (2, 3)]:
    q2d_vectorizer = CountVectorizer(analyzer = "char", ngram_range = t)
    q2d_term_matrix = q2d_vectorizer.fit_transform(newsgroups_train.data)
    print("Character ngram limits:", t, "- Vocabulary size:", len(q2d_vectorizer.vocabulary_), "ngrams")

# Create and train a vectorizer to prune words that appear in fewer than 10 documents 
q2e_vectorizer = CountVectorizer(min_df = 10)
q2e_term_matrix = q2e_vectorizer.fit_transform(newsgroups_train.data)

# What size vocabulary does this yield?
print("Minimum document frequency: 10 words - Vocabulary size:", len(q2e_vectorizer.vocabulary_), "words")

q2f_train_vectorizer = CountVectorizer()
q2f_dev_vectorizer = CountVectorizer()

q2f_train_term_matrix = q2f_train_vectorizer.fit_transform(train_data)
q2f_dev_term_matrix = q2f_dev_vectorizer.fit_transform(dev_data)

missing_from_train_data = 0
for w in q2f_dev_vectorizer.vocabulary_:
    if w not in q2f_train_vectorizer.vocabulary_:
        missing_from_train_data += 1 
    else: 
        pass

fraction_missing_from_train_data = missing_from_train_data / len(q2f_dev_vectorizer.vocabulary_)
print("Fraction of words in dev data missing from training vocabulary: %.4f" % fraction_missing_from_train_data)
