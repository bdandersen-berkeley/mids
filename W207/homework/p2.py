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