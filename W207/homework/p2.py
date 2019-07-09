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
"""
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
"""
# -------------------------------------------------------------------------------- Question 3 -----
"""
def compute_f1(n, train_X, train_y, test_X, test_y):

    # assertions

    print("n =", n)

    knn_classifier = KNeighborsClassifier(n_neighbors = n)
    knn_classifier.fit(train_X, train_y)
    predicted_y = knn_classifier.predict(test_X)

    return metrics.f1_score(test_y, predicted_y, average = "micro")

def identify_optimal_k(train_X, train_y, test_X, test_y, base_k = 5):

    optimal_f1 = 0
    optimal_k = 0
    current_k = base_k
    delta = base_k

    while abs(optimal_k - current_k) != 1:

        current_f1 = compute_f1(current_k, train_X, train_y, test_X, test_y)
        if (0 == optimal_f1):
            optimal_f1 = current_f1
        print("f1 =", current_f1, "\n")

        if (optimal_f1 <= current_f1):

            delta = current_k
            optimal_k = current_k
            current_k += delta

        else:

            optimal_f1 = current_f1
            delta = np.floor(delta / 2)
            current_k -= delta

    return optimal_k

vanilla_vectorizer = CountVectorizer()
train_X = vanilla_vectorizer.fit_transform(train_data)
test_X = vanilla_vectorizer.transform(test_data)

knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(train_X, train_labels)

predicted = knn_classifier.predict(test_X)
print(metrics.f1_score(test_labels, predicted, average = "micro"))
print(metrics.classification_report(test_labels, predicted, target_names = categories))

optimal_k = identify_optimal_k(train_X, train_labels, test_X, test_labels, 5)

k_f1_matrix = list()
vectorizer = CountVectorizer()
train_X = vectorizer.fit_transform(train_data)
test_X = vectorizer.transform(test_data)

for k in range(5, int(len(test_data) / 3), 5):
    
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(train_X, train_labels)
    predicted_labels = knn_classifier.predict(test_X)
    k_f1_matrix.append((k, metrics.f1_score(test_labels, predicted_labels, average = "micro")))

plt.hist(k_f1_matrix[0][0])
plt.show()


def pp1(s):
  return s


def pp2(s):
  return re.sub(r"\W(\w{1,3})\W", " ", \
    re.sub(r"\W(\w{13,})\W", " ", \
      re.sub(r"\d+", " ", \
        re.sub(r"\W(a|able|about|across|after|all|almost|also|am|among|an|and|any|are|as|at|be|because|been|but|by|can|cannot|could|dear|did|do|does|either|else|ever|every|for|from|get|got|had|has|have|he|her|hers|him|his|how|however|i|if|in|into|is|it|its|just|least|let|like|likely|may|me|might|most|must|my|neither|no|nor|not|of|off|often|on|only|or|other|our|own|rather|said|say|says|she|should|since|so|some|than|that|the|their|them|then|there|these|they|this|tis|to|too|twas|us|wants|was|we|were|what|when|where|which|while|who|whom|why|will|with|would|yet|you|your)\W", " ", \
          re.sub(r"[\.\'\-_\?]", "", s.lower())
        )
      )
    )
  )
  # s = re.sub(r"\W+(a|able|about|across|after|all|almost|also|am|among|an|and|any|are|as|at|be|because|been|but|by|can|cannot|could|dear|did|do|does|either|else|ever|every|for|from|get|got|had|has|have|he|her|hers|him|his|how|however|i|if|in|into|is|it|its|just|least|let|like|likely|may|me|might|most|must|my|neither|no|nor|not|of|off|often|on|only|or|other|our|own|rather|said|say|says|she|should|since|so|some|than|that|the|their|them|then|there|these|they|this|tis|to|too|twas|us|wants|was|we|were|what|when|where|which|while|who|whom|why|will|with|would|yet|you|your)\W", " ", s.lower())
  # s = re.sub(r"\W+(\w{1,2})\W+", " ", s)
  # return s

# Create and train a vanilla (i.e. default) count vectorizer for training data.  Transform test
# data to a sparse matrix using the same vectorizer.

# q3_vectorizer = CountVectorizer()
# q3_vectorizer = CountVectorizer(stop_words = "english")
q3_vectorizer = CountVectorizer(preprocessor = pp2)
q3_train_term_matrix = q3_vectorizer.fit_transform(train_data)
q3_test_term_matrix = q3_vectorizer.transform(test_data)
print("Vocabulary length: %d" % len(q3_vectorizer.vocabulary_))

# Calculate the f1 score for a vanilla logistic regression classifier, where C = 1.0.  Note that
# defaults for parameters 'solver' and 'nulti-class' are changing in scikit-learn, so they must be
# explicitly specified (using the current default values) to silence the warning messages.
q3_lr_classifier = LogisticRegression(solver = "liblinear", multi_class = "ovr")
q3_lr_classifier.fit(X = q3_train_term_matrix, y = train_labels)

predicted_labels = q3_lr_classifier.predict(X = q3_test_term_matrix)
print("Logistic regression classifier f1 score (C = 1.0): %.6f" % \
      metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro"))
print("Classification report:\n", \
     metrics.classification_report(y_true = test_labels, y_pred = predicted_labels, target_names = categories))

TEST_C_VALUES = [0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99, 1.01, 1.1, 1.5, 1.9, 5]

for c in TEST_C_VALUES:
  test_lr_classifier = LogisticRegression(C = c, solver = "liblinear", multi_class = "ovr")
  test_lr_classifier.fit(X = q3_train_term_matrix, y = train_labels)
  predicted_labels = test_lr_classifier.predict(X = q3_test_term_matrix)
  print("Logistic regression classifier f1 score (C = %.4f): %.6f" % \
      (c, metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro")))

c_f1 = []
rough_optimal_c = 0.5 - 0.085
rough_maximum_f1 = 0.0

c = rough_optimal_c
while c <= 0.5 + 0.085:
  test_lr_classifier = LogisticRegression(C = c, solver = "liblinear", multi_class = "ovr")
  test_lr_classifier.fit(X = q3_train_term_matrix, y = train_labels)
  predicted_labels = test_lr_classifier.predict(X = q3_test_term_matrix)
  f1 = metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro")
  
  # Append the C-f1 tuple to the list, and check whether a maximum f1 value has been identified
  c_f1.append((c, f1))
  if f1 > rough_maximum_f1:
    rough_maximum_f1 = f1
    rough_optimal_c = c
    
  c += 0.005

plt.plot([x[0] for x in c_f1], [y[1] for y in c_f1])
plt.xlabel("C (inverse regularization strength)")
plt.ylabel("f1 score")
plt.title("f1 Scores for Logistic Regression Classifiers")
plt.ylim(0.753, 0.755)
plt.grid(True)
# plt.show()

print("Rough optimal C value: %.4f, logistic regression classifier's f1 value: %.6f" % (rough_optimal_c, rough_maximum_f1))

optimal_c = rough_optimal_c
maximum_f1 = 0.0

# Iterate through a range of C values to either side of the previously-identified (i.e. rough)
# optimal C
c = rough_optimal_c - ((rough_optimal_c / 100) * 20)
while c <= rough_optimal_c + ((rough_optimal_c / 100) * 20):
  test_lr_classifier = LogisticRegression(C = c, solver = "liblinear", multi_class = "ovr")
  test_lr_classifier.fit(X = q3_train_term_matrix, y = train_labels)
  predicted_labels = test_lr_classifier.predict(X = q3_test_term_matrix)
  f1 = metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro")
  
  # Check whether a maximum f1 value has been identified
  if f1 > maximum_f1:
    maximum_f1 = f1
    optimal_c = c
    
  c += rough_optimal_c / 100

print("Optimal C value: %.4f, logistic regression's f1 value: %.6f" % (optimal_c, maximum_f1))

# -------------------------------------------------------------------------------- Question 4 -----

LARGEST_WEIGHTS_COUNT = 5

def plot_largest_weights(vectorizer):
    '''
    '''
    q4_train_term_matrix = vectorizer.fit_transform(train_data)

    q4_lr_classifier = LogisticRegression(solver = "liblinear", multi_class = "ovr")
    q4_lr_classifier.fit(X = q4_train_term_matrix, y = train_labels)

    label_index_tuples = [()] * (len(q4_lr_classifier.classes_) * LARGEST_WEIGHTS_COUNT)
    for class_idx in q4_lr_classifier.classes_:
  
        assert len(vectorizer.vocabulary_) >= LARGEST_WEIGHTS_COUNT, "Classifier has an insufficient number of labels"
        sorted_weights = sorted(q4_lr_classifier.coef_[class_idx], reverse = True)
        assert len(set(sorted_weights[0:LARGEST_WEIGHTS_COUNT])) == LARGEST_WEIGHTS_COUNT, "Largest weights assigned to features are not unique"
  
        label_indicies = []
        for i in range(LARGEST_WEIGHTS_COUNT):
            label_indicies.append(list(q4_lr_classifier.coef_[class_idx]).index(sorted_weights[i]))

        labels = [None] * LARGEST_WEIGHTS_COUNT
        for label in vectorizer.vocabulary_.keys():
            if vectorizer.vocabulary_[label] in label_indicies:
                offset = label_indicies.index(vectorizer.vocabulary_[label])
                offset_idx = offset + (class_idx * LARGEST_WEIGHTS_COUNT)

                label_index_tuples[offset_idx] = (label, vectorizer.vocabulary_[label])

    label_data = []
    for label_index_tuple in label_index_tuples:
        label_weights_by_class = []
        for class_idx in range(len(q4_lr_classifier.classes_)):
            label_weights_by_class.append(str(q4_lr_classifier.coef_[class_idx][label_index_tuple[1]]))
        label_data.append(label_weights_by_class)

    columns = newsgroups_train.target_names
    rows = []
    for row in label_index_tuples:
        rows.append(row[0])
    print(rows)
    print(columns)
    plt.table(cellText = label_data, colLabels = columns,  loc = "center")
    plt.axis("off")
    plt.show()

plot_largest_weights(CountVectorizer())
plot_largest_weights(CountVectorizer(analyzer = "word", ngram_range = (2, 2)))

# -------------------------------------------------------------------------------- Question 5 -----

def pp1(s):
  return s

def pp2(s):
  return re.sub(r"\W(\w{1,3})\W", " ", \
    re.sub(r"\W(\w{13,})\W", " ", \
      re.sub(r"\d+", " ", \
        re.sub(r"\W(a|able|about|across|after|all|almost|also|am|among|an|and|any|are|as|at|be|because|been|but|by|can|cannot|could|dear|did|do|does|either|else|ever|every|for|from|get|got|had|has|have|he|her|hers|him|his|how|however|i|if|in|into|is|it|its|just|least|let|like|likely|may|me|might|most|must|my|neither|no|nor|not|of|off|often|on|only|or|other|our|own|rather|said|say|says|she|should|since|so|some|than|that|the|their|them|then|there|these|they|this|tis|to|too|twas|us|wants|was|we|were|what|when|where|which|while|who|whom|why|will|with|would|yet|you|your)\W", " ", \
          re.sub(r"[\.\'\-_\?]", "", s.lower())
        )
      )
    )
  )
  # s = re.sub(r"\W+(a|able|about|across|after|all|almost|also|am|among|an|and|any|are|as|at|be|because|been|but|by|can|cannot|could|dear|did|do|does|either|else|ever|every|for|from|get|got|had|has|have|he|her|hers|him|his|how|however|i|if|in|into|is|it|its|just|least|let|like|likely|may|me|might|most|must|my|neither|no|nor|not|of|off|often|on|only|or|other|our|own|rather|said|say|says|she|should|since|so|some|than|that|the|their|them|then|there|these|they|this|tis|to|too|twas|us|wants|was|we|were|what|when|where|which|while|who|whom|why|will|with|would|yet|you|your)\W", " ", s.lower())
  # s = re.sub(r"\W+(\w{1,2})\W+", " ", s)
  # return s

# Create and train a vanilla (i.e. default) count vectorizer for training data.  Transform test
# data to a sparse matrix using the same vectorizer.

# q3_vectorizer = CountVectorizer()
# q3_vectorizer = CountVectorizer(stop_words = "english")
q3_vectorizer = CountVectorizer(preprocessor = pp2)
q3_train_term_matrix = q3_vectorizer.fit_transform(train_data)
q3_test_term_matrix = q3_vectorizer.transform(test_data)
print("Vocabulary length: %d" % len(q3_vectorizer.vocabulary_))

# Calculate the f1 score for a vanilla logistic regression classifier, where C = 1.0.  Note that
# defaults for parameters 'solver' and 'nulti-class' are changing in scikit-learn, so they must be
# explicitly specified (using the current default values) to silence the warning messages.
q3_lr_classifier = LogisticRegression(solver = "liblinear", multi_class = "ovr")
q3_lr_classifier.fit(X = q3_train_term_matrix, y = train_labels)

predicted_labels = q3_lr_classifier.predict(X = q3_test_term_matrix)
print("Logistic regression classifier f1 score (C = 1.0): %.6f" % \
      metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro"))
print("Classification report:\n", \
     metrics.classification_report(y_true = test_labels, y_pred = predicted_labels, target_names = categories))


# -------------------------------------------------------------------------------- Question 6 -----

def order_vocabulary(vocabulary):

  ordered_vocabulary = [None] * len(vocabulary)
  for s in vocabulary.keys():
    ordered_vocabulary[vocabulary[s]] = s

  return ordered_vocabulary

def count_nonzero_weights(classifier, class_idx):

  nonzero_count = 0
  for weight in classifier.coef_[class_idx]:
    if weight != 0.0:
      nonzero_count += 1
  return nonzero_count

def print_nonzero_weights(classifier):

  nonzero_weight_total = 0
  print("Logistic regression classifier (penalty: %s) - Non-zero weights" % classifier.penalty)
  for class_idx in classifier.classes_:
    nonzero_weights = count_nonzero_weights(classifier, class_idx)
    print("  Class %d (%s): %d" % \
      (class_idx, newsgroups_train.target_names[class_idx], nonzero_weights))
    nonzero_weight_total += nonzero_weights
  print("  Total: %d\n" % nonzero_weight_total)

def get_features_with_nonzero_weights(vocabulary, classifier):

  features = []
  for i in range(len(classifier.coef_[0])):
    for class_idx in classifier.classes_:
      if classifier.coef_[class_idx][i] != 0.0:
        features.append(vocabulary[i])
        break

  return features

def get_c_adjusted_classifier_stats(reduced_vocabulary, m):

  c_adjusted_classifier_stats = []

  vectorizer = CountVectorizer(vocabulary = reduced_vocabulary)
  train_term_matrix = vectorizer.fit_transform(train_data)
  test_term_matrix = vectorizer.transform(test_data)

  for c in range(2, 102, 2):

    lr_l2_classifier = LogisticRegression(C = c / m, tol=.01, solver = "liblinear", multi_class = "ovr")
    lr_l2_classifier.fit(X = train_term_matrix, y = train_labels)

    nonzero_features = \
      get_features_with_nonzero_weights(reduced_vocabulary, lr_l2_classifier)

    predicted_labels = lr_l2_classifier.predict(X = test_term_matrix)
    f1 = metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro")

    c_adjusted_classifier_stats.append((c / m, f1))

  return c_adjusted_classifier_stats

np.random.seed(0)

q6_vectorizer = CountVectorizer()
q6_train_term_matrix = q6_vectorizer.fit_transform(train_data)
q6_test_term_matrix = q6_vectorizer.transform(test_data)

q6_lr_l1_classifier = LogisticRegression(penalty = "l1", solver = "liblinear", multi_class = "ovr")
q6_lr_l1_classifier.fit(X = q6_train_term_matrix, y = train_labels)
print_nonzero_weights(q6_lr_l1_classifier)

q6_lr_l2_classifier = LogisticRegression(penalty = "l2", solver = "liblinear", multi_class = "ovr")
q6_lr_l2_classifier.fit(X = q6_train_term_matrix, y = train_labels)
print_nonzero_weights(q6_lr_l2_classifier)

ordered_vocabulary = order_vocabulary(q6_vectorizer.vocabulary_)
reduced_vocabulary = get_features_with_nonzero_weights(ordered_vocabulary, q6_lr_l1_classifier)

q6_reduced_vectorizer = CountVectorizer(vocabulary = reduced_vocabulary)
q6_reduced_train_term_matrix = q6_reduced_vectorizer.fit_transform(train_data)
q6_reduced_test_term_matrix = q6_reduced_vectorizer.transform(test_data)

q6_reduced_lr_l2_classifier = LogisticRegression(penalty = "l2", solver = "liblinear", multi_class = "ovr")
q6_reduced_lr_l2_classifier.fit(X = q6_reduced_train_term_matrix, y = train_labels)
predicted_labels = q6_reduced_lr_l2_classifier.predict(X = q6_reduced_test_term_matrix)

print("Logistic regression classifier f1 score, reduced vocabulary (C = 1.0): %.6f" % \
      metrics.f1_score(y_true = test_labels, y_pred = predicted_labels, average = "micro"))
print("Classification report:\n", \
     metrics.classification_report(y_true = test_labels, y_pred = predicted_labels, target_names = categories))

print("Logistic regression classifier accuracy (C = 0.02 to 0.1)")
stats = get_c_adjusted_classifier_stats(reduced_vocabulary, 1000)

plt.plot([x[0] for x in stats], [y[1] for y in stats])
plt.xlabel("C (Inverse of Regularization Strength)")
plt.ylabel("f1 Score")
plt.title("f1 Scores vs Logistic Regression f1 Scores")
plt.ylim(0.64, 0.74)
plt.grid(True)
plt.show()

print("Logistic regression classifier accuracy (C = 2 to 100)")
stats = get_c_adjusted_classifier_stats(reduced_vocabulary, 1)

plt.plot([x[0] for x in stats], [y[1] for y in stats])
plt.xlabel("C (Inverse of Regularization Strength)")
plt.ylabel("f1 Score")
plt.title("f1 Scores vs Logistic Regression f1 Scores")
plt.ylim(0.64, 0.74)
plt.grid(True)
plt.show()
"""

# -------------------------------------------------------------------------------- Question 7 -----

# Create and fit the tf-idf vectorizer 
q7_vectorizer = TfidfVectorizer()
q7_train_term_matrix = q7_vectorizer.fit_transform(train_data)
q7_test_term_matrix = q7_vectorizer.transform(test_data)
q7_dev_term_matrix = q7_vectorizer.transform(dev_data)

# Create and traing the logistic regression model using training data
q7_lr_classifier = LogisticRegression(C = 100, solver = "liblinear", multi_class = "ovr")
q7_lr_classifier.fit(X = q7_train_term_matrix, y = train_labels)

predicted_probabilities = q7_lr_classifier.predict_proba(X = q7_dev_term_matrix)

# Maintain tuples (R ratio, document index) of the top three R ratios
top_r_ratios = [(0, 0)] * 3

# Iterate over all documents whose class was predicted
document_idx = 0
for class_probabilities in predicted_probabilities:
    
    # Identify the index of the document's class (0 through 3)
    class_idx = dev_labels[document_idx]
    
    # Calculate the R ratio
    r_ratio = predicted_probabilities.max() / class_probabilities[class_idx]
    
    # Probably more efficient that building a big probability->document_idx dictionary, sorting
    # it by key values, etc. but ugly nonetheless.  Since we're only preserving the top three
    # documents, the logic's easy to understand.
    if (r_ratio > top_r_ratios[0][0]):
        top_r_ratios[2] = top_r_ratios[1]
        top_r_ratios[1] = top_r_ratios[0]
        top_r_ratios[0] = (r_ratio, document_idx)
    elif (r_ratio > top_r_ratios[1][0]):
        top_r_ratios[2] = top_r_ratios[1]
        top_r_ratios[1] = (r_ratio, document_idx)
    elif (r_ratio > top_r_ratios[2][0]):
        top_r_ratios[2] = (r_ratio, document_idx)
        
    document_idx += 1

print(top_r_ratios)