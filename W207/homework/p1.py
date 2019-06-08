# This tells matplotlib not to try opening a new window for each plot.
# For iPython notebooks only!
# %matplotlib inline

# Import a bunch of libraries.
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Set the randomizer seed so results are the same each time.
np.random.seed(0)

# ----------

# Load the digit data from https://www.openml.org/d/554 or from default local location '~/scikit_learn_data/...'
X, Y = fetch_openml(name='mnist_784', return_X_y=True, cache=False)


# Rescale grayscale values to [0,1].
X = X / 255.0

# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

# print 'data shape: ', X.shape
# print 'label shape:', Y.shape
print('data shape: ', X.shape)
print('label shape:', Y.shape)


# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[61000:], Y[61000:]
dev_data, dev_labels = X[60000:61000], Y[60000:61000]
train_data, train_labels = X[:60000], Y[:60000]
mini_train_data, mini_train_labels = X[:1000], Y[:1000]

# -------------------------------------------------------------------------------- Question 1 -----

'''
def get_example_digit_indices(digit_list, num_examples = 10):
    """
    Retrieves randomly-selected indices within MNIST data identifying digits 0 through 9.

    Indices are returned as a list of lists: the former list of length 10, one list per digit in
    zero-indexed numeric order; the latter lists being indices within specified MNIST data 
    classified as a digit associated with the list (e.g. indices of 0s in one list, indices
    of 1s in the next, etc.)

    Arguments
    ---------
    digit_list: list
    
    List of numbers whose values identify digit classifications (e.g. '3', '5') and
    whose indices map to the digit's data within the MNIST image data (required)

    num_examples: int
    
    Number of examples of each digit to identify and whose indices to retrieve
    (default: 10)

    Returns
    -------
    List of lists.  Specifically, 10 lists (one for each digit 0 through 9) of indices within 
    MNIST digit image data.  Each index is associated with image data for a digit of the
    associated classification (i.e. the first list contains a list of indices identifying images
    classified as 0s, the second list contains a list of indices identifying images classified as
    1s, etc.)
    """

    assert (type(digit_list) is np.ndarray), "digit_list must be of type NumPy ndarray"
    assert (digit_list.size > 0), "digit_list must not be empty"
    assert (type(num_examples) is int), "num_examples must be of type int"
    assert (0 < num_examples & num_examples <= 10), "num_examples must be an int between 1 and 10"

    # Create the list of lists to populate with image indices
    digit_indices = [[] for i in range(10)]

    # How many lists we have filled with the requisite number of digit indices
    complete_index_lists = 0

    # Randomly select an index from which to be sourcing image data.  Because we iterate through
    # the data and loop to the beginning should we reach the end of the data without completing 
    # all 10 lists, also identify the index at which to stop.
    current_digit_list_idx = np.random.randint(1, digit_list.size)
    final_digit_list_idx = current_digit_list_idx - 1

    # Iterate through the digit data, exiting only if we've filled all 10 lists with the requisite
    # number of digit indices, or if we've sourced/examined all digit data
    while (complete_index_lists < 10 and current_digit_list_idx != final_digit_list_idx):

        # Get the element, ensuring it is a string representing a single digit (e.g. '3', '5')
        example_digit = digit_list[current_digit_list_idx]
        if (type(example_digit) is str and example_digit.isdigit() and len(example_digit) == 1):

            # Convert digit string to integer value
            example_digit = int(example_digit)
            
            # If the list of indices associated with this digit is not filled to the requisite
            # number, append it to the list
            if (len(digit_indices[example_digit]) < num_examples):
                digit_indices[example_digit].append(current_digit_list_idx)

                # If the list of indices is now filled, increment the number of completed lists
                if (len(digit_indices[example_digit]) == num_examples):
                    complete_index_lists += 1

        # Move to the next element in the digit data
        current_digit_list_idx += 1
        if (current_digit_list_idx == digit_list.size):
            current_digit_list_idx = 0

    return digit_indices

# Number of images of each digit to retrieve from the MNIST data and render
IMAGE_NUM = 10

# Retrieve the lists of indices identifying digits' image data within the MNIST data
digit_indices = get_example_digit_indices(digit_list = Y, num_examples = IMAGE_NUM)

# Create the 10 x 10 grid of axes (i.e. subplots for each digit image)
fig, ax = plt.subplots(nrows = 10, ncols = IMAGE_NUM)
plt.title("Randomly-Selected MNIST Digit Images")

# Iterate through the list of lists, rendering each digit image associated with the current
# index.  An assumption is made that each image is represented using 784 values (a matrix of
# 28 x 28) between 0 and 1, inclusive.
for i in range(0, 10):
    for j in range(0, IMAGE_NUM):
        ax[i][j].axis("off")
        ax[i][j].imshow(
            X = np.reshape(X[digit_indices[i][j]], (28, 28)),
            aspect = "auto",
            cmap = plt.get_cmap("Greys")
        )

plt.show()
'''

# -------------------------------------------------------------------------------- Question 2 -----

'''
def P2(k_values):
    """
    Evaluates KNearestNeighbor classifier accuracy using models trained on "mini" data set.

    Arguments
    ---------
    k_values: list

    List of k values with which to create and evaluate KNearestNeighbor classifiers
    """

    assert (type(k_values) is list), "k_values must be of type list"
    assert (len(k_values) > 0), "k_values list must not be empty"

    # Iterate over the list of k values, creating and evaluating the accuracy of 
    # K-nearest-neighbor classifiers for each value of k
    k_list = list()
    accuracy_list = list()
    k1_predicted = None
    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(X = mini_train_data, y = mini_train_labels)
        k_list.append(str(k))
        accuracy_list.append(classifier.score(X = dev_data, y = dev_labels))
        if (k == 1):
            k1_predicted = classifier.predict(X = dev_data)

    # Plot the data using a horizontal histogram
    fig, ax = plt.subplots()
    xlim_minimum = min(accuracy_list) * 0.99
    ax.set(
        xlim = [xlim_minimum, max(accuracy_list) * 1.01],
        title = "KNearest Neighbor Classifier Accuracy\n(Train: mini, Test: dev)",
        xlabel = "Mean Accuracy",
        ylabel = "Neighbors (k)"
    )
    for i, accuracy in enumerate(accuracy_list):
        ax.text(xlim_minimum + (accuracy - xlim_minimum) / 2, y = i, s = accuracy, color = "white")
    ax.barh(k_list, accuracy_list)
    plt.show()

    # Print the classification report showing the performance of the k=1 classifier for each digit
    # represented in the development data set
    print(classification_report(dev_labels, k1_predicted))


k_values = [1, 3, 5, 7, 9]
P2(k_values)
'''

# -------------------------------------------------------------------------------- Question 3 -----
'''
def P3(train_sizes, accuracies):
    """
    Trains, evaluates accuracy, and performs MNIST data predictions using specified training set
    sizes.

    A KNearestNeighbor classifier with k = 1 is used for each training set size.

    Arguments
    ---------
    train_sizes: list

    Integer list of sizes of various training data sets with which to train that classifier

    accuracies: list

    List with no elements.  P3 returns classifier accuracy and elapsed time for predictions in the
    form of a two-element tuple -- one tuple per training set size evaluated.  Elements of each
    tuple are classifier accuracy and elapsed time of prediction in seconds.
    """

    assert (type(train_sizes) is list), "train_sizes must be of type list"
    assert (len(train_sizes) > 0), "train_sizes must not be empty"
    assert (all(isinstance(train_size, int) for train_size in train_sizes)), "train_sizes list must contain only integers"
    assert (type(accuracies) is list), "accuracies must be of type list"
    assert (len(accuracies) == 0), "accuracies must be empty"

    # Iterate through the list of training data sizes, building and evaluating KNearestNeighbor
    # classifiers of the specified sizes
    for train_size in train_sizes:

        # Identify the training data set
        cur_train_data, cur_train_labels = X[:train_size], Y[:train_size]

        # Create and train/fit the KNearestNeighbor classifier using k = 1
        cur_classifier = KNeighborsClassifier(n_neighbors = 1)
        cur_classifier.fit(X = cur_train_data, y = cur_train_labels)

        # Calculate the classifier's accuracy
        cur_accuracy = cur_classifier.score(X = dev_data, y = dev_labels)

        # Perform predictions using test data, and calculate each prediction's elapsed time
        time_begin = time.time()
        cur_classifier.predict(X = dev_data)
        seconds_elapsed = time.time() - time_begin

        # Append the accuracy-elapsed time tuple to the specified "accuracies" list
        accuracies.append((cur_accuracy, seconds_elapsed))

train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000]
accuracies = []
P3(train_sizes, accuracies)

# Create lists from each tuple returned by P3
accuracy_list = [accuracy[0] for accuracy in accuracies]
elapsed_time_list = [round(accuracy[1], 3) for accuracy in accuracies]

# Create and render the horizontal histogram for classifier accuracies
fig, ax = plt.subplots()
xlim_minimum = min(accuracy_list) * 0.9
ax.set(
    xlim = [xlim_minimum, max(accuracy_list) * 1.01],
    title = "KNearest Neighbor Classifier Accuracy\n(Train: <variable>, Test: dev)",
    xlabel = "Mean Accuracy",
    ylabel = "Training Size"
)
for i, accuracy in enumerate(accuracy_list):
    ax.text(xlim_minimum + (accuracy - xlim_minimum) / 2, y = i, s = accuracy, color = "white")
ax.barh([str(train_size) for train_size in train_sizes], accuracy_list)
plt.show()

# Create and render the horizontal histogram for classifiers' prediction's elapsed times
fig, ax = plt.subplots()
ax.set(
    xlim = [0, max(elapsed_time_list) * 1.01],
    title = "KNearest Neighbor Predictions - Elapsed Times\n(Train: <variable>, Test: dev)",
    xlabel = "Elapsed Time (Seconds)",
    ylabel = "Training Size"
)
for i, elapsed_time in enumerate(elapsed_time_list):
    if (elapsed_time < 10):
        ax.text(elapsed_time + 1, y = i, s = elapsed_time)
    else:
        ax.text(
            xlim_minimum + (elapsed_time - xlim_minimum) / 2, 
            y = i, 
            s = elapsed_time,
            color = "white"
        )
ax.barh([str(train_size) for train_size in train_sizes], elapsed_time_list)
plt.show()
'''

# -------------------------------------------------------------------------------- Question 5 -----
'''
# def P5():

# Create and train/fit the KNearestNeighbor classifier using k = 1
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X = train_data, y = train_labels)
predicted = classifier.predict(X = dev_data)
print(classification_report(dev_labels, predicted))

confusion_mx = confusion_matrix(dev_labels, predicted)
print(confusion_mx)

for i in range(len(dev_labels)):
    if (dev_labels[i] != predicted[i]):
        print("Index:", i, "Actual:", dev_labels[i], "Predicted:", predicted[i])

   # Create the 10 x 10 grid of axes (i.e. subplots for each digit image)
    fig, ax = plt.subplots(nrows = 10, ncols = 3)

    # Iterate through the list of lists, rendering each digit image associated with the current
    # index.  An assumption is made that each image is represented using 784 values (a matrix of
    # 28 x 28) between 0 and 1, inclusive.
    for i in range(0, 10):
        for j in range(0, 3):
            ax[i][j].axis("off")
            ax[i][j].imshow(
                X = np.reshape(X[95], (28, 28)),
                aspect = "auto",
                cmap = plt.get_cmap("Greys")
            )

    plt.show()
'''

# -------------------------------------------------------------------------------- Question 6 -----

def get_valid_neighbor_coordinates(pixel, dimension = (28, 28)):
    """
    Retrieves a list of coordinates adjacent to the specified pixel.
    """
    
    assert (type(pixel) == tuple), "pixel must be of type tuple"
    assert (type(pixel[0]) == int and type(pixel[1]) == int), "pixel coordinates must be integers"
    assert (pixel[0] >= 0 and pixel[1] >= 0), "pixel coordinates must be greater than 0"
    
    assert (type(dimension) == tuple), "dimension must be of type tuple"
    assert (type(dimension[0]) == int and type(dimension[1]) == int), "dimension coordinates must be integers"
    assert (dimension[0] >= 0 and dimension[1] >= 0), "dimension coordinates must be greater than 0"
    
    assert (pixel[0] < dimension[0] and pixel[1] < dimension[1]), "pixel coordinates must be within dimension bounds"

    col_neighbors = [pixel[0] - 1, pixel[0], pixel[0] + 1]
    row_neighbors = [pixel[1] - 1, pixel[1], pixel[1] + 1]
    
    if (col_neighbors[2] > dimension[0] - 1):
        del col_neighbors[2]
    if (col_neighbors[0] < 0):
        del col_neighbors[0]
        
    if (row_neighbors[2] > dimension[1] - 1):
        del row_neighbors[2]
    if (row_neighbors[0] < 0):
        del row_neighbors[0]
    
    neighbors = list()
    for row in row_neighbors:
        for col in col_neighbors:
            neighbors.append((col, row))
            
    return neighbors

def blur(image, dimension = (28, 28)):

    assert (type(image) == np.ndarray), "image must be of type np.ndarray"

    image = np.reshape(image, dimension)

    blurred_image = [[0.0 for i in range(dimension[0])] for j in range(dimension[1])]

    for i in range(dimension[0]):
        for j in range(dimension[1]):

            neighbors = get_valid_neighbor_coordinates((i, j))

            sum_of_neighbors = 0.0
            for neighbor in neighbors:
                sum_of_neighbors += image[neighbor[0]][neighbor[1]]
            blurred_image[i][j] = sum_of_neighbors / len(neighbors)

    return blurred_image

blurred_image = blur(X[2])
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(
    blurred_image,
    aspect = "auto",
    cmap = plt.get_cmap("Greys")
)
plt.show()
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(
    np.reshape(X[2], (28, 28)),
    aspect = "auto",
    cmap = plt.get_cmap("Greys")
)
plt.show()