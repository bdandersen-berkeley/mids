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

# ----------

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