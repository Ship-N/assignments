import numpy as np
from scipy.ndimage import rotate

def partial_gradient(w, w0, example_train, label_train):
    """
    Computes the derivatives of the partial loss Li with respect to each of the classifier parameters.

    Parameters:
    - w: weight vector (same shape as example_train)
    - w0: bias term (scalar)
    - example_train: input image / example (same shape as w)
    - label_train: 0 or 1 (negative or positive example)

    Returns:
    - wgrad: gradient with respect to w
    - w0grad: gradient with respect to w0
    """

    # Write your code here

    wgrad = np.zeros_like(w)
    w0grad = 0

    for (example, label) in zip(example_train, label_train):
        y = np.sum(example * w) + w0
        # p = np.exp(y) / (1 + np.exp(y))
        p = 1 / (1 + np.exp(-y))

        wgrad += (p - label) * example
        w0grad += p - label

    return wgrad, w0grad


def process_epoch(w, w0, lrate, examples_train, labels_train, random_order=True):
    """
    Performs one epoch of stochastic gradient descent.

    Parameters:
    - w: weight array (same shape as examples)
    - w0: bias term (scalar)
    - lrate: learning rate (scalar)
    - examples_train: list or array of training examples (e.g., shape (N, 35, 35))
    - labels_train: array of labels (shape (N,))

    Returns:
    - Updated w and w0 after one epoch
    """

    # Write your code here
    np.random.seed(45)

    amount = len(examples_train)
    indices = range(amount)

    if(random_order):
        indices = np.random.permutation(amount)

    new_examples = examples_train[indices]
    new_labels = labels_train[indices]

    for (example, label) in zip(new_examples, new_labels):
        wgrad, w0grad = partial_gradient(w, w0, [example], [label])

        w = w - wgrad * lrate
        w0 = w0 - w0grad * lrate

    return w, w0


def classify(examples_val, w, w0):
    """
    Applies a classifier to the example data.

    Parameters:
    - examples_val: List of validation examples (each example is a 1D array)
    - w: weight array (same shape as each example in examples_val)
    - w0: bias term (scalar)

    Returns:
    - predicted_labels: Array of predicted labels (0 or 1) for each example
    """

    # Write your code here

    predicted_labels = []

    for example in examples_val:
        y = np.sum(example * w) + w0
        # p = np.exp(y) / (1 + np.exp(y))
        p = 1 / (1 + np.exp(-y))

        if(p > 0.5):
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return predicted_labels


def augment_data(examples_train, labels_train, M):
    """
    Data augmentation: Takes each sample of the original training data and
    applies M random rotations, which result in M new examples.

    Parameters:
    - examples_train: List of training examples (each example is a 2D array)
    - labels_train: Array of labels corresponding to the examples
    - M: Number of random rotations to apply to each training example

    Returns:
    - examples_train_aug: Augmented examples after rotations
    - labels_train_aug: Corresponding labels for augmented examples
    """
    # Write your code here
    examples_train_aug = []
    labels_train_aug = []

    for (example, label) in zip(examples_train, labels_train):
        angles = np.random.uniform(0, 360, size=M)
        for i in angles:
            aug_example = rotate(example, angle=i, reshape=False, order=1)
            examples_train_aug.append(aug_example)
            labels_train_aug.append(label)

    examples_train_aug = np.array(examples_train_aug)
    labels_train_aug = np.array(labels_train_aug)

    return examples_train_aug, labels_train_aug