import numpy as np
from PIL import Image

def create_covariance_filter(pos_patches):
    """
    Creates a covariance filter from a set of positive patches.
    
    Parameters:
    pos_patches (numpy array): Array of positive patches.
    
    Returns:
    numpy array: The computed covariance filter.
    """
    nbr_pos = len(pos_patches)
    w = np.zeros_like(pos_patches[0])
    
    # Your code here
    for pos in pos_patches:
        w += pos

    w = w / nbr_pos

    mean = w.mean()
    w = (w - mean) / (w.shape[0] * w.shape[1])
    
    return w


def compute_threshold(pos_patches, neg_patches, w):
    """
    Computes the optimal threshold that minimizes misclassification.

    Parameters:
    pos_patches (numpy array): Foreground patches.
    neg_patches (numpy array): Background patches.
    w (numpy array): The covariance filter.

    Returns:
    float: Optimal threshold for classification.
    """
    thr_arr = np.arange(-0.001, 0.001, 0.0001)
    misclassified_result_arr = []
    specific_misclassified_result_arr = []

    for temp in thr_arr:
        pos_misclassified = 0
        neg_misclassified = 0

        for pos in pos_patches:
            pos_similarity = np.sum(pos * w)
            if pos_similarity < temp:
                pos_misclassified += 1

        for neg in neg_patches:
            neg_similarity = np.sum(neg * w)
            if neg_similarity > temp:
                neg_misclassified += 1

        misclassified_result_arr.append(pos_misclassified + neg_misclassified)
        specific_misclassified_result_arr.append([pos_misclassified, neg_misclassified])

    min_index = np.argmin(misclassified_result_arr)
    thr = thr_arr[min_index]
    specific_misclassified_result = specific_misclassified_result_arr[min_index]

    nbr_pos = len(pos_patches)
    nbr_neg = len(neg_patches)

    fn = specific_misclassified_result[0]
    fp = specific_misclassified_result[1]
    tp = nbr_pos - fn
    tn = nbr_neg - fp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    conf_matrix = [[tp, fn],
                   [fp, tn]]

    return thr, precision, recall, conf_matrix


def read_as_grayscale(image_path):
    image = Image.open(image_path).convert('L')

    image = np.array(image)

    image = image / 255

    return image
