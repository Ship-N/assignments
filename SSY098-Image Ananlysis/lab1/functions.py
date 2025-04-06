import numpy as np
from PIL import Image
from scipy.ndimage import rank_filter
import cv2
from supplied import *

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

    mean = w.mean(axis=(0, 1), keepdims=True)
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

def compute_threshold_improved(pos_patches, neg_patches, w):
    """
    Computes the optimal threshold that minimizes misclassification.

    Parameters:
    pos_patches (numpy array): Foreground patches.
    neg_patches (numpy array): Background patches.
    w (numpy array): The covariance filter.

    Returns:
    float: Optimal threshold for classification.
    """
    pos_scores = np.array([np.sum(pos * w) for pos in pos_patches])
    neg_scores = np.array([np.sum(neg * w) for neg in neg_patches])
    all_scores = np.concatenate((pos_scores, neg_scores))

    thr_arr = np.linspace(min(all_scores), max(all_scores), 1000)

    misclassified_result_arr = []
    specific_misclassified_result_arr = []

    for temp in thr_arr:
        pos_misclassified = 0
        neg_misclassified = 0

        for pos in pos_scores:
            if pos < temp:
                pos_misclassified += 1

        for neg in neg_scores:
            if neg > temp:
                neg_misclassified += 1

        misclassified_result_arr.append(pos_misclassified + neg_misclassified)
        specific_misclassified_result_arr.append([pos_misclassified, neg_misclassified])

    min_index = np.argmin(misclassified_result_arr)
    thr = thr_arr[min_index]
    # misclassified_result = misclassified_result_arr[min_index]

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


def strict_local_maxima(response, threshold):
    """
    Computes the coordinates of all strict local maxima in the response image.

    Parameters:
    response (numpy array): Input response image.
    threshold (float): Threshold for classification

    Returns:
    numpy array: 2 x n array with column coordinates in the first row
                 and row coordinates in the second row.
    """

    nhood_size = (3, 3)
    next_best = rank_filter(response, -2, size=nhood_size)  # Selecting the second highest pixel value from the neighborhood of each pixel.

    # Your code here
    mask = (response > next_best) & (response > threshold)
    row_coords, col_coords = np.nonzero(mask)

    return (col_coords, row_coords)


def detector(image, w, thr):
    """
    Detects cell centers in an image using a linear classifier and non-maximum suppression.

    Parameters:
    image (numpy array): Input image
    w (numpy array): The covariance filter
    thr (float): Threshold for classification

    Returns:
    numpy array: Cell centers.
    numpy array: Thresholded response image
    """

    # Your code here
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = np.zeros(image.shape[:2])
        for c in range(3):
            channel_result = cv2.filter2D(image[:, :, c], -1, w)
            result += channel_result

        result = result / 3
    else:
        result = cv2.filter2D(image, -1, w)

    centers = strict_local_maxima(result, thr)

    thresholded_response = np.zeros_like(result)
    thresholded_response[result > thr] = 1

    return centers, thresholded_response

# Task 2:
def read_image(image_path):

    # Load the image
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image = np.array(image)

    # Convert image to 0-1 range
    image = image / 255

    return image

def classify_church(image_path, training_data):

    keypoints, descriptors = extract_sift_features(image_path)
    if descriptors is None:
        return None


    matches = match_descriptors(descriptors, training_data)
    if len(matches) == 0:
        return None


    votes = [0] * len(training_data['names'])
    for m in matches:
        train_label = training_data['labels'][m.trainIdx]
        votes[train_label] += 1


    predicted_label = votes.index(max(votes))
    return predicted_label