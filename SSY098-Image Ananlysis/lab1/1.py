import numpy as np

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
    for pos_index in range(nbr_pos):
        pos = pos_patches[pos_index]

        mean = pos.mean()
        w += (pos - mean) / (pos.shape[0]+pos.shape[1])

    w = w / nbr_pos
    
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
    thr = 0
    lr = 0.1

    best_misclassified = 0

    for epoch in range(200):
        pos_misclassified = 0
        neg_misclassified = 0

        pos_misclassified_add = 0
        neg_misclassified_add = 0

        pos_misclassified_minus = 0
        neg_misclassified_minus = 0

        for pos in pos_patches:
            pos_similarity = np.sum(pos * w)
            if pos_similarity < thr:
                pos_misclassified += 1
            if pos_similarity < thr + lr:
                pos_misclassified_add += 1
            if pos_similarity < thr - lr:
                pos_misclassified_minus += 1

        for neg in neg_patches:
            neg_similarity = np.sum(neg * w)
            if neg_similarity > thr:
                neg_misclassified += 1
            if neg_similarity > thr + lr:
                neg_misclassified_add += 1
            if neg_similarity > thr - lr:
                neg_misclassified_minus += 1

        all_misclassified = pos_misclassified + neg_misclassified
        all_misclassified_minus = pos_misclassified_minus + neg_misclassified_minus
        all_misclassified_add = pos_misclassified_add + neg_misclassified_add

        if((all_misclassified_minus < all_misclassified_add) and (all_misclassified > all_misclassified_minus)):
            thr -= lr
            best_misclassified = all_misclassified_minus
        elif((all_misclassified_minus > all_misclassified_add) and (all_misclassified > all_misclassified_add)):
            thr += lr
            best_misclassified = all_misclassified_add
        else:
            best_misclassified = all_misclassified
            

    return thr