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