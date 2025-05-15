import numpy as np
import itertools

def minimal_triangulation(Ps, us):
    """
    Ps: list of two (3x4) camera projection matrices as numpy arrays
    us: (2x2) numpy array of image coordinates, where each column corresponds to a point in one image
    Returns: 3D point (3,) as a numpy array
    """

    # Your code here
    list = []

    n = us.shape[1]

    for i in range(n):
        x_image, y_image = us[:, i]
        p1_camera, p2_camera, p3_camera = Ps[i]

        list.append(p3_camera * x_image - p1_camera)
        list.append(p3_camera * y_image - p2_camera)

    equ = np.vstack(list)

    _, _, v = np.linalg.svd(equ)
    X = v[-1]
    U = X / X[3]

    return U[:3]


def check_depths(Ps, U):
    """
    Ps: list of camera matrices (each 3x4)
    U: 3D point as a NumPy array of shape (3,)

    Returns: NumPy array of 0s and 1s indicating positive depth for each camera
    """

    # Your code here
    lam = []
    n = len(Ps)
    U_new = np.append(U, 1)

    for i in range(n):
        P = Ps[i]

        x = P @ U_new
        lam.append(x[2])


    positive = np.array(lam) > 0

    return positive


def reprojection_errors(Ps, us, U):
    """
    Ps: list of camera matrices (each 3x4 numpy array)
    us: 2xN numpy array of image coordinates (each column is a point)
    U: 3D point as a numpy array of shape (3,)

    Returns: Numpy array of reprojection errors (N,)
    """

    # Your code here
    n = len(Ps)
    U_new = np.append(U, 1)

    errors = []

    for i in range(n):
        P = Ps[i]

        x_rep = P @ U_new
        if x_rep[2] < 0:
            errors.append(np.inf)
        else:
            x_rep /= x_rep[2]
            residuals = us[:, i] - x_rep[:2]
            errors.append(np.sqrt(np.sum(residuals ** 2, axis=0)))

    return np.array(errors)


def ransac_triangulation(Ps, us, threshold):
    """
    Ps: list of camera matrices (each 3x4)
    us: 2xN numpy array of image coordinates
    threshold: reprojection error threshold for inlier selection

    Returns:
    - U: best estimated 3D point (3,)
    - nbr_inliers: number of inliers for best estimate
    """

    # Your code here
    nbr_inliers = -1
    best_U = None

    Ps = np.stack(Ps)
    n = len(Ps)

    for i, j in itertools.combinations(range(n), 2):
        index = [i, j]
        shuffled_Ps = Ps[index]
        shuffled_us = us[:, index]

        estimate_U = minimal_triangulation(shuffled_Ps[:2], shuffled_us[:, :2])

        errors = reprojection_errors(Ps, us, estimate_U)

        inlier = errors < threshold
        temp_nbr_inliers = inlier.sum()

        if temp_nbr_inliers > nbr_inliers:
            nbr_inliers = temp_nbr_inliers
            best_U = estimate_U

    return best_U, nbr_inliers


def compute_residuals(Ps, us, U):
    """
    Ps: list of camera matrices (each 3x4 numpy array)
    us: 2xN numpy array of observed image coordinates
    U: 3D point as a numpy array of shape (3,)

    Returns:
    - all_residuals: (2N,) numpy array of residuals (x and y reprojection errors)
    """

    # Your code here
    residuals = []

    U_new = np.append(U, 1)
    n = len(Ps)

    for i in range(n):
        p = Ps[i]

        estimate_x = p @ U_new

        error = us[:, i] - [estimate_x[0] / estimate_x[2], estimate_x[1] / estimate_x[2]]
        residuals.append((error))

    all_residuals = np.concatenate(residuals)

    return all_residuals

def compute_jacobian(Ps, U):
    """
    Ps: list of camera matrices (each 3x4 numpy array)
    U: 3D point as a numpy array of shape (3,)

    Returns:
    - jacobian: (2N x 3) numpy array, Jacobian matrix of reprojection errors w.r.t. U
    """

    # Your code here
    jacobian_list = []
    n = len(Ps)
    U_new = np.append(U, 1)
    for i in range(n):
        a, b, c = Ps[i]
        A = a @ U_new
        B = b @ U_new
        C = c @ U_new

        x = -(C * a - A * c) / (C * C)
        y = -(C * b - B * c) / (C * C)


        jacobian_list.append(x[:3])
        jacobian_list.append(y[:3])

    jacobian = np.vstack(jacobian_list)

    return jacobian

def refine_triangulation(Ps, us, Uhat, iterations=5):
    """
    Refines a 3D point estimate using Gauss-Newton optimization.

    Parameters:
    - Ps: list of camera matrices (3x4 numpy arrays)
    - us: 2xN numpy array of 2D image points
    - Uhat: initial estimate of the 3D point (3,)
    - iterations: number of Gauss-Newton iterations (default: 5)

    Returns:
    - U: refined 3D point (3,)
    """

    # Your code here
    x = Uhat

    for i in range(iterations):
        r = compute_residuals(Ps, us, x) # 2N
        j = compute_jacobian(Ps, x) # 2N x 3

        loss = np.sum(r ** 2)
        print(f"Iteration {i}: The sum of squared residuals: {loss}")

        gradient = j.T @ r # 3x
        step = np.linalg.inv(j.T @ j) # 3x3

        x -= step @ gradient

    print("Iteration End=======================")
    U_temp = x

    return U_temp