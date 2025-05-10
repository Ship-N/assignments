import numpy as np
from supplied import match_descriptors, extract_sift_features, extract_keypoint_matches, affine_warp, transform_coordinates
from visualization import plot_matches

def affine_test_case(N: int):
    np.random.seed(42)

    pts = np.column_stack((
        # np.random.randint(0, 11, size=N),
        np.random.uniform(0, 11, size=N),
        np.random.uniform(0, 11, size=N)
    )).T

    A_true = np.column_stack((
        np.random.uniform(-10, 11, size=2),
        np.random.uniform(-10, 11, size=2)
    ))

    t_true = np.random.uniform(-10, 11, size=(2, 1))

    pts_tilde = A_true @ pts + t_true

    # print(f'{pts},\n {pts_tilde},\n {A_true},\n {t_true}')
    return pts, pts_tilde, A_true, t_true


def estimate_affine(pts, pts_tilde):
    A_list = []
    b_list = []

    for i in range(3):
        x, y = pts[:, i]
        x_p, y_p = pts_tilde[:, i]

        A_list.append([x, y, 0, 0, 1, 0])
        A_list.append([0, 0, x, y, 0, 1])
        b_list.append(x_p)
        b_list.append(y_p)

    right_A = np.array(A_list)
    left_b = np.array(b_list)

    h, _, _, _ = np.linalg.lstsq(right_A, left_b, rcond=None)

    A = h[:4].reshape(2, 2)
    t = h[4:].reshape(2, 1)

    return A, t

def residual_lgths(A, t, pts, pts_tilde):

    pts_predicted = A @ pts + t

    residuals = pts_tilde - pts_predicted

    # lgths = np.sqrt(np.sum(np.pow(residuals, 2), axis=0))
    lgths = np.sqrt(np.sum(residuals ** 2, axis=0))

    return lgths

def affine_test_case_outlier(outlier_rate, n_samples, image_height, image_width):
    np.random.seed(42)

    pts = np.column_stack((
        np.random.uniform(0, image_width, size=n_samples),
        np.random.uniform(0, image_height, size=n_samples)
    )).T

    A_true = np.column_stack((
        np.random.uniform(-1, 1, size=2),
        np.random.uniform(-1, 1, size=2)
    ))

    t_true = np.random.uniform(-1, 1, size=(2, 1))

    pts_tilde = A_true @ pts + t_true

    outlier_num = int(n_samples * outlier_rate)
    indices_to_true = np.random.choice(np.arange(0, n_samples), size=outlier_num, replace=False)

    outlier_idxs = np.zeros(n_samples, dtype=bool)
    outlier_idxs[indices_to_true] = True

    for i in indices_to_true:
        pts[0, i] += 10
        pts[1, i] += 10

    # print(f'{pts},\n {pts_tilde},\n {A_true},\n {t_true}')

    return pts, pts_tilde, A_true, t_true, outlier_idxs

def ransac_fit_affine(pts: np.ndarray, pts_tilde: np.ndarray, thresh: float, n_iter: int = 10000, max_inliers: int = 0):

    A = None
    t = None

    # Assume our expectation 0.1
    p = 0.1
    n = pts.shape[1]
    k = 0

    inliers_ratio = max(max_inliers / n, 0.5)
    max_k = np.log(p) / np.log(1 - inliers_ratio ** 3)

    while k < min(max_k, n_iter):
        permutation = np.random.permutation(n)

        shuffled_pts = pts[:, permutation]
        shuffled_pts_tilde = pts_tilde[:, permutation]

        A_est, t_est = estimate_affine(shuffled_pts, shuffled_pts_tilde)

        lgths = residual_lgths(A_est, t_est, pts, pts_tilde)
        inliers_count = np.sum(lgths < thresh)

        if inliers_count > max_inliers:
            max_inliers = inliers_count
            A = A_est
            t = t_est
            inliers_ratio = max(max_inliers / n, 1e-6)
            max_k = np.log(p) / np.log(1 - inliers_ratio ** 3)
        k += 1

    return A, t

def align_images(source: np.ndarray, target: np.ndarray, thresh: float = 10, plot: bool = True):
    src_kp, src_desc = extract_sift_features(source)
    tar_kp, tar_desc = extract_sift_features(target)

    gm, m = match_descriptors(src_desc, tar_desc)

    pts_src, pts_tar = extract_keypoint_matches(src_kp, tar_kp, gm)

    A_est, t_est = ransac_fit_affine(pts_tar, pts_src, thresh=thresh)

    # print(A_est)
    # print(t_est)

    T = np.vstack([np.hstack([A_est, t_est]), np.array([[0, 0, 1]])])

    # print(T)
    warped = affine_warp(source, T, target.shape)

    if plot:
        plot_matches(source, target, src_kp, tar_kp, m)

    return warped


def estimate_affine_ls(pts: np.ndarray, pts_tilde: np.ndarray):

    A_list = []
    b_list = []

    n = pts.shape[1]

    for i in range(n):
        x, y = pts[:, i]
        x_p, y_p = pts_tilde[:, i]

        A_list.append([x, y, 0, 0, 1, 0])
        A_list.append([0, 0, x, y, 0, 1])
        b_list.append(x_p)
        b_list.append(y_p)

    right_A = np.array(A_list)
    left_b = np.array(b_list)

    h, _, _, _ = np.linalg.lstsq(right_A, left_b, rcond=None)

    A = h[:4].reshape(2, 2)
    t = h[4:].reshape(2, 1)

    return A, t

def ransac_fit_affine_ls(pts: np.ndarray, pts_tilde: np.ndarray, thresh: float, n_iter: int = 10000, max_inliers: int = 0):
    # Assume our expectation 0.1
    p = 0.1
    n = pts.shape[1]
    k = 0

    inliers_ratio = max(max_inliers / n, 0.5)
    max_k = np.log(p) / np.log(1 - inliers_ratio ** 3)
    best_mask = None

    while k < min(max_k, n_iter):
        permutation = np.random.permutation(n)

        shuffled_pts = pts[:, permutation]
        shuffled_pts_tilde = pts_tilde[:, permutation]

        A_est, t_est = estimate_affine(shuffled_pts, shuffled_pts_tilde)

        lgths = residual_lgths(A_est, t_est, pts, pts_tilde)
        mask = lgths < thresh
        inliers_count = np.sum(mask)

        if inliers_count > max_inliers:
            max_inliers = inliers_count

            inliers_ratio = max(max_inliers / n, 1e-6)
            max_k = np.log(p) / np.log(1 - inliers_ratio ** 3)

            best_mask = mask
        k += 1

    A_ls, t_ls = estimate_affine_ls(pts[:, best_mask], pts_tilde[:, best_mask])
    return A_ls, t_ls

def align_images_inlier_ls(source: np.ndarray, target: np.ndarray, thresh: float = 10, plot: bool = True):
    src_kp, src_desc = extract_sift_features(source)
    tar_kp, tar_desc = extract_sift_features(target)

    gm, m = match_descriptors(src_desc, tar_desc)

    pts_src, pts_tar = extract_keypoint_matches(src_kp, tar_kp, gm)

    A_est, t_est = ransac_fit_affine_ls(pts_tar, pts_src, thresh=thresh)

    T = np.vstack([np.hstack([A_est, t_est]), np.array([[0, 0, 1]])])

    warped = affine_warp(source, T, target.shape)

    if plot:
        plot_matches(source, target, src_kp, tar_kp, m)

    return warped


def sample_image_at(image: np.ndarray, position):
    x, y = position
    xi, yi = int(x), int(y)

    H, W = image.shape
    if 0 <= yi < H and 0 <= xi < W:
        return int(image[yi, xi])
    else:
        return 255

def warp_16x16(source: np.ndarray):
    H, W = 16, 16
    warped = np.empty((H, W), dtype=source.dtype)

    for y_t in range(H):
        for x_t in range(W):

            x_s, y_s = transform_coordinates((x_t, y_t))

            warped[y_t, x_t] = sample_image_at(source, (x_s, y_s))

    return warped