import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(image, n_keypoints=200):
    """Find keypoints and their descriptors in image.

    image ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    de = ORB(n_keypoints=n_keypoints)
    de.detect_and_extract(rgb2gray(image))
    return de.keypoints, de.descriptors


def pad_points(points):
    return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)


def divide(nom, denom):
    return np.true_divide(nom, denom, out=np.zeros_like(nom), where=denom != 0)


def apply_homography(homography, points):
    padded_points = pad_points(points)
    transformed_points = padded_points @ homography.T
    transformed_coords = transformed_points[:, :2]
    transformed_scale = transformed_points[:, 2]
    return divide(transformed_coords, transformed_scale[:, None])


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    center = np.mean(points, axis=0)
    mean_center_dist = np.mean(np.linalg.norm(
        points - center[None, :], axis=1))
    norm_const = np.sqrt(2) / mean_center_dist

    normalization_matrix = np.array(
        [
            [norm_const, 0, -norm_const * center[0]],
            [0, norm_const, -norm_const * center[1]],
            [0, 0, 1],
        ]
    )
    transformed_points = apply_homography(normalization_matrix, points)
    return normalization_matrix, transformed_points


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    points = len(src_keypoints)

    src_matrix, src_keypoints = center_and_normalize_points(src_keypoints)
    dest_matrix, dest_keypoints = center_and_normalize_points(dest_keypoints)

    # ax =(−x1,−y1,−1,0,0,0,x′2x1,x′2y1,x′2)T
    alpha_x = np.stack(
        [
            -src_keypoints[:, 0],
            -src_keypoints[:, 1],
            -np.ones(points),
            np.zeros(points),
            np.zeros(points),
            np.zeros(points),
            dest_keypoints[:, 0] * src_keypoints[:, 0],
            dest_keypoints[:, 0] * src_keypoints[:, 1],
            dest_keypoints[:, 0],
        ],
        axis=1,
    )

    # ay = (0,0,0,−x1,−y1,−1,y2′ x1,y2′ y1,y2′ )T
    alpha_y = np.stack(
        [
            np.zeros(points),
            np.zeros(points),
            np.zeros(points),
            -src_keypoints[:, 0],
            -src_keypoints[:, 1],
            -np.ones(points),
            dest_keypoints[:, 1] * src_keypoints[:, 0],
            dest_keypoints[:, 1] * src_keypoints[:, 1],
            dest_keypoints[:, 1],
        ],
        axis=1,
    )

    alpha = np.concatenate([alpha_x, alpha_y], axis=0)
    _, _, alpha_v = np.linalg.svd(alpha)
    last_v = alpha_v[-1]
    homography = np.linalg.inv(dest_matrix) @ last_v.reshape(3, 3) @ src_matrix

    return homography


def ransac_transform(
    src_keypoints,
    src_descriptors,
    dest_keypoints,
    dest_descriptors,
    max_trials=1000,
    residual_threshold=1.0,
    return_matches=False,
):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    matches = match_descriptors(src_descriptors, dest_descriptors)
    src_matched_keypoints, dest_matched_keypoints = (
        src_keypoints[matches[:, 0]],
        dest_keypoints[matches[:, 1]],
    )
    max_inliers, inliers = None, None
    for _ in range(max_trials):
        point_pairs = np.random.choice(len(matches), size=4, replace=False)
        homography = find_homography(
            src_matched_keypoints[point_pairs], dest_matched_keypoints[point_pairs]
        )
        transformed_src_keypoints = apply_homography(
            homography, src_matched_keypoints)
        transformation_distances = np.linalg.norm(
            dest_matched_keypoints - transformed_src_keypoints, axis=1
        )

        inliers_mask = transformation_distances < residual_threshold
        n_inliers = np.sum(inliers_mask)
        if max_inliers is None or n_inliers > max_inliers:
            max_inliers = n_inliers
            inliers = matches[inliers_mask]

    homography = find_homography(
        src_keypoints[inliers[:, 0]], dest_keypoints[inliers[:, 1]]
    )
    homography_transform = ProjectiveTransform(homography)
    if return_matches:
        return homography_transform, inliers

    return homography_transform


def inverse_transform(forward_transform):
    return ProjectiveTransform(np.linalg.inv(forward_transform.params))


def combine_transforms(forward_transforms):
    return sum(forward_transforms, start=ProjectiveTransform())


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count

    for i in range(len(result)):
        if i < center_index:
            # Sum all transforms from ith image to the center image (center_index is not included in range).
            result[i] = combine_transforms(
                forward_transforms[j] for j in range(i, center_index)
            )
        elif i == center_index:
            result[i] = ProjectiveTransform()
        else:
            # A bit of dark magic here: result[center_index] is the identity transform, but forward_transforms[center_index]
            # is the transform from center image to the first one to the right. And center_index is included in range whilst
            # i is not. Which yields the following code.
            result[i] = combine_transforms(
                inverse_transform(forward_transforms[j])
                for j in reversed(range(center_index, i))
            )

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for image, transform in zip(image_collection, center_warps):
        height, width, _ = image.shape
        corners = np.array([[0, 0], [height, 0], [height, width], [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
    """
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coords, max_coords = get_min_max_coords(corners)
    min_coords, max_coords = min_coords[::-1], max_coords[::-1]
    shift_matrix = np.array(
        [[1, 0, -min_coords[0]], [0, 1, -min_coords[1]], [0, 0, 1]])
    shift = ProjectiveTransform(shift_matrix)
    return (
        tuple(warp + shift for warp in simple_center_warps),
        tuple((max_coords - min_coords).astype(int)),
    )


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    inversed_transform = rotate_transform_matrix(inverse_transform(transform))

    warped_image = warp(image, inversed_transform, output_shape=output_shape)

    mask = np.ones(image.shape[:2])
    warped_mask = warp(mask, inversed_transform,
                       output_shape=output_shape).astype(bool)

    return warped_image, warped_mask


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=bool)

    for image, transform in zip(image_collection, final_center_warps):
        warped_image, warped_mask = warp_image(image, transform, output_shape)
        result[warped_mask] = warped_image[warped_mask]
        result_mask[warped_mask] = True

    return result


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    gaussian_pyramid = [image]
    for _ in range(n_layers - 1):
        gaussian_pyramid.append(
            gaussian(gaussian_pyramid[-1], sigma, multichannel=True))

    return tuple(gaussian_pyramid)


def get_laplacian_pyramid(image, n_layers=4, sigma=3):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    gaussian_pyramid = get_gaussian_pyramid(image, n_layers, sigma)
    laplacian_pyramid = []
    for i in range(n_layers - 1):
        laplacian_pyramid.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])

    laplacian_pyramid.append(gaussian_pyramid[-1])
    return tuple(laplacian_pyramid)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for image in image_collection:
        image = image.copy()
        for i in range(image.shape[-1]):
            image[:, :, i] -= image[:, :, i].min()
            image[:, :, i] /= image[:, :, i].max()
        result.append(image)

    return result


def gaussian_merge_pano(
    image_collection,
    final_center_warps,
    output_shape,
    n_layers=4,
    image_sigma=2,
    merge_sigma=10,
):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
    (output_shape) np.ndarray: final pano
    """
    assert image_collection, "image collection should have at least one image"
    warped_image, warped_mask = warp_image(
        image_collection[0], final_center_warps[0], output_shape
    )
    images, masks = [warped_image], [warped_mask]
    result_mask = np.copy(warped_mask)

    for image, transform in zip(image_collection[1:], final_center_warps[1:]):
        warped_image, warped_mask = warp_image(image, transform, output_shape)
        crossing_region = result_mask & warped_mask
        assert np.any(crossing_region), "consecutive images don't cross"

        _, crossing_cols = np.where(crossing_region)
        crossing_min_col = np.min(crossing_cols)
        crossing_max_col = np.max(crossing_cols)
        crossing_mid_col = (crossing_min_col + crossing_max_col) // 2
        crossing_region[:, crossing_mid_col:] = False
        warped_mask[crossing_region] = False
        images.append(warped_image)
        masks.append(warped_mask)

        result_mask[warped_mask] = True

    mask_pyramids = [
        get_gaussian_pyramid(mask.astype(float), n_layers, merge_sigma) for mask in masks
    ]
    layer_sums = [
        sum(mask_pyramid[layer] for mask_pyramid in mask_pyramids)
        for layer in range(n_layers)
    ]
    mask_pyramids = [
        [divide(mask, layer_sum)
         for mask, layer_sum in zip(mask_pyramid, layer_sums)]
        for mask_pyramid in mask_pyramids
    ]

    image_pyramids = [
        get_laplacian_pyramid(image, n_layers, image_sigma) for image in images
    ]

    merged_image = sum(
        sum(
            image * mask[:, :, None] for image, mask in zip(image_pyramid, mask_pyramid)
        )
        for image_pyramid, mask_pyramid in zip(image_pyramids, mask_pyramids)
    )

    return merged_image


def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    K = np.array(
        [
            [scale, 0, w / 2],
            [0, scale, h / 2],
            [0, 0, 1]
        ]
    )
    coords = apply_homography(np.linalg.inv(K), coords)
    cylindrical_coords = np.stack(
        [np.tan(coords[:, 0]), divide(coords[:, 1], np.cos(coords[:, 0]))], axis=-1)
    cylindrical_coords = apply_homography(K, cylindrical_coords)
    return cylindrical_coords


def warp_cylindrical(image, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    image ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : warped image (H and W may differ from original)
    """
    h, w, _ = image.shape
    if scale is None:
        scale = w / 2.5

    warped_image = warp(
        image, lambda coords: cylindrical_inverse_map(coords, h, w, scale))
    if crop:
        # Remove channels dim
        has_nonzero = np.any(warped_image != 0, axis=2)

        # Rows
        has_nonzero_rows = np.any(has_nonzero, axis=1)
        warped_image = warped_image[has_nonzero_rows, :, :]

        # Cols
        has_nonzero_cols = np.any(has_nonzero, axis=0)
        warped_image = warped_image[:, has_nonzero_cols, :]

    return warped_image
