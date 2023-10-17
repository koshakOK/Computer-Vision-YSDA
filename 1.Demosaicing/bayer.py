import numpy as np
import copy


def get_bayer_masks(n_rows, n_cols):
    red_filter = np.array([[0, 1], [0, 0]])
    red_filter = np.tile(red_filter, (int((n_rows + 1) / 2),
                         int((n_cols + 1) / 2)))[:n_rows, :n_cols]

    green_filter = np.array([[1, 0], [0, 1]])
    green_filter = np.tile(green_filter,
                           (int((n_rows + 1) / 2), int((n_cols + 1) / 2)))[:n_rows, :n_cols]

    blue_filter = np.array([[0, 0], [1, 0]])
    blue_filter = np.tile(blue_filter,
                          (int((n_rows + 1) / 2), int((n_cols + 1) / 2)))[:n_rows, :n_cols]
    filters = np.dstack((red_filter, green_filter, blue_filter))
    return filters.astype('bool')


def get_colored_img(raw_img):
    n_rows = raw_img.shape[0]
    n_cols = raw_img.shape[1]

    red_filter = raw_img * get_bayer_masks(n_rows, n_cols)[:, :, 0]
    green_filter = raw_img * get_bayer_masks(n_rows, n_cols)[:, :, 1]
    blue_filter = raw_img * get_bayer_masks(n_rows, n_cols)[:, :, 2]
    filters = np.dstack((red_filter, green_filter, blue_filter))
    return filters


def bilinear_interpolation(colored_img):
    n_rows = colored_img.shape[0]
    n_cols = colored_img.shape[1]

    red_filter = copy.deepcopy(colored_img[:, :, 0])
    green_filter = copy.deepcopy(colored_img[:, :, 1])
    blue_filter = copy.deepcopy(colored_img[:, :, 2])

    red_filter_copy = copy.deepcopy(colored_img[:, :, 0])
    green_filter_copy = copy.deepcopy(colored_img[:, :, 1])
    blue_filter_copy = copy.deepcopy(colored_img[:, :, 2])

    for i in np.arange(1, n_rows - 1):
        for j in np.arange(1, n_cols - 1):
            if red_filter[i, j] == 0 and (i % 2 != 0 or (j + 1) % 2 != 0):
                summ_red = 0
                exist_red = 0
                for ii in np.arange(i - 1, i + 2):
                    for jj in np.arange(j - 1, j + 2):
                        if red_filter[ii, jj] or (ii % 2 == 0 and (jj + 1) % 2 == 0):
                            summ_red += red_filter[ii, jj]
                            exist_red += 1
                red_filter_copy[i, j] = float(
                    summ_red)/exist_red if summ_red else 0

            if green_filter[i, j] == 0 and ((i + 1) % 2 != 0 or (j + 1) % 2 != 0) and (i % 2 != 0 or j % 2 != 0):
                summ_green = 0
                exist_green = 0
                for ii in np.arange(i - 1, i + 2):
                    for jj in np.arange(j - 1, j + 2):
                        if green_filter[ii, jj] or ((ii + 1) % 2 == 0 and (jj + 1) % 2 == 0) or (ii % 2 == 0 and jj % 2 == 0):
                            summ_green += green_filter[ii, jj]
                            exist_green += 1
                green_filter_copy[i, j] = float(
                    summ_green)/exist_green if summ_green else 0

            if blue_filter[i, j] == 0 and ((i + 1) % 2 != 0 or j % 2 != 0):
                summ_blue = 0
                exist_blue = 0
                for ii in np.arange(i - 1, i + 2):
                    for jj in np.arange(j - 1, j + 2):
                        if blue_filter[ii, jj] or ((ii + 1) % 2 == 0 and jj % 2 == 0):
                            summ_blue += blue_filter[ii, jj]
                            exist_blue += 1
                blue_filter_copy[i, j] = float(
                    summ_blue)/exist_blue if summ_blue else 0

    filters = np.dstack((red_filter_copy, green_filter_copy, blue_filter_copy))

    filters[0, :, :] = 0
    filters[:, 0, :] = 0
    filters[n_rows - 1, :, :] = 0
    filters[:, n_cols - 1, :] = 0
    return filters


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype("float64")
    img_gt = img_gt.astype("float64")
    C = img_gt.shape[2]
    H = img_gt.shape[0]
    W = img_gt.shape[1]
    mse = 0
    for h in range(H):
        for w in range(W):
            for c in range(C):
                mse += (img_pred[h, w, c] - img_gt[h, w, c])**2
    if mse == 0:
        raise ValueError
    mse = float(mse) / (C*H*W)
    return 10*np.log10((float(np.max(img_gt)**2))/mse)
