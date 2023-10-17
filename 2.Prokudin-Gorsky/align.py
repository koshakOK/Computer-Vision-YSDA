import numpy as np


def align(img, g_coord):
    h = img.shape[0]
    w = img.shape[1]

    border = int(1/3 * h)

    img_blue = img[:border, :]
    img_green = img[border:2*border, :]
    img_red = img[2*border:3*border, :]

    alpha = 0.11
    cut_img_blue = img_blue[int(h*alpha):int(h*(1-alpha)),
                            int(w*alpha):int(w*(1-alpha))]
    cut_img_green = img_green[int(
        h*alpha):int(h*(1-alpha)), int(w*alpha):int(w*(1-alpha))]
    cut_img_red = img_red[int(h*alpha):int(h*(1-alpha)),
                          int(w*alpha):int(w*(1-alpha))]
    # print(int(h*alpha))

    im_fft_green = np.fft.fft2(cut_img_green)

    im_fft_blue = np.fft.fft2(cut_img_blue)
    # (im_fft_green * np.conj(im_fft_blue))
    blue_mat = np.fft.ifft2(im_fft_blue * np.conj(im_fft_green))
    b_row_diff, b_col_diff = np.unravel_index(
        np.argmax(blue_mat[:100, :100], axis=None), [100, 100])
    b_row_diff, b_col_diff = -b_row_diff, -b_col_diff

    im_fft_red = np.fft.fft2(cut_img_red)
    red_mat = np.fft.ifft2(im_fft_green * np.conj(im_fft_red))
    r_row_diff, r_col_diff = np.unravel_index(
        np.argmax(red_mat[:100, :100], axis=None), [100, 100])

    aligned_img = np.dstack((cut_img_red, cut_img_green, cut_img_blue))
    b_row = g_coord[0] - b_row_diff - border
    b_col = g_coord[1] - b_col_diff

    r_row = g_coord[0] - r_row_diff - border + 2 * border + 2
    r_col = g_coord[1] - r_col_diff

    return (aligned_img * 255).astype('uint8'), (b_row, b_col), (r_row, r_col)
