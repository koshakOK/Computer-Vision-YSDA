import numpy as np
from skimage.color import rgb2gray


def compute_grad(brightness):
    grad_x = np.zeros((brightness.shape[0], brightness.shape[1]))
    grad_x[:, 1:-1] = brightness[:, 2:] - brightness[:, :-2]
    grad_x[:, 0] = brightness[:, 1] - brightness[:, 0]
    grad_x[:, -1] = brightness[:, -1] - brightness[:, -2]

    grad_y = np.zeros((brightness.shape[0], brightness.shape[1]))
    grad_y[1:-1, :] = brightness[2:, :] - brightness[:-2, :]
    grad_y[0, :] = brightness[1, :] - brightness[0, :]
    grad_y[-1, :] = brightness[-1, :] - brightness[-2, :]

    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def seam_carve(image, method, mask=None):
    new_image = image.copy().astype(np.float64)

    brightness = np.zeros(
        (new_image.shape[0], new_image.shape[1])).astype(np.float64)
    for i in range(brightness.shape[0]):
        for j in range(brightness.shape[1]):
            brightness[i][j] = 0.299 * new_image[i][j][0] + 0.587 * \
                new_image[i][j][1] + 0.114 * new_image[i][j][2]
    edge = compute_grad(brightness)
    if mask is not None:
        max_energy_coef = 256. * edge.shape[0] * edge.shape[1]
        for i in range(edge.shape[0]):
            for j in range(edge.shape[1]):
                if mask[i][j] == -1:
                    edge[i][j] -= max_energy_coef
                elif mask[i][j] == 1:
                    edge[i][j] += max_energy_coef

    edge_cum = edge.copy()
    ans = np.zeros_like(edge_cum)
    if method == "horizontal shrink" or method == "horizontal expand":
        for i in range(1, edge.shape[0]):
            for j in range(edge.shape[1]):
                if j == 0:
                    edge_cum[i][j] = edge_cum[i][j] + \
                        min(edge_cum[i-1][j], edge_cum[i-1][j+1])
                elif j == edge.shape[1] - 1:
                    edge_cum[i][j] = edge_cum[i][j] + \
                        min(edge_cum[i-1][j-1], edge_cum[i-1][j])
                else:
                    edge_cum[i][j] = edge_cum[i][j] + \
                        min(edge_cum[i-1][j-1], edge_cum[i-1]
                            [j], edge_cum[i-1][j+1])
        ans = np.zeros(edge_cum.shape)
        min_el_col = np.argmin(edge_cum[-1, :])
        ans[-1][min_el_col] = 1

        for i in range(edge_cum.shape[0] - 1, -1, -1):
            if min_el_col == 0:
                if edge_cum[i][min_el_col] <= edge_cum[i][min_el_col + 1]:
                    ans[i][min_el_col] = 1
                else:
                    min_el_col = min_el_col + 1
                    ans[i][min_el_col] = 1
            elif min_el_col == edge_cum.shape[1] - 1:
                if edge_cum[i][min_el_col - 1] <= edge_cum[i][min_el_col]:
                    min_el_col = min_el_col - 1
                    ans[i][min_el_col] = 1
                else:
                    ans[i][min_el_col] = 1
            else:
                a = edge_cum[i][min_el_col - 1]
                b = edge_cum[i][min_el_col]
                c = edge_cum[i][min_el_col + 1]
                if a <= b and a <= c:
                    min_el_col = min_el_col - 1
                    ans[i][min_el_col] = 1
                elif b <= c:
                    ans[i][min_el_col] = 1
                else:
                    min_el_col = min_el_col + 1
                    ans[i][min_el_col] = 1

    if method == "vertical shrink" or method == "vertical expand":
        for j in range(1, edge.shape[1]):
            for i in range(edge.shape[0]):
                if i == 0:
                    edge_cum[i][j] = edge_cum[i][j] + \
                        min(edge_cum[i][j-1], edge_cum[i+1][j-1])
                elif i == edge.shape[0] - 1:
                    edge_cum[i][j] = edge_cum[i][j] + \
                        min(edge_cum[i-1][j-1], edge_cum[i][j-1])
                else:
                    edge_cum[i][j] = edge_cum[i][j] + \
                        min(edge_cum[i-1][j-1], edge_cum[i]
                            [j-1], edge_cum[i+1][j-1])
        ans = np.zeros(edge_cum.shape)
        min_el_raw = np.argmin(edge_cum[:, -1])
        ans[min_el_raw][-1] = 1

        for j in range(edge_cum.shape[1] - 1, -1, -1):
            if min_el_raw == 0:
                if edge_cum[min_el_raw][j] <= edge_cum[min_el_raw + 1][j]:
                    ans[min_el_raw][j] = 1
                else:
                    min_el_raw = min_el_raw + 1
                    ans[min_el_raw][j] = 1
            elif min_el_raw == edge_cum.shape[0] - 1:
                if edge_cum[min_el_raw - 1][j] <= edge_cum[min_el_raw][j]:
                    min_el_raw = min_el_raw - 1
                    ans[min_el_raw][j] = 1
                else:
                    ans[min_el_raw][j] = 1
            else:
                a = edge_cum[min_el_raw - 1][j]
                b = edge_cum[min_el_raw][j]
                c = edge_cum[min_el_raw + 1][j]
                if a <= b and a <= c:
                    min_el_raw = min_el_raw - 1
                    ans[min_el_raw][j] = 1
                elif b <= c:
                    ans[min_el_raw][j] = 1
                else:
                    min_el_raw = min_el_raw + 1
                    ans[min_el_raw][j] = 1
    return (True, True, ans)
