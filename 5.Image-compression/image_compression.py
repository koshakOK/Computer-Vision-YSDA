import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """

    # Your code here
    M = matrix.copy()
    # Отцентруем каждую строчку матрицы
    diff = np.mean(matrix, axis=1)[:, None]
    M = matrix - diff
    # Найдем матрицу ковариации
    C = np.cov(M)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(C)
    # Посчитаем количество найденных собственных векторов
    n = eig_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    sorted_eig_val = np.argsort(eig_val)[::-1]
    # sorted_eig_vec = np.array([eig_vec[:, i] for i in sorted_eig_val])

    sorted_eig_vec = np.zeros_like(eig_vec)
    for index, raw in enumerate(sorted_eig_val):
        sorted_eig_vec[:, index] = eig_vec[:, raw]

    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    # Оставляем только p собственных векторов
    sorted_eig_vec = sorted_eig_vec[:, :p]

    # Проекция данных на новое пространство
    result = np.dot(sorted_eig_vec.T, M)

    return sorted_eig_vec, result, diff.T


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        res = np.dot(comp[0], comp[1]) + comp[2].T
        res = np.clip(res, 0, 255)
        result_img.append(res)
    result_img = np.array(result_img)
    return result_img.transpose((1, 2, 0))


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            eig_vec, result, diff = pca_compression(img[:, :, j], p)
            compressed.append((eig_vec, result, diff))
        axes[i // 3, i % 3].imshow(pca_decompression(compressed) / 255)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    A = np.array([[0.299, 0.587, 0.114],
                  [-0.1687, -0.3313, 0.5],
                  [0.5, -0.4187, -0.0813]])
    YCbCr = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R = np.array(img[i, j, 0])
            G = np.array(img[i, j, 1])
            B = np.array(img[i, j, 2])
            ycbcr = np.array([0, 128, 128]).T + \
                np.dot(A, np.array([R, G, B]).T)
            YCbCr[i, j, :] = ycbcr
    return YCbCr


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    # img[:, :, 1] -= 128
    # img[:, :, 2] -= 128
    A = np.array([[1, 0, 1.402],
                  [1, -0.34414, -0.71414],
                  [1, 1.77, 0]])
    RGB = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Y = np.array(img[i, j, 0])
            Cb = np.array(img[i, j, 1]) - 128
            Cr = np.array(img[i, j, 2]) - 128
            rgb = np.dot(A, np.array([Y, Cb, Cr]).T)
            RGB[i, j, :] = rgb
    return RGB


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    a = np.array([ycbcr_img[:, :, 1], ycbcr_img[:, :, 2]])

    b = gaussian_filter(a, sigma=1)
    ycbcr_img[:, :, 1] = b[0]
    ycbcr_img[:, :, 2] = b[1]

    rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(rgb_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0], sigma=1)
    rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(rgb_img)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    component = gaussian_filter(component, sigma=10)
    n = (component.shape[0] + 1) // 2
    m = (component.shape[1] + 1) // 2
    two_times_downsampled = []
    for i in range(component.shape[0]):
        for j in range(component.shape[1]):
            if i % 2 == 0 and j % 2 == 0:
                two_times_downsampled.append(component[i][j])
    return np.array(two_times_downsampled).reshape((n, m))


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    dc_block = np.zeros_like(block).astype(float)
    for u in range(block.shape[0]):
        for v in range(block.shape[1]):
            alpha_u = 1 / np.sqrt(2) if u == 0 else 1
            alpha_v = 1 / np.sqrt(2) if v == 0 else 1
            g_sum = 0
            for x in range(8):
                for y in range(8):
                    g_x = ((2 * x + 1) * u * np.pi) / 16
                    g_y = ((2 * y + 1) * v * np.pi) / 16
                    g_sum += block[x][y] * np.cos(g_x) * np.cos(g_y)

            dc_block[u][v] = 1 / 4 * alpha_u * alpha_v * g_sum

    return dc_block


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    quantization_block = np.round(block / quantization_matrix)

    return quantization_block


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    s = 5000 / q
    if q == 100:
        s = 1
    elif q >= 50 and q <= 99:
        s = 200 - 2 * q

    d = ((s * default_quantization_matrix + 50) / 100).astype(int)

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if not d[i][j]:
                d[i][j] = 1
    return d


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    zig_zag = [block[0, 0]]

    def right(i, j):
        j += 1
        zig_zag.append(block[i, j])
        return (i, j)

    def down(i, j):
        i += 1
        zig_zag.append(block[i, j])
        return (i, j)

    def right_up(i, j):
        i -= 1
        j += 1
        zig_zag.append(block[i, j])
        return (i, j)

    def left_down(i, j):
        i += 1
        j -= 1
        zig_zag.append(block[i, j])
        return (i, j)

    i = 0
    j = 0
    i, j = right(i, j)
    i, j = left_down(i, j)
    i, j = down(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)

    i, j = right(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = down(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)

    i, j = right(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = down(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)

    i, j = right(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)

    # reverse
    i, j = right(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)

    i, j = right(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)
    i, j = left_down(i, j)

    i, j = right(i, j)
    i, j = right_up(i, j)
    i, j = right_up(i, j)
    i, j = down(i, j)
    i, j = left_down(i, j)

    i, j = right(i, j)

    return zig_zag


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    zig_zag = []
    i = 0
    while True:
        if i >= len(zigzag_list):
            break
        element = zigzag_list[i]
        if element:
            zig_zag.append(element)
            i += 1
        else:
            zig_zag.append(element)
            zero_count = 1
            i += 1
            while True:
                if i >= len(zigzag_list):
                    zig_zag.append(zero_count)
                    break
                if zigzag_list[i]:
                    zig_zag.append(zero_count)
                    break
                else:
                    zero_count += 1
                    i += 1
    return zig_zag


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Переходим из RGB в YCbCr
    ycbcr = ycbcr2rgb(img)
    # Уменьшаем цветовые компоненты
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    cb = downsampling(cb)
    cr = downsampling(cr)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    ys = []
    cbs = []
    crs = []

    n = y.shape[0]
    m = y.shape[1]
    assert n == m and n % 8 == 0
    for i in range(0, n, 8):
        block_y = y[i:i+8, i:i+8]
        block_cbs = cb[i:i+8, i:i+8]
        block_crs = cr[i:i+8, i:i+8]
        ys.append(block_y - 128)
        cbs.append(block_cbs - 128)
        crs.append(block_crs - 128)
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    for i in range(len(ys)):
        ys[i] = compression(
            zigzag(quantization(dct(ys[i]), quantization_matrixes[0])))
        cbs[i] = compression(
            zigzag(quantization(dct(cbs[i]), quantization_matrixes[1])))
        crs[i] = compression(
            zigzag(quantization(dct(crs[i]), quantization_matrixes[1])))
    return [ys, cbs, crs]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    zig_zag = []
    i = 0
    while True:
        if i >= len(compressed_list):
            break
        el = compressed_list[i]
        if el:
            zig_zag.append(el)
            i += 1
        else:
            count = compressed_list[i + 1]
            i += 2
            for _ in range(count):
                zig_zag.append(0)

    return zig_zag


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    zig_zag = np.zeros((8, 8))
    zig_zag[0][0] = input[0]

    def right(i, j, index):
        j += 1
        zig_zag[i, j] = input[index]
        index += 1
        return (i, j, index)

    def down(i, j, index):
        i += 1
        zig_zag[i, j] = input[index]
        index += 1
        return (i, j, index)

    def right_up(i, j, index):
        i -= 1
        j += 1
        zig_zag[i, j] = input[index]
        index += 1
        return (i, j, index)

    def left_down(i, j, index):
        i += 1
        j -= 1
        zig_zag[i, j] = input[index]
        index += 1
        return (i, j, index)

    i = 0
    j = 0
    index = 1
    i, j, index = right(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = down(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)

    i, j, index = right(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = down(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)

    i, j, index = right(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = down(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)

    i, j, index = right(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)

    # reverse
    i, j, index = right(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)

    i, j, index = right(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)
    i, j, index = left_down(i, j, index)

    i, j, index = right(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = right_up(i, j, index)
    i, j, index = down(i, j, index)
    i, j, index = left_down(i, j, index)

    i, j, index = right(i, j, index)

    return zig_zag


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return quantization_matrix * block


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    dc_block = np.zeros_like(block).astype(float)
    for x in range(block.shape[0]):
        for y in range(block.shape[1]):
            g_sum = 0
            for u in range(8):
                for v in range(8):
                    alpha_u = 1 / np.sqrt(2) if u == 0 else 1
                    alpha_v = 1 / np.sqrt(2) if v == 0 else 1
                    g_x = ((2 * x + 1) * u * np.pi) / 16
                    g_y = ((2 * y + 1) * v * np.pi) / 16
                    g_sum += alpha_u * alpha_v * \
                        block[u][v] * np.cos(g_x) * np.cos(g_y)

            dc_block[x][y] = 1 / 4 * g_sum

    return np.round(dc_block)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    n = component.shape[0]
    m = component.shape[1] * 2
    two_times_upsampled = []
    for i in range(component.shape[0]):
        for j in range(component.shape[1]):
            two_times_upsampled.append(component[i][j])
            two_times_upsampled.append(component[i][j])
    two_times_upsampled = np.array(two_times_upsampled).reshape((n, m))
    return np.array([x for pair in zip(two_times_upsampled, two_times_upsampled) for x in pair])


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Переходим из RGB в YCbCr
    ycbcr = ycbcr2rgb(img)
    # Уменьшаем цветовые компоненты
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    cb = downsampling(cb)
    cr = downsampling(cr)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    ys = []
    cbs = []
    crs = []

    n = y.shape[0]
    m = y.shape[1]
    assert n == m and n % 8 == 0
    for i in range(0, n, 8):
        for j in range(0, m, 8):
            block_y = y[i:i+8, j:j+8] - 128
            # block_y = np.clip(block_y, -128, 127)
            ys.append(block_y)

    for i in range(0, int(n / 2), 8):
        for j in range(0, int(m / 2), 8):
            block_cbs = cb[i:i+8, j:j+8] - 128
            block_crs = cr[i:i+8, j:j+8] - 128
            # block_cbs = np.clip(block_cbs, -128, 127)
            # block_crs = np.clip(block_crs, -128, 127)
            cbs.append(block_cbs)
            crs.append(block_crs)

    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    for i in range(len(ys)):
        ys[i] = compression(
            zigzag(quantization(dct(ys[i]), quantization_matrixes[0])))

    for i in range(len(cbs)):
        cbs[i] = compression(
            zigzag(quantization(dct(cbs[i]), quantization_matrixes[1])))
        crs[i] = compression(
            zigzag(quantization(dct(crs[i]), quantization_matrixes[1])))
    ys = np.array(ys)
    cbs = np.array(cbs)
    crs = np.array(crs)
    return [ys, cbs, crs]


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    ys_jpeg = result[0]
    cbs_jpeg = result[1]
    crs_jpeg = result[2]

    ys = []
    cbs = []
    crs = []
    for i in range(ys_jpeg.shape[0]):
        ys.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(ys_jpeg[i])),
                                                   quantization_matrixes[0])))
    for i in range(cbs_jpeg.shape[0]):
        cbs.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(cbs_jpeg[i])),
                                                    quantization_matrixes[1])))
        crs.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(crs_jpeg[i])),
                                                    quantization_matrixes[1])))
    ys = np.array(ys)
    cbs = np.array(cbs)
    crs = np.array(crs)

    y = np.zeros((256, 256))
    cb = np.zeros((128, 128))
    cr = np.zeros((128, 128))
    i = 0
    j = 0
    for index in range(ys.shape[0]):
        y[i:i+8, j:j+8] = ys[index] + 128
        # y[i:i+8, j:j+8] = np.clip(y[i:i+8, j:j+8], 0, 255)
        j += 8
        if j == 256:
            j = 0
            i += 8
    i = 0
    j = 0
    for index in range(cb.shape[0]):
        cb[i:i+8, j:j+8] = cbs[index] + 128
        cr[i:i+8, j:j+8] = crs[index] + 128

        # cb[i:i+8, j:j+8] = np.clip(cb[i:i+8, j:j+8], 0, 255)
        # cr[i:i+8, j:j+8] = np.clip(cr[i:i+8, j:j+8], 0, 255)
        j += 8
        if j == 128:
            j = 0
            i += 8

    cb = upsampling(cb)
    cr = upsampling(cr)

    cb[128:, :] += 256
    cr[128:, :] += 256
    # y[128:, :] += 128
    y = np.clip(y, 0, 255)
    cb = np.clip(cb, 0, 255)
    cr = np.clip(cr, 0, 255)

    YCbCr = np.zeros((256, 256, 3))
    YCbCr[:, :, 0] = y
    YCbCr[:, :, 1] = cb
    YCbCr[:, :, 2] = cr
    RGB = np.round(ycbcr2rgb(YCbCr))
    # RGB[128:, :, 0] += 256
    # RGB[128:, :, 1] -= 128
    # RGB[128:, :, 2] += 256
    RGB[:, :, 0] = np.clip(RGB[:, :, 0], 0, 255)
    RGB[:, :, 1] = np.clip(RGB[:, :, 1], 0, 255)
    RGB[:, :, 2] = np.clip(RGB[:, :, 2], 0, 255)
    print(RGB[:128, :, 0])
    print("))))))))))")
    print(RGB[128:, :, 0])
    print("))))))))))")
    print(RGB[:128, :, 1])
    print("))))))))))")
    print(RGB[128:, :, 1])
    print("))))))))))")
    print(RGB[:128, :, 2])
    print("))))))))))")
    print(RGB[128:, :, 2])
    return RGB


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([100]):
        y_m = own_quantization_matrix(y_quantization_matrix, p)
        c_m = own_quantization_matrix(color_quantization_matrix, p)
        quantization_matrixes = [y_m, c_m]

        result = jpeg_compression(img, quantization_matrixes)

        rgb = jpeg_decompression(result, img.shape, quantization_matrixes)

        axes[i // 3, i % 3].imshow(rgb / 255.)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(
            color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append(
                (pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(
            img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [
                       1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == '__main__':
    # pca_visualize()
    # get_gauss_1()
    # get_gauss_2()
    # jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
