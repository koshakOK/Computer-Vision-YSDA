import numpy as np
import scipy.fftpack as sff


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    kernel = np.zeros((size, size))
    x0 = int(size/2)
    y0 = int(size/2)
    for i in range(size):
        for j in range(size):
            r_2 = (i - x0)**2 + (j - y0)**2
            kernel[i][j] = 1. / (2 * np.pi * sigma ** 2) * \
                np.exp(-r_2/(2*sigma**2))
    return kernel / kernel.sum()


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h_new = np.zeros((shape[0], shape[1]))
    h_new[:h.shape[0], :h.shape[1]] = h
    return sff.fft2(h_new)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros((H.shape[0], H.shape[1]), dtype='complex_')
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.abs(H[i][j]) > threshold:
                H_inv[i][j] = 1. / H[i][j]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    F = sff.fft2(blurred_img) * inverse_kernel(fourier_transform(h,
                                                                 (blurred_img.shape[0], blurred_img.shape[1])), threshold)
    f = sff.ifft2(F)
    return np.abs(f)


def wiener_filtering(blurred_img, h, K=0.0001):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, (blurred_img.shape[0], blurred_img.shape[1]))
    H_conj = np.conj(H)
    G = sff.fft2(blurred_img)
    F = H_conj / (H_conj * H + K) * G
    f = sff.ifft2(F)
    return np.abs(f)


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    MAX1 = 255
    MSE = np.mean((img2 - img1) ** 2)
    return 20 * np.log10(MAX1 / np.sqrt(MSE))
