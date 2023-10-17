from interface import *
import torch
import scipy.signal
# ================================= 1.4.1 SGD ================================


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """

            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter + parameter * self.momentum - self.lr * parameter_grad
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        mask = inputs >= 0
        return inputs * mask

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        mask = self.forward_inputs >= 0
        return grad_outputs * mask


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # start with normalization for numerical stability
        n = inputs.shape[0]  # batches
        d = inputs.shape[1]  # features

        x = np.subtract(inputs, inputs.max(axis=1, keepdims=True))
        output = np.zeros(shape=(n, d))
        for i in np.arange(n):
            output[i] = np.exp(x[i])/np.sum(np.exp(x[i]))
        return output

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        n = self.forward_inputs.shape[0]  # batches
        d = self.forward_inputs.shape[1]  # features

        x = np.subtract(self.forward_inputs,
                        self.forward_inputs.max(axis=1, keepdims=True))
        df = np.subtract(np.einsum('ij,jk->ijk', self.forward_impl(x), np.eye(d, d)),
                         np.einsum('ij,ik->ijk', self.forward_impl(x), self.forward_impl(x)))
        return np.einsum('ijk,ik->ij', df, grad_outputs)


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """

        return inputs @ self.weights + self.biases

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        ans = grad_outputs @ self.weights.T
        self.weights_grad = (
            grad_outputs.T @ self.forward_inputs).T / ans.shape[0]
        self.biases_grad = np.ones(
            shape=(self.forward_inputs.shape[0])).T @ grad_outputs / ans.shape[0]
        return ans


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n,)), loss scalars for batch

                n - batch size
                d - number of units
        """
        return -np.sum(y_gt*np.log(y_pred), axis=1)

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), gradient loss to y_pred

                n - batch size
                d - number of units
        """
        y = np.copy(y_pred)
        mask = y <= 1e-20
        eps = 2.22043451e-16
        y[mask] += eps
        return - y_gt / y


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    #     # your code here \/
    #     # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=1e-2)
    model = Model(loss=loss, optimizer=optimizer)
    model.add(Dense(input_shape=(784,), units=128))
    model.add(ReLU())
    model.add(Dense(units=128))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    model.fit(x_train, y_train, 128, 5)

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    def pad3D(c_x, padlen=1):
        m, n, r = c_x.shape
        c_y = np.zeros((m, n+2*padlen, r+2*padlen), dtype=c_x.dtype)
        c_y[:, padlen:-padlen, padlen:-padlen] = c_x
        return c_y

    n = inputs.shape[0]
    c = kernels.shape[0]
    kernels = kernels[:, :, ::-1, ::-1]
    n_ans = []
    for n_i in range(n):
        c_ans = []
        for c_i in range(c):
            inputs_n_i = inputs[n_i, :, :, :]
            kernel_c_i = kernels[c_i, :, :, :]
            if padding >= 1:
                inputs_n_i = pad3D(inputs_n_i, padlen=padding)
            conv = scipy.signal.correlate(inputs_n_i.transpose(
                1, 2, 0), kernel_c_i.transpose(1, 2, 0), mode='valid').transpose(2, 0, 1)
            c_ans.append(conv)
        n_ans.append(c_ans)
    n_ans = np.array(n_ans)
    mask = np.abs(n_ans) < 10e-8
    n_ans[mask] = 0
    return np.squeeze(n_ans, axis=2)


# =============================== 4.1.1 Conv2D ===============================

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   'constant', constant_values=(0, 0))

    return X_pad


class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        p = int((self.kernel_size + 1) / 2) - 1
        self.p = p
        ans = convolve_pytorch(inputs, self.kernels, padding=p)
        c = ans.shape[1]
        ans += self.biases[None, :, None, None]
        return ans

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        A = self.forward_inputs.copy()
        A = np.flip(A, 3)
        A = np.flip(A, 2)

        # print(A.shape)
        # print(grad_outputs.shape)

        self.kernels_grad = convolve(A.swapaxes(0, 1), grad_outputs.swapaxes(
            0, 1), padding=self.kernels.shape[2]//2).swapaxes(0, 1) / self.forward_inputs.shape[0]

        self.biases_grad = np.ndarray((grad_outputs.shape[1]))
        for i in range(grad_outputs.shape[1]):
            self.biases_grad[i] = np.sum(
                grad_outputs[:, i, :, :]) / self.forward_inputs.shape[0]

        return convolve(grad_outputs, np.flip(np.flip(self.kernels.swapaxes(0, 1), 2), 3), padding=self.kernels.shape[2]//2)


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        S = self.pool_size
        self.inputs = inputs
        ans = np.ndarray((n, d, ih//self.pool_size, iw//self.pool_size))
        self.save = np.ndarray((n, d, ih//self.pool_size, iw//self.pool_size))

        for i in range(n):
            for j in range(d):
                for k in range(ih//S):
                    for l in range(iw//S):
                        if self.pool_mode == 'max':
                            ans[i][j][k][l] = np.max(
                                inputs[i, j, S*k:S*(k+1), S*l:S*(l+1)])
                            T = np.argmax(
                                inputs[i, j, S*k:S*(k+1), S*l:S*(l+1)])
                          #  print(k, l, inputs[i,j,S*k:S*(k+1),S*l:S*(l+1)], T, np.max(inputs[i,j,S*k:S*(k+1),S*l:S*(l+1)]))
                            self.save[i, j, k, l] = T
                        else:
                            ans[i][j][k][l] = np.average(
                                inputs[i, j, S*k:S*(k+1), S*l:S*(l+1)])
        return ans
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        S = self.pool_size
        n = grad_outputs.shape[0]
        d, ih, iw = self.input_shape
        ans = np.zeros((n, d, ih, iw))

       # print(self.inputs)

        for i in range(n):
            for j in range(d):
                for k in range(ih):
                    for l in range(iw):
                        if self.pool_mode == 'avg':
                            ans[i][j][k][l] = grad_outputs[i,
                                                           j, k//S, l//S] / (S*S)
                        else:
                            Q = self.save[i, j, k//S, l//S]
                            dc = Q//S
                            dd = Q % S

                            if (k//S)*S + dc == k and (l//S)*S + dd == l:
                                ans[i][j][k][l] = grad_outputs[i, j, k//S, l//S]

        return ans.astype('float64')
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        d = inputs.shape[1]
        self.input = inputs

        if self.is_training:

            self.mu = np.zeros(d)
            self.sigma2 = np.zeros(d)

            for i in range(d):
                self.mu[i] = np.mean(inputs[:, i, :, :])
                self.sigma2[i] = np.var(inputs[:, i, :, :])

            if self.running_mean is not None:
                self.running_mean = self.running_mean * \
                    self.momentum + self.mu * (1 - self.momentum)
            else:
                self.running_mean = self.mu

            if self.running_var is not None:
                self.running_var = self.running_var * \
                    self.momentum + self.sigma2 * (1 - self.momentum)
            else:
                self.running_var = self.sigma2
        else:
            self.mu = self.running_mean
            self.sigma2 = self.running_var

        self.output = np.zeros_like(inputs)
        for i in range(d):
            self.output[:, i, :, :] = (
                inputs[:, i, :, :] - self.mu[i]) / np.sqrt(self.sigma2[i] + eps)
        # print(self.output.shape, self.gamma.shape)

        for i in range(inputs.shape[1]):
            self.output[:, i, :, :] *= self.gamma[i]
            self.output[:, i, :, :] += self.beta[i]
        # your code here /\

        return self.output

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        d = self.input.shape[1]
        m = self.input.shape[0] * self.input.shape[2] * self.input.shape[3]
        self.grad_input = np.zeros_like(grad_outputs)

        if self.is_training:

            self.mu = np.zeros(d)
            self.sigma2 = np.zeros(d)

            for i in range(d):
                self.mu[i] = np.mean(self.input[:, i, :, :])
                self.sigma2[i] = np.var(self.input[:, i, :, :])
        else:
            self.mu = self.running_mean
            self.sigma2 = self.running_var

        for i in range(d):

            d_sigma = -(grad_outputs[:, i, :, :] * (self.input[:, i, :, :] -
                        self.mu[i])).sum() * ((self.sigma2[i] + eps)**(-3/2)) / 2
            d_mean = (-grad_outputs[:, i, :, :] / np.sqrt(self.sigma2[i] + eps)).sum(
            ) + (-2 * (self.input[:, i, :, :] - self.mu[i])).sum() * d_sigma / m
            self.grad_input[:, i, :, :] = grad_outputs[:, i, :, :] / np.sqrt(
                self.sigma2[i] + eps) + 2 * (self.input[:, i, :, :] - self.mu[i]) * d_sigma / m + d_mean / m
            self.grad_input[:, i, :, :] *= self.gamma[i]
        # your code here /\

        self.beta_grad = np.zeros((d))
        self.gamma_grad = np.zeros((d))

        # print(self.gamma.shape)

        for i in range(d):
            self.beta_grad[i] = np.average(
                grad_outputs[:, i, :, :]) * grad_outputs.shape[2] * grad_outputs.shape[3]
            self.gamma_grad[i] = np.average(np.multiply(grad_outputs[:, i, :, :], (
                self.input[:, i, :, :] - self.mu[i]) / np.sqrt(self.sigma2[i]))) * grad_outputs.shape[2] * grad_outputs.shape[3]

        return self.grad_input
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        self.inputs = inputs
        return np.reshape(inputs, (inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        return np.reshape(grad_outputs, (self.inputs.shape[0], self.inputs.shape[1], self.inputs.shape[2], self.inputs.shape[3]))
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        if self.is_training:
            q = np.random.uniform(size=inputs.shape)
            q[q <= self.p] = 0
            q[q > self.p] = 1

            self.forward_mask = q
            self.output = inputs * self.forward_mask
        else:
            self.output = inputs
        return self.output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        if self.is_training:
            self.grad_input = grad_outputs * self.forward_mask
        else:
            self.grad_input = grad_outputs

        return self.grad_input
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.01))
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(5, input_shape=(3, 32, 32)))
    model.add(ReLU())
    model.add(Pooling2D())
    model.add(Conv2D(10, input_shape=(3, 32, 32)))
    model.add(ReLU())
    model.add(Pooling2D())
    model.add(Flatten(input_shape=(3, 32, 32)))
    model.add(Dense(128))
#     model.add(ReLU())
#     model.add(Dense(128))
#     model.add(ReLU())
#     model.add(Dense(100))
#     model.add(ReLU())
#     model.add(Dense(50))
#     model.add(ReLU())
#     model.add(Dense(28))
#     model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 100, 1)
    model.fit(x_valid, y_val, 100, 1)
#
    # your code here /\
    return model

# ============================================================================
