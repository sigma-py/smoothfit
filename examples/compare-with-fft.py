import numpy
import matplotlib.pyplot as plt


def uniform_dft(interval_length, data):
    '''Discrete Fourier Transform of real-valued data, interpreted for a uniform series
    over an interval of a given length, including starting and end point.
    The function returns the frequencies in units of the coordinate.
    '''
    X = numpy.fft.rfft(data)
    n = len(data)
    # The input data is assumed to cover the entire time interval, i.e., including start
    # and end point. The data produced from RFFT however assumes that the end point is
    # excluded. Hence, stretch the interval_length such that cutting off the end point
    # results in the interval of length interval_length.
    interval_length *= n / float(n - 1)
    # Note that the following definition of the frequencies slightly differs from the
    # output of np.fft.freqs which is
    #
    #     freqs = [i / interval_length / n for i in range(n//2 + 1)].
    freqs = numpy.arange(n // 2 + 1) / interval_length
    # Also note that the angular frequency is  omega = 2*pi*freqs.
    #
    # With RFFT, the amplitudes need to be scaled by a factor of 2.
    X /= n
    X[1:-1] *= 2
    if n % 2 != 0:
        X[-1] *= 2
    assert len(freqs) == len(X)
    return freqs, X


def uniform_idft(X, n):
    # Equivalent:
    # n = 2 * len(freqs) - 1
    # data = numpy.zeros(n)
    # t = numpy.linspace(0.0, len_interval, n)
    # for x, freq in zip(X, freqs):
    #     alpha = x * numpy.exp(1j * 2 * numpy.pi * freq * t)
    #     data += alpha.real

    # transform back
    X[1:-1] /= 2
    if n % 2 != 0:
        X[-1] /= 2
    X *= n
    data = numpy.fft.irfft(X, n)
    return data


numpy.random.seed(0)

x0 = numpy.linspace(-1.0, 1.0, 1000)
y0 = 1 / (1 + 25 * x0**2)

n = 51
x1 = numpy.linspace(-1.0, 1.0, n)
y1 = 1 / (1 + 25 * x1**2) + 1.0e-1 * (2 * numpy.random.rand(x1.shape[0]) - 1)

plt.plot(x0, y0, color="k", alpha=0.2)
plt.plot(x1, y1, "xk")
plt.grid()

# We could also use numpy.fft.[i]rfft directly here, but the above functions are a bit
# more instructive.
length = x1[-1] - x1[0]
_, X = uniform_dft(length, y1)
# cut off the high frequencies
X[5:] = 0.0
y2 = uniform_idft(X, n)

plt.plot(x1, y2, "-")
plt.show()
