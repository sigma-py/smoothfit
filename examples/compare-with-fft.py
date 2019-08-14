import matplotlib.pyplot as plt
import numpy

numpy.random.seed(0)

x0 = numpy.linspace(-1.0, 1.0, 1000)
y0 = 1 / (1 + 25 * x0 ** 2)

n = 51
x1 = numpy.linspace(-1.0, 1.0, n)
y1 = 1 / (1 + 25 * x1 ** 2) + 1.0e-1 * (2 * numpy.random.rand(x1.shape[0]) - 1)

plt.plot(x0, y0, color="k", alpha=0.2)
plt.plot(x1, y1, "xk")
plt.grid()

# One could also use the proper transforms
# <https://gist.github.com/nschloe/3b34c21ed3cc8c8c6a77f4b7e4167f8b> here, but let's
# keep things simple.
length = x1[-1] - x1[0]
X = numpy.fft.rfft(y1)
# cut off the high frequencies
X[5:] = 0.0
y2 = numpy.fft.irfft(X, n)

plt.plot(x1, y2, "-", label="5 lowest frequencies")
plt.show()
