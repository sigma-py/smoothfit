# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy

import meshzoo

import smoothfit


numpy.random.seed(123)


def test_1d():
    n = 50
    x0 = numpy.random.rand(n) * 2 - 1
    # y0 = 1 / (1 + 25*x0**2) + 5.0e-2 * (2*numpy.random.rand(n)-1)
    # y0 = numpy.exp(x0) + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    # y0 = x0 + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    # y0 = x0**3 + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    y0 = numpy.sin(1*numpy.pi*x0)  # + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    # y0 = 1 / (x0 + 2) + 1.0e-2 * (2*numpy.random.rand(n) - 1)

    a = -1.5
    b = +1.5

    x, vals = smoothfit.fit1d(x0, y0, a, b, 50, eps=1.0e-1)

    plt.plot(x0, y0, 'xk', label='data')
    plt.plot(
        x, vals, '-',
        color='k', alpha=0.3,
        label='smooth fit'
        )
    plt.xlim(a, b)
    plt.legend()
    plt.show()
    return


def test_2d():
    n = 50
    x0 = numpy.random.rand(n, 2)
    y0 = numpy.sin(numpy.pi*x0.T[0]) * numpy.sin(numpy.pi*x0.T[1])

    points, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 20, 20)

    u = smoothfit.fit(x0, y0, points, cells, eps=1.0e-1, verbose=True)

    from dolfin import XDMFFile
    xdmf = XDMFFile('temp.xdmf')
    xdmf.write(u)
    return


if __name__ == '__main__':
    test_2d()
