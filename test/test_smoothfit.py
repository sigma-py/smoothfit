# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy

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

    u = smoothfit.fit1d(x0, y0, a, b, 50, eps=1.0e-1)

    x = u.function_space().mesh().coordinates()

    plt.plot(x0, y0, 'xk', label='data')
    plt.plot(
        x, [u(xx) for xx in x], '-',
        color='k', alpha=0.3,
        label='smooth fit'
        )
    plt.xlim(a, b)
    plt.legend()
    plt.show()
    return


def test_2d():
    n = 200
    x0 = numpy.random.rand(n, 2) - 0.5
    # y0 = numpy.ones(n)
    # y0 = x0[:, 0]
    y0 = x0[:, 0]**2
    # y0 = numpy.cos(numpy.pi*x0.T[0])
    # y0 = numpy.cos(numpy.pi*x0.T[0]) * numpy.cos(numpy.pi*x0.T[1])
    # y0 = numpy.cos(numpy.pi*numpy.sqrt(x0.T[0]**2 + x0.T[1]**2))

    import meshzoo
    points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, 10, 10)

    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_circle([0.0, 0.0, 0.0], 1.0, 0.1)
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    u = smoothfit.fit(x0, y0, points, cells, eps=1.0e-0, verbose=True)

    from dolfin import XDMFFile
    xdmf = XDMFFile('temp.xdmf')
    xdmf.write(u)
    return


if __name__ == '__main__':
    test_1d()
