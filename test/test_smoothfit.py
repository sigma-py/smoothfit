# -*- coding: utf-8 -*-
#
from __future__ import division

from dolfin import assemble, dx
import matplotlib.pyplot as plt
import numpy
import pytest

import smoothfit


def test_1d_show():
    n = 20
    # x0 = numpy.linspace(-1.0, 1.0, n)
    numpy.random.seed(123)
    x0 = numpy.random.rand(n) * 2 - 1

    # y0 = x0.copy()
    # y0 = 1 / (1 + 25*x0**2) + 5.0e-2 * (2*numpy.random.rand(n)-1)
    # y0 = numpy.exp(x0) + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    # y0 = x0 + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    # y0 = x0**3 + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    y0 = numpy.sin(1*numpy.pi*x0)  # + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    # y0 = 1 / (x0 + 2) + 1.0e-2 * (2*numpy.random.rand(n) - 1)

    a = -1.5
    b = +1.5

    u = smoothfit.fit1d(x0, y0, a, b, 10, degree=2, eps=1.0e-2)

    ref = 1.5955290123404824
    val = assemble(u*u * dx)
    assert abs(val - ref) < 1.0e-10 * ref

    # x = u.function_space().mesh().coordinates()
    x = numpy.linspace(a, b, 201)
    vals = [u(xx) for xx in x]

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


def test_1d_scale():
    n = 20
    # x0 = numpy.linspace(-1.0, 1.0, n)
    numpy.random.seed(123)
    x0 = numpy.random.rand(n) * 2 - 1
    y0 = numpy.sin(1*numpy.pi*x0)  # + 1.0e-1 * (2*numpy.random.rand(n) - 1)
    a = -1.5
    b = +1.5
    u1 = smoothfit.fit1d(x0, y0, a, b, n=10, degree=1, eps=1.0e-1)

    x = numpy.linspace(a, b, 201)
    vals = [u1(xx) for xx in x]
    plt.plot(x0, y0, 'xk', label='data')
    plt.plot(
        x, vals, '-',
        color='k', alpha=0.3,
        label='smooth fit'
        )
    plt.xlim(a, b)
    plt.legend()

    # now scale the input points and values and do it again
    alpha = 0.1
    x0 *= alpha
    y0 *= alpha
    a *= alpha
    b *= alpha
    u2 = smoothfit.fit1d(x0, y0, a, b, n=10, degree=1, eps=1.0e-1)

    x = numpy.linspace(a, b, 201)
    vals = [u2(xx) for xx in x]
    plt.figure()
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


@pytest.mark.parametrize(
    'solver', ['dense', 'gmres']
    )
def test_2d(solver):
    n = 200
    numpy.random.seed(123)
    x0 = numpy.random.rand(n, 2) - 0.5
    # y0 = numpy.ones(n)
    # y0 = x0[:, 0]
    # y0 = x0[:, 0]**2
    # y0 = numpy.cos(numpy.pi*x0.T[0])
    # y0 = numpy.cos(numpy.pi*x0.T[0]) * numpy.cos(numpy.pi*x0.T[1])
    y0 = numpy.cos(numpy.pi*numpy.sqrt(x0.T[0]**2 + x0.T[1]**2))

    import meshzoo
    points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, 20, 20)

    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_circle([0.0, 0.0, 0.0], 1.0, 0.1)
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    u = smoothfit.fit2d(x0, y0, points, cells, eps=1.0e-2, solver=solver)

    ref = 4.4112141557993105
    val = assemble(u*u * dx)
    assert abs(val - ref) < 1.0e-10 * ref

    # from dolfin import XDMFFile
    # xdmf = XDMFFile('temp.xdmf')
    # xdmf.write(u)
    return


if __name__ == '__main__':
    test_1d_show()
    # test_2d('dense')
    # test_2d('gmres')
    # test_1d_scale()
