import matplotlib.pyplot as plt
import numpy as np
import pytest

import smoothfit

dolfin = pytest.importorskip("dolfin")
assemble = dolfin.assemble
dx = dolfin.dx


@pytest.mark.parametrize("solver", ["dense-direct", "sparse-cg", "Nelder-Mead"])
def test_1d(solver, show=False):
    n = 20
    # x0 = np.linspace(-1.0, 1.0, n)
    np.random.seed(2)
    x0 = np.random.rand(n) * 2 - 1

    # y0 = x0.copy()
    # y0 = 1 / (1 + 25*x0**2) + 5.0e-2 * (2*np.random.rand(n)-1)
    # y0 = np.exp(x0) + 1.0e-1 * (2*np.random.rand(n) - 1)
    # y0 = x0 + 1.0e-1 * (2*np.random.rand(n) - 1)
    # y0 = x0**3 + 1.0e-1 * (2*np.random.rand(n) - 1)
    y0 = np.sin(1 * np.pi * x0)  # + 1.0e-1 * (2*np.random.rand(n) - 1)
    # y0 = 1 / (x0 + 2) + 1.0e-2 * (2*np.random.rand(n) - 1)

    a = -1.5
    b = +1.5

    lmbda = 1.0e-2
    u = smoothfit.fit1d(
        x0, y0, a, b, 64, lmbda, degree=1, solver=solver, variant="dolfin"
    )

    if solver == "Nelder-Mead":
        ref = 0.659_143_389_243_738_2
    else:
        ref = 1.417_961_801_095_804_4

    val = assemble(u * u * dx)
    assert abs(val - ref) < 1.0e-10 * ref

    if show:
        # x = u.function_space().mesh().coordinates()
        x = np.linspace(a, b, 201)
        vals = [u(xx) for xx in x]
        plt.plot(x, vals, "-", label="smooth fit")

        plt.plot(x0, y0, "xk", label="samples")
        plt.plot(x, np.sin(np.pi * x), "-", color="0.8", label="original")
        plt.xlim(a, b)
        plt.legend()
        plt.show()


def test_runge_show():
    n = 21
    x0 = np.linspace(-1.0, 1.0, n)
    y0 = 1 / (1 + 25 * x0 ** 2)

    a = -1.5
    b = +1.5

    plt.plot(x0, y0, "xk")
    x = np.linspace(a, b, 201)
    plt.plot(x, 1 / (1 + 25 * x ** 2), "-", color="0.8", label="1 / (1 + 25 * x**2)")

    # u = smoothfit.fit1d(x0, y0, a, b, 1000, degree=1, lmbda=1.0e-6)
    # x = np.linspace(a, b, 201)
    # vals = [u(xx) for xx in x]
    # plt.plot(x, vals, "-", label="smooth fit")

    for degree in [2, 4, 16]:
        x = np.linspace(a, b, 201)
        p = np.polyfit(x0, y0, degree)
        vals = np.polyval(p, x)
        plt.plot(x, vals, "-", label=f"polyfit {degree}")

    plt.xlim(a, b)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.ylim(-0.2, 1.2)
    plt.show()
    # plt.savefig("runge-polyfit.svg", bbox_inches="tight", transparent=True)


def test_noisy_runge():
    n = 100
    np.random.seed(3)
    x0 = 2 * np.random.rand(n) - 1.0
    y0 = 1 / (1 + 25 * x0 ** 2)
    y0 += 1.0e-1 * (2 * np.random.rand(*x0.shape) - 1)

    a = -1.5
    b = +1.5

    plt.plot(x0, y0, "xk")
    x = np.linspace(a, b, 201)
    # plt.plot(x, 1 / (1 + 25 * x ** 2), "-", color="0.8", label="1 / (1 + 25 * x**2)")

    lmbda = 0.2
    u = smoothfit.fit1d(x0, y0, a, b, 200, degree=1, lmbda=lmbda, variant="dolfin")
    x = np.linspace(a, b, 201)
    vals = [u(xx) for xx in x]
    plt.plot(x, vals, "-")
    # plt.title(f"lmbda = {lmbda:.1e}")
    plt.xlim(a, b)
    plt.ylim(-0.2, 1.2)
    plt.grid()
    plt.gca().set_aspect("equal")
    # plt.show()
    plt.savefig("runge-noise-02.svg", bbox_inches="tight", transparent=True)


def test_samples():
    # From <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm#Example>
    x0 = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
    y0 = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
    u = smoothfit.fit1d(x0, y0, 0, 4, 1000, degree=1, lmbda=1.0, variant="dolfin")

    # plot the function
    x = np.linspace(0, 4, 201)
    vals = [u(xx) for xx in x]
    plt.plot(x, vals, "-", label="smooth fit")
    plt.plot(x0, y0, "xk")
    plt.xlim(0, 4)
    plt.ylim(0)
    plt.grid()
    # plt.show()
    plt.savefig("smoothfit-samples.svg", bbox_inches="tight", transparent=True)


# def test_1d_scale():
#     n = 20
#     # x0 = np.linspace(-1.0, 1.0, n)
#     np.random.seed(3)
#     x0 = np.random.rand(n) * 2 - 1
#     y0 = np.sin(1 * np.pi * x0)  # + 1.0e-1 * (2*np.random.rand(n) - 1)
#     a = -1.5
#     b = +1.5
#     u1 = smoothfit.fit1d(x0, y0, a, b, n=100, degree=1, lmbda=2.0e-1)
#
#     x = np.linspace(a, b, 201)
#     vals = [u1(xx) for xx in x]
#     plt.plot(x0, y0, "xk", label="data")
#     plt.plot(x, vals, "-", color="k", alpha=0.3, label="smooth fit")
#     plt.xlim(a, b)
#     plt.legend()
#
#     # now scale the input points and values and do it again
#     alpha = 0.1
#     x0 *= alpha
#     y0 *= alpha
#     a *= alpha
#     b *= alpha
#     u2 = smoothfit.fit1d(x0, y0, a, b, n=100, degree=1, lmbda=2.0e-1 * alpha**3)
#
#     x = np.linspace(a, b, 201)
#     vals = [u2(xx) for xx in x]
#     plt.figure()
#     plt.plot(x0, y0, "xk", label="data")
#     plt.plot(x, vals, "-", color="k", alpha=0.3, label="smooth fit")
#     plt.xlim(a, b)
#     plt.legend()
#     plt.show()


@pytest.mark.parametrize(
    "solver",
    [
        "dense-direct",
        # "sparse-cg",  # fails on circleci
        "lsqr",
        "lsmr",
    ],
)
def test_2d(solver, write_file=False):
    n = 200
    np.random.seed(123)
    x0 = np.random.rand(n, 2) - 0.5
    # y0 = np.ones(n)
    # y0 = x0[:, 0]
    # y0 = x0[:, 0]**2
    # y0 = np.cos(np.pi*x0.T[0])
    # y0 = np.cos(np.pi*x0.T[0]) * np.cos(np.pi*x0.T[1])
    y0 = np.cos(np.pi * np.sqrt(x0.T[0] ** 2 + x0.T[1] ** 2))

    import meshzoo

    points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, 32, 32)

    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_circle([0.0, 0.0, 0.0], 1.0, 0.1)
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    u = smoothfit.fit(x0, y0, points, cells, lmbda=1.0e-5, solver=solver)

    # ref = 4.411_214_155_799_310_5
    # val = assemble(u * u * dx)
    # assert abs(val - ref) < 1.0e-10 * ref

    if write_file:
        from dolfin import XDMFFile

        xdmf = XDMFFile("temp.xdmf")
        xdmf.write(u)


if __name__ == "__main__":
    test_1d("dense-direct", show=True)
    # test_runge_show()
    # test_noisy_runge()
    # test_samples()
    # test_2d("dense-direct", write_file=True)
    # test_2d("minimization")
    # test_2d("sparse")
    # test_1d_scale()
