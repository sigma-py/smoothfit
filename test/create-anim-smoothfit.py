import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import smoothfit

a = -1.5
b = +1.5


def f(x):
    return 1 / (1 + 25 * x ** 2)


fig, ax = plt.subplots()


# plot original function
x = np.linspace(a, b, 201)
plt.plot(x, f(x), "-", color="0.8", label="1 / (1 + 25 * x ** 2)")

samples = plt.plot([], [], "xk")[0]
fit = plt.plot([], [], "-", label="smooth fit")[0]


def update(k):
    # k sample points
    xs = np.linspace(-1.0, 1.0, k)
    ys = f(xs)
    samples.set_xdata(xs)
    samples.set_ydata(ys)
    # plt.plot(xs, ys, "xk")

    basis, coeffs = smoothfit.fit1d(xs, ys, a, b, 200, degree=1, lmbda=1.0e-6)
    fit.set_xdata(basis.mesh.p[0])
    fit.set_ydata(coeffs[basis.nodal_dofs[0]])

    plt.title(f"{k} sample points")
    plt.ylim(-0.1, 1.1)
    plt.xlim(a, b)
    plt.legend()


anim = FuncAnimation(fig, update, frames=range(2, 40), interval=200)
anim.save("out.webp", writer="imagemagick")
