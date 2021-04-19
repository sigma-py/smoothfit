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

# sample points with noise
n = 51
rng = np.random.default_rng(0)
x0 = np.linspace(-1.0, 1.0, n)
y0 = 1 / (1 + 25 * x0 ** 2)
y0 += 1.0e-1 * (2 * rng.random(n) - 1)
plt.plot(x0, y0, "xk")

samples = plt.plot([], [], "xk")[0]
fit = plt.plot([], [], "-", label="smooth fit")[0]


def update(lmbda):
    basis, coeffs = smoothfit.fit1d(x0, y0, a, b, 500, degree=1, lmbda=lmbda)
    fit.set_xdata(basis.mesh.p[0])
    fit.set_ydata(coeffs[basis.nodal_dofs[0]])

    plt.title(f"lambda = {lmbda:.3e}")
    plt.ylim(-0.1, 1.1)
    plt.xlim(a, b)
    plt.legend()


lmbda_range = np.logspace(-3.0, 1.0, 50)

anim = FuncAnimation(fig, update, frames=lmbda_range, interval=200)
anim.save("runge-noise-lambda.webp", writer="imagemagick")
