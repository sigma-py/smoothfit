import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

a = -1.5
b = +1.5


def f(x):
    return 1 / (1 + 25 * x ** 2)


fig, ax = plt.subplots()


# plot original function
x = np.linspace(a, b, 201)
plt.plot(x, f(x), "-", color="0.8", label="1 / (1 + 25 * x ** 2)")

samples = plt.plot([], [], "xk")[0]
fit = plt.plot([], [], "-", label="np.polyfit")[0]


def update(k):
    # k sample points
    xs = np.linspace(-1.0, 1.0, k)
    ys = f(xs)
    samples.set_xdata(xs)
    samples.set_ydata(ys)

    p = np.polyfit(xs, ys, len(xs) - 1)
    xfit = np.linspace(a, b, 500)
    yfit = np.polyval(p, xfit)
    fit.set_xdata(xfit)
    fit.set_ydata(yfit)

    plt.title(f"{k} sample points")
    plt.ylim(-0.1, 1.1)
    plt.xlim(a, b)
    plt.legend()


anim = FuncAnimation(fig, update, frames=range(2, 40), interval=200)
anim.save("runge-polyfit.webp", writer="imagemagick")
