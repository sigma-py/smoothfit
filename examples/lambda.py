import matplotlib.pyplot as plt
import numpy as np

import smoothfit


def lambda_effect():
    n = 100
    np.random.seed(1)
    x0 = 2 * np.random.rand(n) - 1.0
    y0 = 1 / (1 + 25 * x0 ** 2) + 1.0e-1 * (2 * np.random.rand(*x0.shape) - 1)

    a = -1.5
    b = +1.5

    for k, lmbda in enumerate(np.logspace(-3, 1, num=41)):
        plt.plot(x0, y0, "xk")
        x = np.linspace(a, b, 201)
        # plt.plot(x, 1 / (1 + 25 * x ** 2), "-", color="0.8", label="1 / (1 + 25 * x**2)")

        u = smoothfit.fit1d(x0, y0, a, b, 1000, degree=1, lmbda=lmbda)
        x = np.linspace(a, b, 201)
        vals = [u(xx) for xx in x]
        plt.plot(x, vals, "-", color="#d62728")
        plt.title(f"lmbda = {lmbda:.1e}")
        plt.xlim(a, b)
        plt.ylim(-0.2, 1.2)
        # plt.show()
        plt.savefig(
            f"smoothfit-lambda-{k:02d}.png", bbox_inches="tight", transparent=True
        )
        plt.close()
    return


if __name__ == "__main__":
    lambda_effect()
