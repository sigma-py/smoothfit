import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

a = -0.8
b = +0.8

x = np.linspace(a, b, 201)
y = 1 / (1 + 25 * x ** 2)
plt.plot(x, y, "-", color="#61ded4", linewidth=10, label="1 / (1 + 25 * x**2)")

x0 = np.array([-0.7, -0.5, -0.4, -0.2, 0.0, 0.1, 0.2, 0.4, 0.5, 0.75])
y0 = 1 / (1 + 25 * x0 ** 2)
y0 += 1.0e-1 * (2 * rng.random(y0.shape[0]) - 1)
plt.plot(x0, y0, "Xk", markersize=20)

plt.xlim(a, b)
plt.axis("off")
plt.ylim(-0.2, 1.2)
plt.gca().set_aspect("equal")
plt.show()
# plt.savefig("logo.svg", bbox_inches="tight", transparent=True)
