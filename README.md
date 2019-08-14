# smoothfit

Smooth data fitting in N dimensions.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/smoothfit/master.svg?style=flat-square)](https://circleci.com/gh/nschloe/smoothfit)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/smoothfit.svg?style=flat-square)](https://codecov.io/gh/nschloe/smoothfit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![smooth](https://img.shields.io/badge/smooth-yes-8209ba.svg?style=flat-square)](https://github.com/nschloe/smoothfit)
[![PyPi Version](https://img.shields.io/pypi/v/smoothfit.svg?style=flat-square)](https://pypi.org/project/smoothfit)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/smoothfit.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/smoothfit)
[![PyPi downloads](https://img.shields.io/pypi/dm/smoothfit.svg?style=flat-square)](https://pypistats.org/packages/smoothfit)

Given experimental data, it is often desirable to produce a function whose values match
the data to some degree. A classical example is [polynomial
regression](https://en.wikipedia.org/wiki/Polynomial_regression).  Polynomials are
chosen because they are very simple, can be evaluated quickly, and [can be made to fit
any function very closely](https://en.wikipedia.org/wiki/Stone–Weierstrass_theorem).

There are, however, some fundamental problems with this approach:

 * Your data might not actually fit a polynomial of low degree.
 * [Runge's phenomenon](//en.wikipedia.org/wiki/Runge%27s_phenomenon).

This module implements an alternative approach to data fitting, starting from the
general idea that

 * you want your data to fit the curve,
 * you want your curve to be smooth.

This can be molded into an optimization problem: You're looking for a
twice-differentiable function _f_ that minimizes the expression

Σ<sub>i</sub> (f(x<sub>i</sub>) - y<sub>i</sub>)<sup>2</sup> + λ ‖Δf‖<sup>2</sup><sub>L<sup>2</sup>(Ω)</sub> → min.

The first expression is small if the function matches the sample points; the second
expression is small if _f_ is flat.

(The same idea is used in for data smoothing in signal processing (see, e.g., section
8.3 in [this
document](http://eeweb.poly.edu/iselesni/lecture_notes/least_squares/least_squares_SP.pdf)).)

This minimization problem can be discretized in terms of, e.g., finite elements.

Advantages of the new approach:

 * No oscillations.
 * Works in multiple dimensions.

| <img src="https://nschloe.github.io/smoothfit/runge-polyfit.svg" width="70%"> | <img src="https://nschloe.github.io/smoothfit/runge-smoothfit.svg" width="70%"> |
| :----------:         |  :---------:         |
| Polynomial fits. Oscillations at the boundary.  | Smooth fit.          |


The plot can be recreated with

```python
import matplotlib.pyplot as plt
import numpy
import smoothfit

n = 21
x0 = numpy.linspace(-1.0, 1.0, n)
y0 = 1 / (1 + 25 * x0 ** 2)
a = -1.5
b = +1.5

# plot the sample points
plt.plot(x0, y0, "xk")

# create smooth fit
u = smoothfit.fit1d(x0, y0, a, b, 1000, degree=1, lmbda=1.0e-6)

# plot it
x = numpy.linspace(a, b, 201)
vals = [u(xx) for xx in x]
plt.plot(x, vals, "-")

plt.grid()
plt.show()
```

### Comparison with other approaches

#### Polynomial fitting/regression

#### Fourier smoothing

One approach to data fitting with smoothing is to create a function with all data
points, and simply cut off the high frequencies after Fourier transformation.

Advantages:
  * fast

Disadvantages:
  * Only works for evenly spaced samples

```python
import matplotlib.pyplot as plt
import numpy


numpy.random.seed(0)

# original function
x0 = numpy.linspace(-1.0, 1.0, 1000)
y0 = 1 / (1 + 25 * x0 ** 2)
plt.plot(x0, y0, color="k", alpha=0.2)

# create sample points
n = 51
x1 = numpy.linspace(-1.0, 1.0, n)  # only works if samples are evenly spaced
y1 = 1 / (1 + 25 * x1 ** 2) + 1.0e-1 * (2 * numpy.random.rand(x1.shape[0]) - 1)
plt.plot(x1, y1, "xk")

# Cut off the high frequencies in the transformed space and transform back
X = numpy.fft.rfft(y1)
X[5:] = 0.0
y2 = numpy.fft.irfft(X, n)
#
plt.plot(x1, y2, "-", label="5 lowest frequencies")

plt.grid()
plt.show()
```

### Testing

To run the smoothfit unit tests, check out this repository and type
```
pytest
```

### License

smoothfit is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
