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

<img src="https://nschloe.github.io/smoothfit/eq0.png" width="25%">

The first expression is small if _f_ is straight;
the second expression is small if the function matches the sample points.

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

### Pics or it didn't happen

#### Runge's example

<img src="https://nschloe.github.io/smoothfit/runge-noise-05.svg" width="25%"> |
<img src="https://nschloe.github.io/smoothfit/runge-noise-05.svg" width="25%"> |
<img src="https://nschloe.github.io/smoothfit/runge-noise-05.svg" width="25%">
:-------------------:|:------------------:|:----------:|
`lmbda = 0.001`      |  `lmbda = 0.05`    |  `lmbda = 0.2`  |

```python
import matplotlib.pyplot as plt
import numpy
import smoothfit

a = -1.5
b = +1.5

# plot original function
x = numpy.linspace(a, b, 201)
plt.plot(x, 1 / (1 + 25 * x ** 2), "-", color="0.8", label="1 / (1 + 25 * x**2)")

# 21 sample points
x0 = numpy.linspace(-1.0, 1.0, 21)
y0 = 1 / (1 + 25 * x0 ** 2)
plt.plot(x0, y0, "xk")

u = smoothfit.fit1d(x0, y0, a, b, 1000, degree=1, lmbda=1.0e-6)
x = numpy.linspace(a, b, 201)
vals = [u(xx) for xx in x]
plt.plot(x, vals, "-", label="smooth fit")

plt.ylim(-0.1)
plt.grid()
plt.show()
```

[Runge's example function](https://en.wikipedia.org/wiki/Runge%27s_phenomenon) is a
tough nut for classical polynomial regression.

If there is no noise in the input data, the parameter `lmbda` can be chosen quite small
such that all data points are approximated well. Note that there are no oscillations
in the output function `u`.


#### Runge's example with noise

<img src="https://nschloe.github.io/smoothfit/runge-smoothfit.svg" width="40%">

```python
import matplotlib.pyplot as plt
import numpy
import smoothfit

a = -1.5
b = +1.5

# plot original function
x = numpy.linspace(a, b, 201)
plt.plot(x, 1 / (1 + 25 * x ** 2), "-", color="0.8", label="1 / (1 + 25 * x**2)")

# 21 sample points
numpy.random.seed(0)
n = 51
x0 = numpy.linspace(-1.0, 1.0, n)
y0 = 1 / (1 + 25 * x0 ** 2) + 1.0e-1 * (2 * numpy.random.rand(n) - 1)
plt.plot(x0, y0, "xk")

lmbda = 5.0e-2
u = smoothfit.fit1d(x0, y0, a, b, 1000, degree=1, lmbda=lmbda)
x = numpy.linspace(a, b, 201)
vals = [u(xx) for xx in x]
plt.plot(x, vals, "-", label="smooth fit")

plt.grid()
plt.show()
```

#### Few samples

<img src="https://nschloe.github.io/smoothfit/smoothfit-samples.svg" width="40%">

```python
import numpy
import smoothfit

x0 = numpy.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
y0 = numpy.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
u = smoothfit.fit1d(x0, y0, 0, 4, 1000, degree=1, lmbda=1.0)
```
Some noisy example data taken from
[Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm#Example).


#### A two-dimensional example

<img src="https://nschloe.github.io/smoothfit/2d.png" width="40%">

```python
import meshzoo
import numpy
import smoothfit 

n = 200
numpy.random.seed(123)
x0 = numpy.random.rand(n, 2) - 0.5
y0 = numpy.cos(numpy.pi * numpy.sqrt(x0.T[0] ** 2 + x0.T[1] ** 2))

# create a triangle mesh for the square
points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, 32, 32)

u = smoothfit.fit2d(x0, y0, points, cells, lmbda=1.0e-4, solver="dense-direct")

# Write the function to a file
from dolfin import XDMFFile
xdmf = XDMFFile("temp.xdmf")
xdmf.write(u)
```

This example approximates a function from _R<sup>2</sup>_ to _R_ (without noise in the
samples). Note that the absence of noise the data allows us to pick a rather small
`lmbda` such that all sample points are approximated well.


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
