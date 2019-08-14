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
the data to some degree.  A classical example is [polynomial
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


### Some examples


In one dimension, 



### Testing

To run the smoothfit unit tests, check out this repository and type
```
pytest
```

### License

smoothfit is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
