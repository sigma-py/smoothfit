# smoothfit

Smooth data fitting in N dimensions.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/smoothfit/master.svg)](https://circleci.com/gh/nschloe/smoothfit)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/smoothfit.svg)](https://codecov.io/gh/nschloe/smoothfit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![smooth](https://img.shields.io/badge/smooth-yes-8209ba.svg)](https://github.com/nschloe/smoothfit)
[![PyPi Version](https://img.shields.io/pypi/v/smoothfit.svg)](https://pypi.org/project/smoothfit)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/smoothfit.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/smoothfit)

Given experimental data, it is often desirable to produce a function whose values match
the data to some degree. A classical example is [polynomial
regression](https://en.wikipedia.org/wiki/Polynomial_regression).  There are various
pitfalls, however, most notoriously [Runge's
phenomenon](https://en.wikipedia.org/wiki/Runge%27s_phenomenon).

This module implements an alternative approach based on the following idea.

Given a

...

The same idea is used in for data smoothing in signal processing
(see, e.g., section 8.3 in [this
document](http://eeweb.poly.edu/iselesni/lecture_notes/least_squares/least_squares_SP.pdf)).

### Some examples

In one dimension, 



### Testing

To run the smoothfit unit tests, check out this repository and type
```
pytest
```

### License

smoothfit is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
