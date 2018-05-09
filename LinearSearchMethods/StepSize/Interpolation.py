#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-09 20:48
# @Abstract：Interpolation

import numpy as np


# alpha：current stepsize
# phi：a function value about stepsize,h(a)=f(x_k+a*d)
# phi0：h(0)=f(x_k)
# g0：h'(0)=f'(0)
def QuadraticInterpolation(alpha, phi, phi0, g0):
    numerator = g0 * alpha * alpha
    denominator = -2 * ( phi -g0 * alpha - phi0 )

    return numerator/denominator


# alpha0 and alpha1: are stepsize: previous two iterations
# phi0:h(a0)
# phi1:h(a1)
# phi:h(0)=f(x)
# g:h'(0)
def CubicInterpolation(alpha0, phi0, alpha1, phi1, phi, g):
    matrix1 = np.matrix([[  alpha0**2, -alpha1**2],
                         [ -alpha0**3, alpha1**3]])
    matrix2 = np.matrix([[phi1-phi-g*alpha1],
                         [phi0-phi-g*alpha0]])
    ab = (1/(alpha0**2 * alpha1**2 * (alpha1-alpha0)))* (matrix1*matrix2)

    a = ab[0, 0]
    b = ab[1, 0]

    return (-b + np.sqrt(b ** 2 - 3 * a * g)) / (3 * a)


