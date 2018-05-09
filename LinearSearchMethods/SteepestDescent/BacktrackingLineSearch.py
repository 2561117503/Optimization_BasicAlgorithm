#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-05 20:28
# @Abstract： Gradient Descent method based on backtracking line search

from matplotlib.pyplot import *
import numpy as np

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

# search step length
def BacktrackingLineSearch(x0):
    # init data 0 < c < 0.5 (typical:10^-4 0) < rho <= 1
    alpha = 1
    x = x0
    rho = 0.8
    c = 0.0001

    while f( x + alpha * (-f_grad(x)) ) > f(x) + c * alpha * f_grad(x) * (-f_grad(x)) :
        alpha *= rho

    return alpha

# Gradient Descent method
def GradientDescent():
    x0 = 0
    error = 10
    curve_y = [f(x0)]
    curve_x = [x0]

    while error > 1e-4:
        stepLength = BacktrackingLineSearch(x0)
        y0 = f(x0)
        x0 = x0 + stepLength * (-f_grad(x0))
        y1 = f(x0)
        error = y0-y1
        curve_x.append(x0)
        curve_y.append(y1)

    plot(curve_y, 'g*-')
    plot(curve_x, 'r+-')
    xlabel('iterations')
    ylabel('objective function value')
    legend(['backtracking line search algorithm'])
    show()

if __name__ == "__main__":
    GradientDescent()


