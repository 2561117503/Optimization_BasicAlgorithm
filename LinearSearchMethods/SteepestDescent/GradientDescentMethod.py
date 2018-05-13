#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-05 20:28
# @Abstract： Gradient Descent method based on backtracking line search
'''

Gradient descent method.
given a starting point x ∈ domf.
repeat
    1. x := −∇f(x).
    2. Line search. Choose step size t via exact or backtracking line search.
    3. Update. x := x + tx.
until stopping criterion is satisfied.
'''
from matplotlib.pyplot import *
import numpy as np
from LinearSearchMethods.StepSize.WolfeLineSearch import WolfeLineSearch

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

# search step size
# x0: start point
def BacktrackingLineSearch(x0):
    # init data 0 < c < 0.5 (typical:10^-4 0) < rho <= 1
    alpha = 1
    x = x0
    rho = 0.8
    c = 1e-4

    # Armijo condition
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
        # stepSize = WolfeLineSearch(x0) # Wolfe condition
        stepSize = BacktrackingLineSearch(x0) # Armijo condition
        y0 = f(x0)
        x0 = x0 + stepSize * (-f_grad(x0))

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


