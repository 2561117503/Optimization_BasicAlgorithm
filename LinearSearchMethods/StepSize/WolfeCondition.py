#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-20 0:12
# @Abstract：

import numpy as np
from numpy import *
from matplotlib.pyplot import *

# bisection search of wolfe condition
def fun(x_k):
    return 100 * (x_k[0, 0] ** 2 - x_k[1, 0]) ** 2 + (x_k[0, 0] - 1) ** 2

# grad fun
def fun_grad(x_k):
    result = zeros((2, 1))
    result[0, 0] = 400 * x_k[0, 0] * (x_k[0, 0] ** 2 - x_k[1, 0]) + 2 * (x_k[0, 0] - 1)
    result[1, 0] = -200 * (x_k[0, 0] ** 2 - x_k[1, 0])
    return result

def wolfe(x_k,p_k):
    alpha_low = 0
    alpha_high = 1000
    alpha = 1
    c1 = 1e-4
    c2 = 0.9

    maxIt = 20
    k = 0
    while k < maxIt:
        if (fun(x_k + alpha * p_k) > fun(x_k) + c1 * (alpha *  fun_grad(x_k).T * p_k)):
            alpha_high = alpha
            alpha = (alpha_high+alpha_low) / 2
        elif fun_grad(x_k + alpha * p_k).T * p_k < c2 * fun_grad(x_k).T * p_k:
            alpha_low = alpha
            if alpha_high > 100:
                alpha = 2 * alpha_low
            else:
                alpha = (alpha_high + alpha_low) / 2
        else:
            break
        k += 1

    return alpha