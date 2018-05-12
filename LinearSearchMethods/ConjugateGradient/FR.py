#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-12 19:53
# @Abstract： Fletcher-Reeves methods: extend the conjugate gradient method to nonlinear functions

import numpy as np
from numpy import *

def fun(X):
    A= mat([[4.0, 0.0], [0.0, 2.0]])
    X.shape=(2,1)
    return 0.5 * np.transpose(X) * A * X

def fun_grad(X):
    return mat([[X[0,0] * 4] ,[X[1,0] * 2]])

def FR(A, x_k):
    grad_f_k = fun_grad(x_k)  # grad_f_0
    p_k = - grad_f_k
    k = 0
    xArray=[x_k]
    gradArray=[grad_f_k]
    while (abs(grad_f_k[0,0]) >= 1e-4 ) :
        grad_f_k.shape = (2, 1)
        p_k.shape = (2, 1)
        alpha_k = - (np.transpose(grad_f_k) * p_k ) / (np.transpose(p_k) * A * p_k)
        x_k = x_k + p_k * alpha_k
        grad_f_k_old = grad_f_k
        grad_f_k = fun_grad(x_k)
        balta_k_fr = ( np.transpose(grad_f_k) * grad_f_k ) / ( np.transpose(grad_f_k_old) * grad_f_k_old )
        p_k = - grad_f_k + p_k * balta_k_fr

        xArray.append(x_k)
        gradArray.append(grad_f_k)
        k += 1
    print(xArray)
    print(gradArray)
    return x_k

if __name__ == "__main__":
    A = mat([[4.0, 0.0], [0.0, 2.0]])
    x_0 = mat([[2.0], [2.0]])
    x_star = FR(A,x_0)
    print("---------FR-----------\n x_star:\n")
    print(x_star)

