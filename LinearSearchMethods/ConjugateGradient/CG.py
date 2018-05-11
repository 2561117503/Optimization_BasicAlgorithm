#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-11 22:56
# @Abstract：Conjugate Gradient  Version
import numpy as np
from numpy import *

def CG(A,b,x_k):
    k = 0
    r_k  = A * x_k  - b  # r0
    p_k = -r_k           # p0

    xarrar = [x_k]
    rkarrar = [r_k]

    while r_k[0] >= 1e-8:
        r_k.shape=(2,1)
        p_k.shape=(2,1)
        alpha_k =  (np.transpose(r_k) * r_k) / (np.transpose(p_k) * A * p_k)
        x_k = x_k +  p_k * alpha_k
        r_k_old = r_k
        r_k = A * p_k * alpha_k + r_k
        beta_k = (np.transpose(r_k) * r_k ) / (np.transpose(r_k_old) * r_k_old)
        p_k = - r_k +   p_k * beta_k

        xarrar.append(x_k)
        rkarrar.append(r_k)
        k += 1

    print("xarrar",str(xarrar))
    print("\nrkarrar",str(rkarrar))
    return x_k


if __name__ == "__main__":
    A= mat([[4.0, 1.0], [1.0, 3.0]])
    b = mat([[1.0], [2.0]])
    x_0 = mat([[2.0], [1.0]])
    xStar = CG(A, b, x_0)
    print("\n x_star:",str(xStar))


'''
result:
xarrar [matrix([[2.],
        [1.]]), matrix([[0.23564955],
        [0.33836858]]), matrix([[0.09090909],
        [0.63636364]])]

rkarrar [matrix([[8.],
        [3.]]), matrix([[ 0.28096677],
        [-0.74924471]]), matrix([[-5.55111512e-17],
        [ 0.00000000e+00]])]

 x_star: [[0.09090909]
 [0.63636364]]

'''