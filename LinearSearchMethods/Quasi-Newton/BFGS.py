#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-16 18:22
# @Abstract：BFGS Method

'''
min_{x \in R^2} f(x)=100(x^2_1−x_2)^2+(x_1−1)^2
solution: x^*=(1,1)^T,f(x*)=0
grad: g(x)=(400*x_1(x^2_1−x_2)+2(x_1−1),−200(x^2_1−x_2))
'''

from numpy import *
from matplotlib.pyplot import *

def fun(x_k):
    return 100 * (x_k[0, 0] ** 2 - x_k[1, 0]) ** 2 + (x_k[0, 0] - 1) ** 2

# grad fun
def fun_grad(x_k):
    result = zeros((2, 1))
    result[0, 0] = 400 * x_k[0, 0] * (x_k[0, 0] ** 2 - x_k[1, 0]) + 2 * (x_k[0, 0] - 1)
    result[1, 0] = -200 * (x_k[0, 0] ** 2 - x_k[1, 0])
    return result

def wolfe(x_k,p_k):
    alpha = 1
    c1 = 1e-4
    c2 = 0.9
    rho = 0.8

    k = 0
    maxIt = 20
    # wolfe condition
    while ( (k < maxIt) and ((fun(x_k + alpha * p_k) > fun(x_k) + c1 * (alpha *  fun_grad(x_k).T * p_k) ) or (
        (fun_grad(x_k + alpha * p_k).T * p_k) < c2 * (fun_grad(x_k).T * p_k))) ):  # condition 2
        alpha *= rho
        k += 1
    return alpha


def BFGS(x_k):
    epsilon = 1e-4  # iteration condtion
    m = shape(x_k)[0]
    # B_k = eye(m)
    H_k = eye(m)
    I = eye(m)
    y_array= [fun(x_k)]
    k = 1
    while abs(fun_grad(x_k)[0]) > epsilon:
        g_k = mat(fun_grad(x_k))
        p_k = - 1.0 * mat(H_k) * g_k
        # p_k = mat(-linalg.solve(B_k, g_k)) # search direction
        alpha_k = wolfe(x_k,p_k)
        x_k_old = x_k.copy()
        x_k +=  p_k * alpha_k
        g_k_old =g_k
        g_k = mat(fun_grad(x_k))
        s_k = x_k - x_k_old
        y_k = g_k - g_k_old

        if s_k.T * y_k > 0:
            H_k = (I - (s_k * y_k.T) /(y_k.T * s_k) ) * H_k * (I - (y_k * s_k.T) /(y_k.T * s_k) ) + s_k * s_k.T/(y_k.T * s_k)
            # B_k = B_k - 1.0 * (B_k * s_k * s_k.T * B_k) / (s_k.T * B_k * s_k)\
            #       + 1.0 * (y_k * y_k.T) / (s_k.T * y_k)

        k += 1
        y_array.append(fun(x_k))
        print(k)

    plot(y_array, 'g*-')
    show()

    return x_k

if __name__ == "__main__":
    x_0 = mat([[0.], [0.]])
    x_star = BFGS(x_0)
    print("---------------BFGS-------------------")
    print(x_star)
