#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-18 20:28
# @Abstract：Inexact Newton method: Line Search Newton-CG
import scipy.optimize._minimize
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

# grad fun_2
def fun_grad_2(x_k):
    result = zeros((2, 2))
    result[0, 0] = 1200 * x_k[0, 0]**2 - 400 * x_k[1, 0] + 2
    result[0, 1] = -400 * x_k[1, 0]
    result[1, 0] = -400 * x_k[1, 0]
    result[1, 1] = 200
    return result

def sqrt_norm(g_f_k):
    result = g_f_k[0, 0]**2 + g_f_k[1, 0]**2
    return sqrt(result)

def r_norm(g_f_k):
    result = g_f_k[0, 0]**2 + g_f_k[1, 0]**2
    return result


def wolfe(x_k,p_k):
    alpha = 1
    c1 = 1e-4
    c2 = 0.9
    rho = 0.8

    k = 0
    maxIt = 20
    # wolfe condition
    while ( (k < maxIt) and ((fun(x_k + alpha * p_k) > fun(x_k) + c1 * (alpha *  np.dot(np.transpose(x_k),p_k)) ) or (
        np.dot(np.transpose(fun_grad(x_k + alpha * p_k)) , p_k) < c2 * np.dot(np.transpose(fun_grad(x_k)) , p_k))) ):  # condition 2
        alpha *= rho
        k += 1
    return alpha


def LineSearchNewton_CG(x_k):
    epsilon = 1e-4  # iteration condtion
    k = 0
    y_array=[fun(x_k)]
    while abs(fun_grad(x_k)[0]) > epsilon:
        epsilon_k = min(0.5,sqrt_norm(fun_grad(x_k)))  #tolerance
        z_0 = 0
        r_0 = fun_grad(x_k)
        d_0 = - r_0

        B_k = fun_grad_2(x_k)
        p_k = None

        j = 0
        z_j = z_0
        r_j = r_0
        d_j = d_0
        while 1:
            dBd = np.dot(np.dot(np.transpose(d_j), B_k), d_j)
            if (dBd[0,0] <= 0):
                if j==0:
                    p_k = - fun_grad(x_k)
                    break
                else:
                    p_k = z_j
                    break
            alpha_j = np.dot(np.transpose(r_j) , r_j) / np.dot(np.dot(np.transpose(d_j), B_k) , d_j)
            z_j = z_j + alpha_j[0,0] * d_j
            r_j_old = r_j.copy()
            r_j = r_j + alpha_j * np.dot(B_k , d_j)
            if r_norm(r_j) < epsilon_k:
                p_k = z_j
                break
            balta_j = np.dot(np.transpose(r_j) , r_j) / np.dot(np.transpose(r_j_old), r_j_old)
            d_j = -r_j + balta_j * d_j
            print("j:",str(j))
            j+=1

        print("k:", str(k))
        alpha= wolfe(x_k,p_k)
        x_k = x_k + alpha * p_k
        y_array.append(fun(x_k))
        k+=1

    plot(y_array, 'r*-')
    show()
    return x_k

if __name__ == "__main__":
    x_0 = mat([[0.], [0.]])
    x_star = LineSearchNewton_CG(x_0)
    print("---------------Line Search Newton CG-------------------")
    print(x_star)