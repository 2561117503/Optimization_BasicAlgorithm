#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-09 21:17
# @Abstract：Wolfe Line Search method

from LinearSearchMethods.StepSize.Zoom import zoom
from LinearSearchMethods.StepSize.Interpolation import *

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

def f_grad_2(x):
    return 2

def WolfeLineSearch(x_k):
    alpha_0 = 0
    alpha_max = 1
    alpha_1 = 0.7

    c1 = 1e-4  # c1: Armijo condition
    c2 = 0.9  # c2: curvature condition

    alpha_pre = alpha_0
    alpha_cur = alpha_1
    alpha_min = 1e-7

    i = 0
    eps = 1e-16
    while abs(alpha_cur-alpha_pre)>=eps:
        phi_alpha_cur = f(x_k + alpha_cur*-(f_grad(x_k)))
        phi_alpha_pre = f(x_k + alpha_pre*-(f_grad(x_k)))
        phi_alpha_0 = f(x_k)
        phi_grad_alpha_0 = f(x_k) * (-f_grad(x_k))

        if phi_alpha_cur > phi_alpha_0 + c1 * alpha_cur *phi_grad_alpha_0 or (phi_alpha_cur> phi_alpha_pre and i> 0):
            return zoom(x_k,alpha_pre,alpha_cur)

        phi_grad_alpha_cur = f(x_k+alpha_cur * (-f_grad(x_k))) * (-f_grad(x_k))
        if abs(phi_grad_alpha_cur)<= -c2* phi_grad_alpha_0: # satisfy Wolfe condition
            return alpha_cur

        if phi_grad_alpha_cur >= 0:
            aaa=zoom(x_k,alpha_cur,alpha_max)
            return zoom(x_k,alpha_cur,alpha_max)

        alpha_new = QuadraticInterpolation(alpha_cur, phi_alpha_cur, phi_alpha_0, phi_grad_alpha_0)
        alpha_pre = alpha_cur
        alpha_cur = alpha_new
        i+=1

    return alpha_min



