#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-09 21:19
# @Abstract：Zoom

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

def f_grad_2(x):
    return 2

# [a_low,a_high]: interval containing a stepsize satisfying Wolfe condition

def zoom(x_k, alpha_low, alpha_high):
    if alpha_low > alpha_high:
        print("Invalid interval of stepsize in zoom procedure")
        return

    c1 = 1e-4   # c1: Armijo condition
    c2 = 0.9    # c2: curvature condition

    eps = 1e-16
    while abs(alpha_high-alpha_low) >= eps:
        alpha_j = (alpha_low + alpha_high) / 2
        phi_alpha_j = f(x_k + alpha_j * (-f_grad(x_k)))  # direction: here select steepest descent
        phi_alpha_0 = f(x_k)
        phi_alpha_low = f(x_k + alpha_low * (-f_grad(x_k)))
        phi_grad_alpha_0 = f(x_k) * f_grad(x_k)

        if phi_alpha_j > phi_alpha_0 + c1 * alpha_j * ( phi_grad_alpha_0 ) or phi_alpha_j >= phi_alpha_low:
            alpha_high = alpha_j

        else:
            phi_grad_alpha_j = f(x_k + alpha_j * (-f_grad(x_k)))  * f_grad(x_k)

            if abs(phi_grad_alpha_j) <= -c2 * phi_grad_alpha_0:
                return alpha_j

            if phi_grad_alpha_j * (alpha_high-alpha_low) >= 0 :
                alpha_high = alpha_low

            alpha_low = alpha_j

    return alpha_low









