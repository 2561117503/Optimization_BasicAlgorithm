#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-09 20:19
# @Abstract： Armijo backtracking line search

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

def f_grad_2(x):
    return 2

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

