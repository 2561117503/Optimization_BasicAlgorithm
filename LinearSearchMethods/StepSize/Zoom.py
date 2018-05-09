#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-09 21:19
# @Abstract：Zoom


# theta:a vector of parameters of the model
# args: other variables needed for fun and func
# d:search direction
# [a_low,a_high]: interval containing a stepsize satisfying Wolfe condition


def zoom(theta,args,d,a_low,a_high):
    if a_low > a_high:
        print("Invalid interval of stepsize in zoom procedure")
        return

    c1 = 1e-3   # c1: Armijo condition
    c2 = 0.9    # c2: curvature condition






