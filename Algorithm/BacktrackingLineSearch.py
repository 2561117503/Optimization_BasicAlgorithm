#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-05 20:28
# @Abstract： backtracking line search

from matplotlib.pyplot import *
import numpy as np
def f(x):
    return (x-3)*(x-3)
def f_grad(x):
    return 2 * (x - 3)

# init data
np.random.seed(1)
alpha = 1
x = 0
rho = 0.8
c = 0.25

curve = [f(x)]
error = 1
k=0
while error > 1e-4 and f(x + alpha * f_grad(x)) > f(x) + c * alpha * f_grad(x) *f_grad(x):
    k+=1
    alpha = alpha * rho
    y = f(x)
    x= x - alpha * f_grad(x)
    error = y-f(x)
    curve.append(f(x))
    print("a_"+str(k) +":"+ str(alpha))

plot(curve, 'ro-')
xlabel('iterations'); ylabel('objective function value')
legend(['backtracking line search'])
show()