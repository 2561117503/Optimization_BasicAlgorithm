#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-05 20:28
# @Abstract： backtracking line search method

from matplotlib.pyplot import *
import numpy as np

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

# init data
'''
0 < c < 0.5 typical:10^{-4}
0 < rho <= 1
'''
np.random.seed(1)
alpha = 1
x = 0
rho = 0.8
c = 0.0001

curve1 = [x]     # draw x iteration graph
curve2 = [f(x)]  # draw f(x) iteration graph
k = 0

while k<8 and f(x + alpha * f_grad(x)) > f(x) + c * alpha * f_grad(x) * f_grad(x):
    k += 1
    alpha *= rho
    x = x + alpha * -(f_grad(x))

    curve1.append(x)
    curve2.append(f(x))

print("k", k)
print("last a_k:", alpha)

plot(curve2, 'ro-')
plot(curve1, 'g*-')

xlabel('iterations');  ylabel('objective function value')
legend(['backtracking line search algorithm'])
show()
