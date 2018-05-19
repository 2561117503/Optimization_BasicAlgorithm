#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-19 10:40
# @Abstract：Limit-memory BFGS
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
    while ((k < maxIt) and ((fun(x_k + alpha * p_k) > fun(x_k) + c1 * (alpha * np.dot(np.transpose(x_k), p_k))) or (
                np.dot(np.transpose(fun_grad(x_k + alpha * p_k)), p_k) < c2 * np.dot(np.transpose(fun_grad(x_k)), p_k)))):  # condition 2
        alpha *= rho
        k += 1
    return alpha

def L_BFGS(x_k):
    epsilon = 1e-4  # iteration condtion
    y_array = [fun(x_k)]
    H_0 = eye(shape(x_k)[0])
    I = eye(shape(x_k)[0])
    m = 3
    s = []
    y = []

    g_k = fun_grad(x_k)
    p_k = - H_0 * g_k

    k = 0
    while abs(fun_grad(x_k)[0]) > epsilon:
        if k > 0 :
          r= np.dot(np.transpose(s[len(s)-1]),y[len(y)-1]) / np.dot(np.transpose(y[len(y)-1]),y[len(y)-1])
          H_0 =  I * r[0,0]

        # L-BFGS two-loop recursion
        t = len(s)
        q_k =fun_grad(x_k)
        alpha_i = []
        for i in range(t):
            alpha = np.dot(np.transpose(s[t - i - 1]),q_k) / np.dot(np.transpose(y[t - i - 1]), s[t - i - 1])
            q_k = q_k - alpha[0, 0] * y[t - i - 1]
            alpha_i.append(alpha[0, 0])
        r = np.dot(H_0 , q_k)

        for i in range(t):
            beta = np.dot(np.transpose(y[i]) , r) / np.dot(np.transpose(y[i]), s[i])
            r = r + np.dot(s[i],(alpha_i[t - i - 1] - beta[0, 0]))

        p_k = -r

        alpha = wolfe(x_k,p_k)
        x_k_old = x_k.copy()
        x_k = x_k + alpha * p_k

        if k >= m:
            s.pop(0)
            y.pop(0)

        s.append(x_k - x_k_old)
        y.append(fun_grad(x_k) - fun_grad(x_k_old))
        y_array.append(fun(x_k))

        k+=1
        print(k)

    plot(y_array, 'g*-')
    show()

    return x_k

if __name__ == "__main__":
    x_0 = mat([[0.], [0.]])
    x_star = L_BFGS(x_0)
    print("---------------L_BFGS-------------------")
    print(x_star)







