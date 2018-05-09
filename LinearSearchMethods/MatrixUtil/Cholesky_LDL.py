#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-09 18:59
# @Abstract：Cholesky Factorization , LDL^T Form

import numpy as np

# LDL
# H: symmetry matrix
# n: H:n*n
def Cholesky_LDL(H):

    # judge symmetric matrix
    if len(np.shape(H)) != 2 or np.shape(H)[0] != np.shape(H)[1]:
        print("error shape")
        return

    n = np.shape(H)[0]   # dimension of matrix H
    L = np.eye(n)        # diagonal element=1 matrix
    D = np.zeros((n, n)) # zero matrix

    for j in range(n):
        c = 0
        for s in range(j):
            c += D[s][s] * L[j][s] * L[j][s]
        D[j][j] = H[j][j] - c

        for i in range(j+1, n):
            c = 0
            for s in range(j):
                c += D[s][s] * L[i][s] * L[j][s]
                # H[i][j] = H[i][j] - c
            L[i][j] = (H[i][j]-c) / D[j][j]

    return L, D



if __name__ == "__main__":
    H=[
        [   4,  12,  16],
        [  12,  37, -43],
        [ -16, -43,  98]];

    L, D = Cholesky_LDL(H)

    print("L:")
    print(L)
    print("---------------------------\nD:")
    print(D)




'''
 H=[
        [   4,  12,  16],
        [  12,  37, -43],
        [ -16, -43,  98]];

result:
L:
[[ 1.  0.  0.]
 [ 3.  1.  0.]
 [-4.  5.  1.]]
---------------------------
D:
[[4. 0. 0.]
 [0. 1. 0.]
 [0. 0. 9.]]
'''