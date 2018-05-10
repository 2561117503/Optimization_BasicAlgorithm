#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-10 21:30
# @Abstract：Cholesky with Added Multiple of the Identity
import numpy as np
from LinearSearchMethods.MatrixUtil.Cholesky_LDL import Cholesky_LDL

def AddedMultipleOfTheIdentity(A):
    balta = 1e-3
    # judge symmetric matrix
    if len(np.shape(A)) != 2 or np.shape(A)[0] != np.shape(A)[1]:
        print("error shape")
        return

    t_0 = None
    n = np.shape(A)[0]  # dimension of matrix A
    min_aii = -1e8
    for i in range(n):
        if A[i][i] < min_aii:
            min_aii = A[i][i]

    if min_aii > 0:
        t_0 = 0
    else:
        t_0 = -min_aii + balta

    for k in range(1000):
        I = np.eye(n)  # diagonal element=1 matrix
        L = np.eye(n)  # diagonal element=1 matrix
        D = np.zeros((n, n))  # zero matrix
        try:
            L,D = Cholesky_LDL(A + t_0 * I)
            return L
        except:
            t_0 = max(2*t_0, balta)

if __name__ == "__main__":
    A = [
        [4, 12, 16],
        [12, 37, -43],
        [-16, -43, 98]];
    L= AddedMultipleOfTheIdentity(A)
    print(L)

