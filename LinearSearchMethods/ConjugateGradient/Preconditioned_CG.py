#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-12 15:09
# @Abstract：Preconditioned Conjugate Gradient Version

from LinearSearchMethods.ConjugateGradient.CG import CG
from numpy import *
import numpy as np

def solveAxb(A,r_k):
     return A.I * r_k

# M:preconditioning M = C^T * C symmetric and positive definite, C nonsingular matrix
def Preconditioned_CG(A,b,x_k,M):
    r_k = A * x_k  - b  # r0
    y_k = solveAxb(M,r_k) # CG(M,r_k,mat([[2.0], [1.0]]))
    p_k = - y_k

    k = 0
    yarrar = [y_k]; xarrar = [x_k]; rkarrar = [r_k]

    while r_k[0] >= 1e-8:
        r_k.shape = (2, 1)
        p_k.shape = (2, 1)
        alpha_k = (np.transpose(r_k) * y_k) / (np.transpose(p_k) * A * p_k)
        x_k = x_k + p_k * alpha_k
        r_k_old = r_k
        r_k = A * p_k * alpha_k + r_k
        y_k_old = y_k
        y_k = solveAxb(M,r_k) #CG(M, r_k ,mat([[2.0], [1.0]])) # y_{k+1}

        beta_k = (np.transpose(r_k) * y_k) / (np.transpose(r_k_old) * y_k_old)
        p_k = - y_k + p_k * beta_k

        xarrar.append(x_k); yarrar.append(y_k); rkarrar.append(r_k)
        k += 1

    print("---------------Preconditioned_CG---------------")
    print("Preconditioned_CG: xarrar", str(xarrar))
    print("\nPreconditioned_CG: yarrar", str(yarrar))
    print("\nPreconditioned_CG :rkarrar", str(rkarrar))
    return x_k

if __name__ == "__main__":
    A   = mat([[4.0, 1.0],
              [1.0, 3.0]])
    b   = mat([[1.0], [2.0]])
    x_0 = mat([[2.0], [1.0]])
    M =   mat([[1.0, 0.0],
               [0.0, 1.0]])
    # print(solveAxb(M,b))
    xStar = Preconditioned_CG(A, b, x_0,M)
    print("\n Preconditioned_CG: x_star:",str(xStar))
