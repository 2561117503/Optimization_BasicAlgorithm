#coding=utf-8
# @Author: yangenneng
# @Time: 2018-04-24 14:32
# @Abstract：used cvxopt slove linear program problem demo
'''
# CVXOPT is a free software package for convex optimization based on the Python programming language.

'''
from cvxopt import matrix,solvers
import numpy as np
'''
problem:
 mimimize 2 x1 + x2
 subject to
         -x1 +  x2 <= 1
          x1 +  x2 >= 2
                x2 >= 0
          x1 - 2x2 <= 4
'''
# only can use <=, can't use >=, so change notion
c=matrix( [ 2.0, 1.0])
b=matrix( [ 1.0, -2.0, 0.0, 4.0])
A=matrix([[-1.0, -1.0, 0.0,  1.0],
          [1.0,  -1.0, -1.0, -2.0]])

sol=solvers.lp(c,A,b)

print(sol)
print(sol['x'])
print(sol['primal objective'])



'''
run result:
       pcost       dcost       gap    pres   dres   k/t
 0:  2.6471e+00 -7.0588e-01  2e+01  8e-01  2e+00  1e+00
 1:  3.0726e+00  2.8437e+00  1e+00  1e-01  2e-01  3e-01
 2:  2.4891e+00  2.4808e+00  1e-01  1e-02  2e-02  5e-02
 3:  2.4999e+00  2.4998e+00  1e-03  1e-04  2e-04  5e-04
 4:  2.5000e+00  2.5000e+00  1e-05  1e-06  2e-06  5e-06
 5:  2.5000e+00  2.5000e+00  1e-07  1e-08  2e-08  5e-08
Optimal solution found.
{'primal infeasibility': 1.1368786420268754e-08,
 'residual as primal infeasibility certificate': None,
 'iterations': 5,
 'primal slack': 2.0388399547644962e-08,
 'relative gap': 5.589978335987678e-08,
 'status': 'optimal',
 'gap': 1.3974945737847294e-07,
 'x': <2x1 matrix, tc='d'>,
 'dual infeasibility': 2.2578789546638247e-08,
 's': <4x1 matrix, tc='d'>,
 'y': <0x1 matrix, tc='d'>,
 'dual objective': 2.4999999817312526,
 'dual slack': 3.5299159726388937e-09,
 'z': <4x1 matrix, tc='d'>,
 'primal objective': 2.4999999895543072,
 'residual as dual infeasibility certificate': None}

[ 5.00e-01]
[ 1.50e+00]

2.4999999895543072
'''