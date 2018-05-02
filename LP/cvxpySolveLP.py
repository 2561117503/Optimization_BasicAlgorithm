#coding=utf-8
# @Author: yangenneng
# @Time: 2018-04-24 10:31
# @Abstract：a simple LP optimization problem demo

from cvxpy import *
import cvxpy as cvx

'''
problem:
 mimimize 2 x + y
 subject to
         -x +  y <= 1
          x+  y >= 2
                y >= 0
          x - 2*y <= 4
'''

# create two scalar optimization Variable
x = Variable()
y = Variable()

# create two constraints
constraints=[ -x +   y <= 1,
               x +   y >= 2,
                     y >= 0,
               x - 2*y <= 4]

# optimization objective function
obj = Minimize(2*x+y)

prob=Problem(obj,constraints)

# return the optimization value
prob.solve(solver=cvx.GLPK)
print("status:",prob.status)
print("optimization value:",prob.value)
print("optimization var:",x.value,y.value)

'''
run result:
status: optimal
optimization value: 2.5
optimization var: 0.5 1.5
'''



'''
status:
    optimal
    infeasible
    unbounded
    optimal_inaccurate
    infeasible_inaccurate
    unbounded_inaccurate

optimal value:
    value
    inf
    -inf

solver:
prob.solve(solver=cvx.ECOS)      # SOCPs
prob.solve(solver=cvx.ECOS_BB)   # mixed-integer LPs and SOCPs
prob.solve(solver=cvx.CVXOPT)    # handle all problems (except mixed-integer programs)
prob.solve(solver=cvx.SCS)       # handle all problems (except mixed-integer programs)
prob.solve(solver=cvx.GLPK)      # linear programming solver
prob.solve(solver=cvx.GLPK_MI)
prob.solve(solver=cvx.GUROBI)
prob.solve(solver=cvx.MOSEK)     # quadratic and second-order cone programming
prob.solve(solver=cvx.ELEMENTAL)
prob.solve(solver=cvx.CBC)

'''
