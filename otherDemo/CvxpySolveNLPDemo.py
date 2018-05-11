#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-02 20:17
# @Abstract：
'''
problem:
 mimimize (x-2)^2+(y-1)^2
 subject to
         x^2-y<=0
         x+y<=2
'''

from cvxpy import *
import cvxpy as cvx

# create two scalar optimization Variable
x = Variable()
y = Variable()

# create two constraints
constraints=[ cvx.square(x)- y <= 0,
              x + y <= 2]

# optimization objective function
obj = Minimize(cvx.square(x-2)+cvx.square(y-1))
prob=Problem(obj,constraints)

# return the optimization value
prob.solve()
print("status:",prob.status)
print("optimization value:",prob.value)
print("optimization var:",x.value,y.value)

'''
status: optimal
optimization value: 0.9999999975731159
optimization var: 0.999999999955422 0.99999999920646
'''


