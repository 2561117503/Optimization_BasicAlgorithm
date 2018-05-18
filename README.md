# Optimization Basic Algorithm
## Introduction: 
### Save my optimization code demo: convex optimization; numerical optimization algorithm<br>

note: code based on cvxpy package<br>
my notes of optimization:<br>
> [numerical optimization](http://yangenneng.cn/tags/numerical-optimization/)<br>

> [convex optimization](http://yangenneng.cn/tags/convex-optimization/)<br>

## Project struct
> Linear Search Methods: 
>> Steepest Descent Method <br>
>> Newton Method<br> 
>> Quasi-Newton Method<br> 
>> Conjugate Gradient Method<br> 
>> Matrix Util Method<br> 

> Large-Scale Unconstrained Optimization:
>> Inexact Newton method <br>

## Algorithm list
### Linear Search Methods :
> StepLength:
>> { Backtracking Line Search } Algorithm: [BacktrackingLineSearch.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/StepSize/BacktrackingLineSearch.py) <br>
>> { Interpolation: Quadratic; Cubic} Algorithm: [Interpolation.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/StepSize/Interpolation.py) <br>
>> { Zoom} Algorithm: [Zoom.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/StepSize/Zoom.py) <br>
>> { Wolfe Line Search} Algorithm: [WolfeLineSearch.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/StepSize/WolfeLineSearch.py) <br>

> SteepestDescent:
>> { Gradient Descent Method  } Algorithm: [GradientDescentMethod.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/SteepestDescent/GradientDescentMethod.py) <br>

> Newton:
>> { Newton Method  } Algorithm: [NewtonMethod.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/Newton/NewtonMethod.py) <br>
>> { Cholesky with Added Multiple of the Identity  } Algorithm: [AddedMultipleOfTheIdentity.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/Newton/AddedMultipleOfTheIdentity.py) <br>

> Quasi-Newton:
>> { DFP Method  } Algorithm: [DFP.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/Quasi-Newton/DFP.py) <br>
>> { BFGS Method  } Algorithm: [BFGS.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/Quasi-Newton/BFGS.py) <br>

> ConjugateGradient:
>> { Conjugate Gradient Preliminary  } Algorithm: [CG_Preliminary.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/ConjugateGradient/CG_Preliminary.py) <br>
>> { Conjugate Gradient  } Algorithm: [CG.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/ConjugateGradient/CG.py) <br>
>> { Preconditioned Conjugate Gradient  } Algorithm: [Preconditioned_CG.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/ConjugateGradient/Preconditioned_CG.py) <br>
>> { Fletcher-Reeves methods  } Algorithm: [FR.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/ConjugateGradient/FR.py) <br>

> MatrixUtil:
>> { Cholesky Factorization: LDL^T} Algorithm: [Cholesky_LDL.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/MatrixUtil/Cholesky_LDL.py) <br>

### Large-Scale Unconstrained Optimization:
> Inexact Newton method
>> { Line Search Newton-CG  } Algorithm: [LineSearchNewton_CG.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/LinearSearchMethods/Quasi-Newton/LineSearchNewton_CG.py) <br>


### Other demo :
> using CvxOpt or Cvxpy package demo:
>> { CvxOpt Solve LP } Demo: [CvxOptSolveLPDemo.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/otherDemo/CvxOptSolveLPDemo.py) <br>
>> { Cvxpy Solve LP } Demo: [CvxpySolveLPDemo.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/otherDemo/CvxpySolveLPDemo.py) <br>
>> { Cvxpy Solve NLP } Demo: [CvxpySolveNLPDemo.py](https://github.com/YEN-GitHub/Optimization_BasicAlgorithm/tree/master/otherDemo/CvxpySolveNLPDemo.py) <br>

## References
> Jorge Nocedal and Stephen J.Wright : `Numerical optimization`  Second Edition

> Stephen Boyd and Lieven vandenberghe: `Convex optimization` <br>

