# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:12:18 2015

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:01:32 2015

@author: user
"""

import time
import numpy as NP
from pomodoro.problem.problem import Problem
from pomodoro.solver.solver2 import Solver2
from pomodoro.discs.expression import Expression
from casadi import *
import matplotlib.pyplot as plt
from SolACE.MPCproblem import MPCproblem
from SolACE.MPCsolve import MPCsolve
t = time.time()

prob = MPCproblem(50,t_control=10,total_plant=15)
x = prob.addControllerStates(3,[0.0,0.0,0.0],[15.0,30.0,1000.0,20.0])
w = prob.addControllerInputs(3,[-.5,-.5,0.0],[.5]*3)
#w.fix()
w.plot()
u = prob.addControllerInputs(1,[0.0],[1.0])

rhs = Expression(SX.zeros(3))
rhs[0] = x[0] - x[0]*x[1] + w[0]
rhs[1] = -x[1] + x[0]*x[1] + u*x[2]*x[1] + w[1]
rhs[2] = -x[2] + 0.50 + w[2]
m = SX.zeros(2,2)
m[0,0] = 2 + 1.414
m[1,0] = 1 + 1.414
m[0,1] = 1 + 1.414
m[1,1] = 1 + 1.414

prob.addControllerODEs(x,rhs)
f = sum((1 - x[0]('coll'))*(1 - x[0]('coll')) + (1 - x[1]('coll'))*(1-x[1]('coll')) + u('coll')*u('coll')) + (mul((mul((x[0:2](-1).T,m)),x[0:2](-1))))
prob.addControlObjective(f)
#prob.addConstraints(x(0),[1,1,0.50])
x1 = prob.addPlantStates(3)
u1 = prob.addPlantInputs(1)
rhs1 = SX.zeros(3)
rhs1[0] = x1[0] - x1[0]*x1[1]
rhs1[1] = -x1[1] + x1[0]*x1[1] + u1*x1[2]*x1[1]
rhs1[2] = -x1[2] + 0.50
prob.addPlantODEs(x1,rhs1)
x0 = NP.array([1,1,0.50])
prob.setInitCondition(x0)
solver = MPCsolve(prob,printlevel=0)
solver.solve()
solver.plotStates()#
solver.plotControls()#