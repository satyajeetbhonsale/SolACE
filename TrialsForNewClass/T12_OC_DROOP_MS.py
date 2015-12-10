# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:31:17 2015

@author: satyajeet
"""

import time

import numpy as NP
from SolACE.MultipleShootingProblem import MultipleShootingProblem
from SolACE.SolveMS import SolveMS
from pomodoro.solver.solver2 import Solver2
from pomodoro.discs.expression import Expression
from casadi import *

mu_m = 1.6# per day
Ks = 7.5  
rho_m = 0.10
Q0 = 0.04
Sin = 4.0
tph = 3600.0

x0 = NP.array([7.0,0.1,40])
x10 = NP.array([6.0,0.1,30])

prob = MultipleShootingProblem()
tend = prob.setTimeRange(0.0,7.0)

x = prob.addStates(3,[0.0,0.0,0.0],[1000.0,1.0,1000.0],[0.14,0.04,100.],method=['LagrangeStates',7])
u = prob.addControls(1,[0],[0.5],method=['PiecewiseConstant',7])
S = x[0]; Q = x[1]; X = x[2]
rho = rho_m*S/(S + Ks)
mu = mu_m*(1 - (Q0/Q))

rhs = SX.zeros(3)
rhs[0] = -rho*X - u[0]*(S - Sin)
rhs[1] = rho - mu*Q
rhs[2] = mu*X - u[0]*X
prob.addOde(x,u,rhs)
f = sum((x('control')[2]-100)**2)
prob.addObjective(f)
prob.setInitCondition(x0)
#x00 = prob.addFixedParameters(3,x10)
#prob.addConstraints(x(0)-x00.p,0.0)

solver = SolveMS(prob)#,solver='sqpmethod')
#x00.setPVals(x0)
solver.solve()
x.plot()

