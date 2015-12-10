# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:16:26 2015

@author: Satyajeet
"""

import time
import numpy as NP
from pomodoro.problem.problem import Problem
from pomodoro.solver.solver2 import Solver2
from pomodoro.discs.expression import Expression
from casadi import *
import matplotlib.pyplot as plt
#from mpc.solveMPC import solveMPC
from SolACE.MPCproblem import MPCproblem
from SolACE.MPCsolve import MPCsolve
from SolACE.Estimator import Estimator

t = time.time()

Fa = 1.83
W = 2104.669

MPC = MPCproblem(50,total_plant=1500.0)

x = MPC.addControllerStates(6,[0.0,0.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0])
u = MPC.addControllerInputs(2,[2.0,50.0],[10,150])
k1 = 1.6599e6*exp(-6666.37/(273.15+u[1]))
k2 = 7.2117e8*exp(-8333.3/(273.15+u[1]))
k3 = 2.6745e12*exp(-11111.0/(273.15+u[1]))
#Fa = prob1.addFixedParameters(1)
#Fa.setPVals(NP.array([2.4]))
r1 = k1*x[0]*x[1]*W
r2 = k2*x[1]*x[2]*W
r3 = k3*x[2]*x[5]*W

rhs = Expression(SX.zeros(6))
rhs[0] = Fa - (Fa + u[0])*x[0] - r1
rhs[1] = u[0] - (Fa + u[0])*x[1] - r1 - r2
rhs[2] = -(Fa + u[0])*x[2] + 2*r1 - 2*r2 - r3
rhs[3] = -(Fa + u[0])*x[3] + r2
rhs[4] = -(Fa + u[0])*x[4] + 1.5*r3
rhs[5] = -(Fa + u[0])*x[5] + r2 - 0.5*r3
MPC.addControllerODEs(x,rhs)

f1 = -sum(5554.1*(Fa + u[0]('coll'))*x[5]('coll') + 125.91*(Fa + u[0]('coll'))*x[3]('coll') - 370.3*Fa - 555.42*u[0]('coll'))
MPC.addControlObjective(f1)

x1 = MPC.addPlantStates(6)
u1 = MPC.addPlantInputs(2)

k1 = 1.6599e6*exp(-6666.37/(273.15+u1[1]))
k2 = 7.2117e8*exp(-8333.3/(273.15+u1[1]))
k3 = 2.6745e12*exp(-11111.0/(273.15+u1[1]))
#Fa = prob1.addFixedParameters(1)
#Fa.setPVals(NP.array([2.4]))
r1 = k1*x1[0]*x1[1]*W
r2 = k2*x1[1]*x1[2]*W
r3 = k3*x1[2]*x1[5]*W

rhs1 = SX.zeros(6)
rhs1[0] = Fa - (Fa + u1[0])*x1[0] - r1
rhs1[1] = u1[0] - (Fa + u1[0])*x1[1] - r1 - r2
rhs1[2] = -(Fa + u1[0])*x1[2] + 2*r1 - 2*r2 - r3
rhs1[3] = -(Fa + u1[0])*x1[3] + r2
rhs1[4] = -(Fa + u1[0])*x1[4] + 1.5*r3
rhs1[5] = -(Fa + u1[0])*x1[5] + r2 - 0.5*r3
MPC.addPlantODEs(x1,rhs1)
x0 = NP.array([0.0,0.0,0.0,0.0,0.0,0.0])
MPC.setInitCondition(x0)

solver = MPCsolve(MPC,printlevel=0)

solver.solve()
solver.plotStates()#
solver.plotControls()#
