# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:46:45 2015

@author: user
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

mu_m = 1.6# per day
Ks = 7.5  
rho_m = 0.10
Q0 = 0.04
Sin = 4.0
MPC = MPCproblem(14,total_plant=14.)

x = MPC.addControllerStates(3,[0.0,0.0,0.0],[1000.0,1.0,1000.0,])
u = MPC.addControllerInputs(1,[0],[0.5])
S = x[0]; Q = x[1]; X = x[2]
rho = rho_m*S/(S + Ks)
mu = mu_m*(1 - (Q0/Q))

rhs = Expression(SX.zeros(3))
rhs[0] = -rho*X - u[0]*(S - Sin)
rhs[1] = rho - mu*Q
rhs[2] = mu*X - u[0]*X
MPC.addControllerODEs(x,u,rhs)

f = sum((X('coll')-100)*(X('coll')-100))
MPC.addControlObjective(f)

x1 = MPC.addPlantStates(3)
u1 = MPC.addPlantInputs(1)
S = x1[0]; Q = x1[1]; X = x1[2]
mu_m = 1.2# per day
Ks = 6.75 
rho_m = 0.125
rho = rho_m*S/(S + Ks)
mu = mu_m*(1 - (Q0/Q))

rhs1 = SX.zeros(3)
rhs1[0] = -rho*X - u1[0]*(S - Sin)
rhs1[1] = rho - mu*Q
rhs1[2] = mu*X - u1[0]*X
MPC.addPlantODEs(x1,rhs1)
x0 = NP.array([7.0,0.1,40.0])
MPC.setInitCondition(x0)

solver = MPCsolve(MPC,printlevel=0)
solver.solve()
solver.plotStates()#
solver.plotControls()#