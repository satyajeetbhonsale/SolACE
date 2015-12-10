# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:56:11 2015

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

###########
k10 = 1.287E12
k20 = 1.287e12
k30 = 9.043e09
E1 = -9758.3
E2 = -9758.3
E3 = -8560.0

H1 = 4.2
H2 = -11.0
H3 = -41.85
rho = 0.9342
Cp = 3.01
kw = 4032.0
AR = 0.215
VR = 10.0
mK = 5.0
CPK = 2.0

cA0 = 5.6
theta0 = 104.9
S00 = 0.2
S11 = 1.0
S22 = 0.5
S33 = 0.2
S44 = 0.5000
S55 = 0.0000005
r0 = NP.array([2.14, 1.09, 114.2, 112.9, 14.19, -1113.5])
r1 = NP.array([2.9805,0.9612,106.0,100.75,18.038,-4565.88])
r2 = NP.array([3.5176,0.7395,87.0,79.8,8.256,-6239.33])

tph = 3600.0

x0 = NP.array([1.0,0.5,100.0,100.0])
#########
MPC = MPCproblem(50,total_plant=3000,dis='MultipleShooting')
x1 = MPC.addControllerStates(4,[0.0,0.0,50.0,50.0,-1.0e06],[10.0,10.0,250.0,250.0,1.0e06],[2.14,1.09,114.2,112.9,0.0])

u1 = MPC.addControllerInputs(2,[3.0,-9000.0],[35.0,0.0])

cA = x1[0]
cB = x1[1]
theta = x1[2]
thetaK = x1[3]
k1 = k10*exp(E1/(273.15 +theta))
k2 = k20*exp(E2/(273.15 +theta))
k3 = k30*exp(E3/(273.15 +theta))
rhs = SX.zeros(4)
rhs[0] = (1/tph)*(u1[0]*(cA0-cA) - k1*cA - k3*cA*cA)
rhs[1] = (1/tph)* (- u1[0]*cB + k1*cA - k2*cB)
rhs[2] = (1/tph)*(u1[0]*(theta0-theta) - (1/(rho*Cp)) *(k1*cA*H1 + k2*cB*H2 + k3*cA*cA*H3)+(kw*AR/(rho*Cp*VR))*(thetaK -theta))
rhs[3] = (1/tph)*((1/(mK*CPK))*(u1[1] + kw*AR*(theta-thetaK)))
MPC.addControllerODEs(x1,u1,rhs)
