# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:51:38 2015

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
#####################

MPC = MPCproblem(300,total_plant=3000)

x = MPC.addControllerStates(4,[0.0,0.0,50.0,50.0,-1.0e06],[10.0,10.0,250.0,250.0,1.0e06],[2.14,1.09,114.2,112.9,0.0])
u = MPC.addControllerInputs(2,[3.0,-9000.0],[35.0,0.0])

rhs = Expression(SX.zeros(4))

cA = x[0]
cB = x[1]
theta = x[2]
thetaK = x[3]

k1 = k10*exp(E1/(273.15 +theta))
k2 = k20*exp(E2/(273.15 +theta))
k3 = k30*exp(E3/(273.15 +theta))
r = r0
X0 = x[0]-r[0]
X1 = x[1]-r[1]
X2 = x[2]-r[2]
X3 = x[3]-r[3]
U1 = u[0]-r[4]
U2 = u[1]-r[5]
rhs[0] = (1/tph)*(u[0]*(cA0-cA) - k1*cA - k3*cA*cA)
rhs[1] = (1/tph)* (- u[0]*cB + k1*cA - k2*cB)
rhs[2] = (1/tph)*(u[0]*(theta0-theta) - (1/(rho*Cp)) *(k1*cA*H1 + k2*cB*H2 + k3*cA*cA*H3)+(kw*AR/(rho*Cp*VR))*(thetaK -theta))
rhs[3] = (1/tph)*((1/(mK*CPK))*(u[1] + kw*AR*(theta-thetaK)))

MPC.addControllerODEs(x,rhs)
#MPC.addMeasurementNoise([0,1,2,3],[0.5,0.5,0.5,0.5])
#MPC.addControllerConstraints(x(0),x0)
f = sum(S00*X0('coll')*X0('coll') + S11*X1('coll')*X1('coll') + S22*X2('coll')*X2('coll') + S33*X3('coll')*X3('coll') + S44*U1('coll')*U1('coll') + S55*U2('coll')*U2('coll'))
MPC.addControlObjective(f)

x1 = MPC.addPlantStates(4)
u1 = MPC.addPlantInputs(2)
cA = x1[0]
cB = x1[1]
theta = x1[2]
thetaK = x1[3]

k1 = k10*exp(E1/(273.15 +theta))
k2 = k20*exp(E2/(273.15 +theta))
k3 = k30*exp(E3/(273.15 +theta))
rhs1 = SX.zeros(4)
rhs1[0] = (1/tph)*(u1[0]*(cA0-cA) - k1*cA - k3*cA*cA)
rhs1[1] = (1/tph)* (- u1[0]*cB + k1*cA - k2*cB)
rhs1[2] = (1/tph)*(u1[0]*(theta0-theta) - (1/(rho*Cp)) *(k1*cA*H1 + k2*cB*H2 + k3*cA*cA*H3)+(kw*AR/(rho*Cp*VR))*(thetaK -theta))
rhs1[3] = (1/tph)*((1/(mK*CPK))*(u1[1] + kw*AR*(theta-thetaK)))
MPC.addPlantODEs(x1,rhs1)
MPC.setInitCondition(x0)
MPC.addMeasurementNoise([0,3],[0.05,5])
#### ESTIMATOR
sigma_y = NP.array([0.05,5])
V = (sigma_y**2) * NP.eye(2)
sigma_x0 = 0.05
P00 = (sigma_x0**2) * NP.eye(4)
est = Estimator('ukf',[0,3],V,P00)

x2 = est.addStates(4,[0.0,0.0,50.0,50.0],[10.0,10.0,250.0,250.0],[2.14,1.09,114.2,112.9,0.0])
u2 = est.addControls(2,[3.0,-9000.0],[35.0,0.0])

cA = x2[0]
cB = x2[1]
theta = x2[2]
thetaK = x2[3]

k1 = k10*exp(E1/(273.15 +theta))
k2 = k20*exp(E2/(273.15 +theta))
k3 = k30*exp(E3/(273.15 +theta))
rhs2 = SX.zeros(4)
rhs2[0] = (1/tph)*(u2[0]*(cA0-cA) - k1*cA - k3*cA*cA)
rhs2[1] = (1/tph)* (- u2[0]*cB + k1*cA - k2*cB)
rhs2[2] = (1/tph)*(u2[0]*(theta0-theta) - (1/(rho*Cp)) *(k1*cA*H1 + k2*cB*H2 + k3*cA*cA*H3)+(kw*AR/(rho*Cp*VR))*(thetaK -theta))
rhs2[3] = (1/tph)*((1/(mK*CPK))*(u2[1] + kw*AR*(theta-thetaK)))
yE = SX.zeros(2)
yE[0] = x2[0]; yE[1] = x2[3]
est.addMeasurements(yE)
est.addEstimatorODEs(x2,rhs2)
solver = MPCsolve(MPC,Estimator = est)
solver.solve()
solver.plotStates()#
solver.plotControls()#