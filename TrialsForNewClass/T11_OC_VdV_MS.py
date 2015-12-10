
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 17:52:31 2014

@author: satyajeet
"""

import time

#import pomodoro
import numpy as NP
#import pomodoro
from SolACE.MultipleShootingProblem import MultipleShootingProblem
from SolACE.SolveMS import SolveMS
from pomodoro.solver.solver2 import Solver2
from pomodoro.discs.expression import Expression
from casadi import *

t = time.time()

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

prob = MultipleShootingProblem()
tend = prob.setTimeRange(0.0,3000.0)

x = prob.addStates(4,[0.0,0.0,50.0,50.0],[10.0,10.0,250.0,250.0],[2.14,1.09,114.2,112.9],method=['LagrangeStates',50])

#prob.addConstraints(x(0),x0)

#theta0 = prob.addParameters(1,100.0,110.0)
u = prob.addControls(2,[3.0,-9000.0],[35.0,0.0],method=['PiecewiseConstant',50])
#u.load('control_in_fix.txt')
#u.plot()
#theta0 = prob.addControls(1,[90.0],[120.0],method=['PiecewiseConstant',50])
#theta0.load(NP.random.normal(104.9,5,50))
#theta0.fix()
rhs = SX.zeros(4)

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
#rhs[4] = S00*X0*X0 + S11*X1*X1 + S22*X2*X2 + S33*X3*X3 + S44*U1*U1 + S55*U2*U2
prob.addOde(x,u,rhs)
r0 = NP.array([2.14, 1.09, 114.2, 112.9, 14.19, -1113.5])
f = sum(S00*(x('control')[0]-r0[0])**2+S11*(x('control')[1]-r0[1])**2 +S22*(x('control')[2]-r0[2])**2+S33*(x('control')[3]-r0[3])**2 )+sum(S44*(u('control')[0]-r0[4])**2 + S55*(u('control')[1]-r0[5])**2)#print x0
prob.addObjective(f)
prob.setInitCondition(x0)
solver = SolveMS(prob)
solver.solve()
x.plot()

#solver.solve()
##f = x[4](-1)
#f = sum(S00*X0('coll')*X0('coll') + S11*X1('coll')*X1('coll') + S22*X2('coll')*X2('coll') + S33*X3('coll')*X3('coll') + S44*U1('coll')*U1('coll') + S55*U2('coll')*U2('coll'))
##f = 0.0
#prob.addObjective(f)
#
#solver = Solver2(prob,printlevel=5,max_iter=10000)
##plist = NP.array([[1.1,1.2,1.3,1.4]])
#print time.time()-t
#solver.solve()
#print time.time()-t

#tv,uv = u.plot()
#tv, xv= x.plot()
#NP.savetxt('state_input',xv)
#NP.savetxt('control_input',uv)
#NP.savetxt('time_input',tv)
#z = x('control',True) + 0.05*NP.random.randn(4,50)
#
#u.save('control_in_fix.txt')
#a = x0.T
#c = NP.column_stack((a,z))
#t = NP.linspace(0,3000,50)
#NP.savetxt('state_MHE.txt',c)
#tv,xv = prob.savePlottedStates(N=50)
#tv,uv = prob.savePlottedControls(N=50) 
#NP.savetxt('time1.txt',tv)
#NP.savetxt('states1.txt',xv)
#NP.savetxt('controls1.txt',uv)
#
#
#
#print type(prob)
#plant = prob
#print type(plant)
#solver = Solver2(plant,printlevel=5,max_iter=10000)
#plant.addObjective(0.0)
##plist = NP.array([[1.1,1.2,1.3,1.4]])
#print time.time()-t
#solver.solve()
#print time.time()-t