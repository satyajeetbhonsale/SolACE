# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:40:37 2015

@author: Satyajeet
"""

import numpy as NP
from pomodoro.problem.problem import Problem
from pomodoro.solver.solver2 import Solver2
from pomodoro.discs.expression import Expression
from casadi import *
import matplotlib.pyplot as plt
from SolACE.MPCproblem import MPCproblem

#from SolACE.MPCsolve import MPCsolve

class EstimationRoutine(object):
    def __init__(self,solver):
        self.solverData = solver
    def getMeasurements(self,x):
        #print self.solverData.nmx
        #print self.solverData.MPC.nx
        I = NP.zeros((len(self.solverData.nmx),self.solverData.MPC.nx))
        xm = I.dot(x)
        j = 0
        for i in self.solverData.nmx:
            xm[j] = x[i] + self.solverData.std[j]*NP.random.randn()
            j+=1#print xm
        if self.solverData.Estimator is not None:            
            if self.solverData.Estimator.meth.upper() == "MHE":
                self.x_plant = x
        return xm

    def estimate(self,xm,P,xe,ue,dt):
        if self.solverData.Estimator is None:
            xe = xm
            P_out = 0.0
            return (xe,P_out)
        elif self.solverData.Estimator.meth.upper() == 'EKF':
            xe, P_out = self.solverData.Estimator.solveEKF(P,xe,ue,xm,dt)
            return (xe,P_out)
        elif self.solverData.Estimator.meth.upper() == 'UKF':
            xe, P_out = self.solverData.Estimator.solveUKF(P,xe,ue,xm,dt)
            return (xe,P_out)
        elif self.solverData.Estimator.meth.upper() == 'MHE':       
            if self.solverData.k < self.solverData.Estimator.Nh-1:
                if self.solverData.k == 0:
                    self.x_ARR = self.solverData.MPC.x0
                xe = self.x_plant
                P_out = P
            else:
                xe,P_out,self.x_ARR = self.solverData.Estimator.solveMHE(P,self.solverData.x_final,self.solverData.u_final,xm,self.x_ARR,self.solverData.x_measured,self.solverData.k,dt)              
            return(xe,P_out)