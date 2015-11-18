# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:05:12 2015

@author: user
"""

import numpy as NP
from casadi import *
import matplotlib.pyplot as plt
from SolACE.Passer import Passer
from SolACE.MSstates import MSstates
class MultipleShootingProblem():
    def __init__(self):
        self.V = SX.zeros(0)
        self.vdesc = []
        self.pas = Passer()
#        self.vnumber = [[]]
        self.constraints = SX.zeros(0)
        self.cmin = NP.zeros(0)
        self.cmax = NP.zeros(0)
        self.t = SX.sym("t",1)
        self.sens = False
        self.check = False
    def setTimeRange(self,tstart,tend):
        self.trange = NP.zeros(2)
        self.trange[0] = tstart
        self.trange[1] = tend
    def addStates(self,n,xmin,xmax,xstart=None,method=['LagrangeColl',10],constr=True):
        X = MSstates(SX.sym("X",n))
        self.pas.addObj(X)
        X.initialize(self.pas,self.trange,method[1],xmin,xmax,xstart)
        return X