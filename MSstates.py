# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:42:25 2015

@author: user
"""

import numpy as NP
from casadi import *
from SolACE.caller import caller
class MSstates(caller):
    def initialize(self,obj,trange,ns,xmin,xmax,pstart=None):
        super(MSstates,self).initialize(obj)
        self.nx1 = self.shape[0]
        self.nx2 = self.shape[1]
        self.xmin = xmin
        self.xmax = xmax
        self.trange = trange
        self.ns = ns
        self.ts = NP.linspace(trange[0],trange[1],self.ns+1)
        self.S = NP.zeros((self.nx1,self.nx2),dtype=object)
        self.np = self.nx1*(self.ns + 1)   
        self.p = MX.sym("px",self.np)
    def x(self,*args):
        xx = SX.zeros((self.nx1,self.nx2))
        if len(args) == 2:
            for i in range(self.nx1):
                for j in range(self.nx2):
                    xx[i,j] = self.S[i,j].xin(args[0],args[1])
        else:
            for i in range(self.nx1):
                for j in range(self.nx2):
                    xx[i,j] = self.S[i,j].xt(args[0])
        return xx

    def evalAtStart(self):
        return self.p
#        t = SX.sym("t",1)
#        self.tfun = SXFunction([t],[self.x(t)])
#        self.tfun.init()
##        self.init(pstart)
#        self.xx = SX(self)

