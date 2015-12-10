# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:00:23 2015

@author: user
"""
import numpy as NP
from casadi import *
from SolACE.caller import caller
import matplotlib.pyplot as plt

class MScontrols(caller):
    def initialize(self,obj,trange,ns,xmin,xmax,pstart,i):
        super(MScontrols,self).initialize(obj)
        self.nx1 = self.shape[0]
        self.nx2 = self.shape[1]
        self.xmin = xmin
        self.xmax = xmax
        self.trange = trange
        self.ns = ns
        self.ts = NP.linspace(trange[0],trange[1],self.ns+1)
        self.S = NP.zeros((self.nx1,self.nx2),dtype=object)
        self.np = self.nx1*(self.ns)   
        self.p = MX.sym("pu",self.np)
        self.teller = [i, i+self.np]
        return (i + self.np)
    def evalAtStart(self):
        xx = MX.zeros(self.nx1)
        x = self.p.reshape((self.nx1,self.ns))
        for i in range(self.nx1):
            xx[i] = x[i,0]
        return xx
    def evalAtControls(self):
        xx = []
        x = self.p.reshape((self.nx1,self.ns))
        for i in range(self.nx1):
            xx.append(x[i,:])
        return xx
    def evalAtEnd(self):
        xx = MX.zeros(self.nx1)
        x = self.p.reshape((self.nx1,self.ns))
        for i in range(self.nx1):
            xx[i] = x[i,self.ns-1]
        return xx
    def getBounds(self):
        pmin = NP.ones(self.np)
        pmax = NP.ones(self.np)
        for i in range(self.nx1):
            pmin[i::self.nx1] = self.xmin[i]*pmin[i::self.nx1]
            pmax[i::self.nx1] = self.xmax[i]*pmax[i::self.nx1]
        return pmin, pmax
    def setPVals(self,pin):
        self.pvals = pin
    def getPVals(self):
        return self.pvals
    def plot(self):
        x = self.pvals.reshape((self.ns,self.nx1))
        t = NP.linspace(self.trange[0],self.trange[1],self.ns)
        np = NP.ceil(NP.sqrt(self.nx1))
        for i in range(self.nx1):
            plt.subplot(np,np,i+1)
            plt.step(t,x[:,i])
            plt.xlabel('t')
            plt.ylabel('u'+str(i))
        plt.show()