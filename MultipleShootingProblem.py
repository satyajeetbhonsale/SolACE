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
from SolACE.MScontrols import MScontrols
from SolACE.caller import caller
from SolACE.FixedParameter import FixedParameters
class MultipleShootingProblem():
    def __init__(self):
        self.V = MX.zeros(0)
        self.vdesc = []
        self.pas = Passer()
        self.lbx = []
        self.ubx = []
        self.xIN = False
#        self.vnumber = [[]]
        self.constraints = MX.zeros(0)
        self.cmin = NP.zeros(0)
        self.cmax = NP.zeros(0)
        self.t = SX.sym("t",1)
        self.sens = False
        self.check = False
        self.nvar = 0
        self.i = 0
    def setTimeRange(self,tstart,tend):
        self.trange = NP.zeros(2)
        self.trange[0] = tstart
        self.trange[1] = tend
    def addStates(self,n,xmin,xmax,xstart=None,method=['LagrangeColl',50],constr=True):
        X = MSstates(SX.sym("X",n))
        self.pas.addObj(X)        
        self.nvar = self.nvar + n*(method[1]+1)
        self.i = X.initialize(self.pas,self.trange,method[1],xmin,xmax,xstart,self.i)
        self.n_dis = method[1]       
        self.addVars(X.p,"X")
        return X
    def addControls(self,n,umin,umax,ustart=None,method=['PiecewiseConstant',50]):
        U = MScontrols(SX.sym("U",n))
        self.pas.addObj(U)
        self.nvar = self.nvar + n*(method[1])
        self.i = U.initialize(self.pas,self.trange,method[1],umin,umax,ustart,self.i)
        self.addVars(U.p,"U")        
        return U
    def addOde(self,x,u,rhs):
        f = SXFunction(daeIn(x=x,p=u,t=self.t),daeOut(ode=rhs))
        self.I = Integrator('cvodes',f)
        self.I.setOption('tf',self.trange[1]/self.n_dis)
        self.I.init()
        X = x.p
        X = X.reshape((x.shape[0],x.p.shape[0]/x.shape[0]))
        U = u.p
        U = U.reshape((u.shape[0],u.p.shape[0]/u.shape[0]))
        for k in range(self.n_dis):
            Xk = X[:,k]
            Xk_next = X[:,k+1]
            Uk = U[:,k]
            Xk_end, = integratorOut(self.I(integratorIn(x0=Xk,p=Uk)),"xf")
            c = Xk_next - Xk_end
            self.addConstraints(c,0.0,0.0)
    def addObjective(self,f):
        self.objective = f            
    def initNLP(self):
        nv = self.i
        V = MX.sym('V',nv)
    def addVars(self,v,desc):
        vold = self.V.size()
        self.V.append(vec(v))
        vnew = self.V.size()
        self.vdesc.append([desc,vold,vnew])
    def addConstraints(self,c,cmin,cmax=None):
        ab = c#        
        c = vec(c)
        a = c##
        self.constraints.append(c)
        n = c.shape[0]
        if cmax is None:
            if isinstance(cmin,NP.float):
                cmin = cmin*NP.ones(n)
            cmax = cmin
        else:
            if isinstance(cmin,NP.float):
                cmin = cmin*NP.ones(n)
            if isinstance(cmax,NP.float):
                cmax = cmax*NP.ones(n)
        self.cmin = NP.append(self.cmin,cmin)
        self.cmax = NP.append(self.cmax,cmax)
    def getBounds(self):
        vmin = NP.zeros(self.V.shape[0])
        vmax = NP.zeros(self.V.shape[0])
        for i in range(len(self.vdesc)):
            low = self.vdesc[i][1]
            up = self.vdesc[i][2]
            vmin[low:up],vmax[low:up] = self.pas.obj[i].getBounds()
            if self.vdesc[i][0] == 'X':
                if self.xIN:
                    vmin[low:low+self.pas.obj[i].shape[0]] = vmax[low:low+self.pas.obj[i].shape[0]] = self.x_init
        return vmin,vmax
    def setInitCondition(self,x):
        self.xIN = True
        self.x_init = x
    def saveOptResults(self,res):
        for i in range(len(self.vdesc)):
            low = self.vdesc[i][1]
            up = self.vdesc[i][2]
            pvals = res[low:up]
            self.pas.obj[i].setPVals(pvals)
    def makeExpression(self,f):
        out = caller(f)
        out.initialize(self.pas)
        return out
    def addFixedParameters(self,n,pstart=None):
        F = FixedParameters(SX.sym("F",n))
        self.pas.addObj(F)
        if pstart is None:
            pstart = NP.zeros(n)
        F.initialize(self.pas,pstart)
        self.addVars(F.p,"F")
        return F