# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:32:14 2015

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
import scipy.linalg

class Estimator(object):
    def __init__(self,meth,nmx,V,P00):
        self.meth = meth # Method: ekf, ukf, mhe
        self.V = V # Variance of the Noise!!!
        self.P00 = P00
        self.te = SX.sym('te')
        if self.meth.upper() == 'MHE':
            self.MHE = Problem()
    def addMeasurements(self,yEST):
        self.HFunc = self.calcJacobian(yEST,self.xEST)
        self.YFun = SXFunction([self.xEST,self.uEST],[yEST])
        self.YFun.init()
        self.yEST = yEST
        self.nmx = self.yEST.shape[0]

    def addStates(self,n,xmin,xmax,xstart=None,method=['LagrangeColl',10],constr=True):
        if self.meth.upper() == 'EKF':
            self.xEST = SX.sym('xe',n)
        elif self.meth.upper() == 'UKF':
            self.xEST = SX.sym('xe',n)
        elif self.meth.upper() == 'MHE':
            self.xEST = self.MHE.addStates(n,xmin,xmax,xstart=None,method=['LagrangeColl',self.Nh],constr=True)
        j = 0
        self.nx = n
        return self.xEST

    def addControls(self,n,umin,umax,ustart=None,method=['PiecewiseConstant',10]):
        if self.meth.upper() == 'EKF':
            self.uEST = SX.sym('ue',n)
        elif self.meth.upper() == 'UKF':
            self.uEST = SX.sym('ue',n)
        elif self.meth.upper() == 'MHE':
            self.uEST = self.MHE.addControls(n,umin,umax,ustart=None,method=['PiecewiseConstant',self.Nh])
            self.uEST.fix()
        return self.uEST

    def defineHorizon(self,Nh,Th):
        if self.meth.upper() != 'MHE':
            raise IOError('Horizon Length Defined Only For MHE')
        self.Nh = Nh
        self.Th = Th
        self.MHE.setTimeRange(0.0,Nh*Th)

    def addEstimatorODEs(self,x,rhs):
        if self.meth.upper() == 'EKF' or self.meth.upper() == 'UKF': 
            self.est_model = SXFunction(daeIn(x=self.xEST,p=self.uEST,t=self.te),daeOut(ode=rhs))
            self.est_model.setOption('name','Estimator')
        elif self.meth.upper() == 'MHE':
            self.MHE.addOde(self.xEST,rhs) ## INCOMPLETE
            self.RHSFunc = SXFunction([self.xEST,self.uEST],[rhs])
            self.RHSFunc.init()
            self.setupMHE()
#        RHSFunc = SXFunction([xEST,uEST],[rhs])
        self.JFunc = self.calcJacobian(rhs,self.xEST)

    def calcJacobian(self,fx,x):
        J = jacobian(fx,x)
        JFunc = SXFunction([self.xEST,self.uEST],[J])
        JFunc.init()
        return JFunc

    def evalFunction(self,f,xval,uval):
        f.setInput(xval,0)
        f.setInput(uval,1)
        f.evaluate()
        dFdx = NP.array(f.output())
        return dFdx

    def InitIntegrate(self,f,deltat_int):
        I = Integrator('cvodes',f)
        I.setOption('tf',deltat_int)
        I.setOption('abstol',1e-4)
        I.setOption('reltol',1e-4)
        I.init()
        return I

    def getMeasurements(self,xp,up):
        yo = NP.array(self.evalFunction(self.YFun,xp,up))
        return yo

    def setupMHE(self):
        sigma_w = 0.00005 # Noise Charaterization
        self.W = 1/sigma_w * NP.eye(self.nx)
        self.xARR0 = self.MHE.addFixedParameters(self.nx)
        wMHE = self.MHE.addControls(self.nx,[-5]*self.nx,[5]*self.nx,method=['PiecewiseConstant',self.Nh])
        self.ym = self.MHE.addFixedParameters((self.Nh+1)*self.nmx)
        ymM = reshape(self.ym.T,self.Nh+1,self.nmx).T
        self.Psym = self.MHE.addFixedParameters(self.nx*self.nx)
        Pm = reshape(self.Psym.T,self.nx,self.nx).T
        cont_y = self.yEST('control')
        ff = 0.0
        ff += mul([(self.xEST(0)-self.xARR0).T,Pm.T,Pm,(self.xEST(0)-self.xARR0)])
        for i in range(self.Nh):
            ff += mul([(cont_y[:,i] - ymM[:,i+1]).T,self.V.T,self.V,(cont_y[:,i] - ymM[:,i+1])])
        cont_w = wMHE('control')
        ff += mul([wMHE(0).T,self.W.T,self.W,wMHE(0)])
        for i in range(self.Nh):
            ff += mul([(cont_w[:,i]).T,self.W.T,self.W,cont_w[:,i]])
        self.MHE.addObjective(ff)
        self.solMHE = Solver2(self.MHE,tol=1e-6)

    def solveEKF(self,P,xep,ue,xm,dt):
        I = self.InitIntegrate(self.est_model,dt) #dt is t_plant
        #print ue
        I.setInput(ue,'p')
        I.setInput(xep,'x0')
        I.evaluate()
        xpred = NP.array(I.getOutput('xf')).flatten()
        Hx = self.evalFunction(self.HFunc,xep,ue)
        #print Hx        
        ypred = Hx.dot(xpred) #Hx is the jacobian of measurement! For now one of the state!
        Jx = self.evalFunction(self.JFunc,xep,ue)
#        print Jx
#        print type(Jx)
#        print type(P)
        Ppred = Jx.dot(P.reshape(self.nx,self.nx)).dot(Jx.T)
        yout = xm - ypred
        S = Hx.dot(Ppred).dot(Hx.T) + self.V
        Kk = Ppred.dot(Hx.T).dot(scipy.linalg.inv(S))
        estimated_x = xpred + Kk.dot(yout)
        Pout = (NP.eye(4) - Kk.dot(Hx)).dot(Ppred)
        #print Pout    
        estimated_P = NP.reshape(Pout,4*4)
        return (estimated_x, estimated_P)
    def solveUKF(self,P,xep,ue,xm,dt):
        kappa = 3 - self.nx
        I = self.InitIntegrate(self.est_model,dt)
        sigma = scipy.linalg.cholesky((self.nx+kappa)*NP.reshape(P,(self.nx,self.nx)))
        xlist = xep
        nlist = 2*self.nx + 1
        for i in range(self.nx):
            xlist = NP.vstack((xlist,xep + sigma[:,i]))
        for i in range(self.nx):
            xlist = NP.vstack((xlist,xep - sigma[:,i]))
        xout = NP.zeros((nlist,self.nx))
        yout = NP.zeros((nlist,self.nmx))
        I.setInput(ue,'p')
        for i in range(nlist):
            I.setInput(xlist[i,:],'x0')
            I.evaluate()
            xout[i,:] = NP.array(I.getOutput('xf')).flatten()
            yout[i,:] = self.evalFunction(self.YFun,xout[i,:],ue).flatten()
        xpred = (1.0/(self.nx+kappa))*kappa*xout[0,:]
        ypred = (1.0/(self.nx+kappa))*kappa*yout[0,:]
        for i in range(1,nlist):
            xpred+= (1.0/(self.nx+kappa))*0.5*xout[i,:]
            ypred+= (1.0/(self.nx+kappa))*0.5*yout[i,:]
        Ppred = 0
        Pypred = 0
        Pxypred = 0
        for j in range(1,nlist):
            Ppred = Ppred + (1.0/(self.nx+kappa))*0.5*NP.outer((xout[j,:]-xpred),(xout[j,:]-xpred))
            Pypred =Pypred + (1.0/(self.nx+kappa))*0.5*NP.outer((yout[j,:]-ypred),(yout[j,:]-ypred))
            Pxypred =Pxypred + (1.0/(self.nx+kappa))*0.5*NP.outer((xout[j,:]-xpred),(yout[j,:]-ypred))
        Pinnov = Pypred + self.V
        W = Pxypred.dot(scipy.linalg.inv(Pinnov))
        estimated_x = xpred + W.dot(xm-ypred).flatten()
        Pout = Ppred - W.dot(Pinnov).dot(W.T)
        estimated_P = (NP.reshape(Pout,self.nx*self.nx))
        return (estimated_x,estimated_P)
    def solveMHE(self,P,xep,ue,xm,xARR,y_out,i,dt):
        ymin = NP.reshape(y_out[:,i-self.Nh+1:i+2],self.nmx*(self.Nh+1))
        self.ym.setPVals(ymin)
        self.uEST.setPVals(ue[:,i-self.Nh+1:i+1].flatten())
        self.xARR0.setPVals(xARR.flatten())
        self.Psym.setPVals(P)
        self.solMHE.solve()
        estimated_xMHE = self.xEST(-1,True).flatten()


        if i >= 2*self.Nh-1:
            intin = xep[:,i-self.Nh+1]
        else:
            intin = xARR
        rhsf = self.evalFunction(self.RHSFunc,intin,ue[:,i-self.Nh+1]).flatten()
        dFdx = self.evalFunction(self.JFunc,intin,ue[:,i-self.Nh+1])
        xtilda = rhsf - dFdx.dot(intin.flatten())
        Xfuitg = dFdx
        Xfuitg += NP.eye(4)/dt    #why the eye and dt??
        Hx = self.evalFunction(self.HFunc,intin,ue[:,i-self.Nh+1])
        
        A1 = NP.hstack((NP.reshape(P,(self.nx,self.nx)),NP.zeros((self.nx,self.nx))))
        A2 = NP.hstack((-self.V.dot(Hx),NP.zeros((self.nmx,self.nx))))
        A3 = NP.hstack((-self.W.dot(Xfuitg),self.W/dt)) # why the dt??
        A = NP.vstack((A1,A2,A3))    
        b1 = (NP.reshape(P,(self.nx,self.nx))).dot(xARR)
        b2 = (-self.V.dot(xm))
        b3 = self.W.dot(xtilda)
        B = NP.hstack((b1,b2,b3))
        Q,R = NP.linalg.qr(A)
        R2 = R[self.nx:,self.nx:]
        Qb = Q.T.dot(B)
        Qb2 = Qb[self.nx:]
        result = NP.linalg.solve(R2,Qb2)
        estimated_P = NP.reshape(R2,4*4)
        xARR = result
        estimated_x = estimated_xMHE
        return (estimated_x,estimated_P,xARR)
#self.y