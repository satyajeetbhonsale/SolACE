# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:19:37 2015

@author: Satyajeet
"""
import numpy as NP
from pomodoro.problem.problem import Problem
from pomodoro.solver.solver2 import Solver2
from pomodoro.discs.expression import Expression
from casadi import *
import matplotlib.pyplot as plt
from SolACE.EstimationRoutine import EstimationRoutine

class MPCsolve(object):
    def __init__(self,MPC,Estimator=None,printlevel=0,max_iter=4000,save=None):
        self.MPC = MPC
        self.Estimator = Estimator
        self.printlevel = printlevel
        self.max_iter = max_iter
        if self.MPC.nmx is None:
            self.nmx = list(xrange(self.MPC.nx))
            self.std = self.MPC.nx*[0]
        else:
            self.nmx = self.MPC.nmx
            self.std = self.MPC.std
        self.x_final = NP.zeros((self.MPC.nx,MPC.N_sampling+1))
        self.x_measured = NP.zeros((len(self.nmx),MPC.N_sampling+1))
        self.u_final = NP.zeros((MPC.nu,MPC.N_sampling))     
        self.save = save
        if save is not None and isinstance(save,str) is False:
            raise IOError('File Name to save should be a string')
    def solveOCP(self,prob,printlevel=0,max_iter=4000):
        solver = Solver2(prob,printlevel=self.printlevel,max_iter=self.max_iter)
        solver.solve()
    def InitIntegrate(self,f,tfinal,Nintegrate = 1):
        self.Nintegrate = Nintegrate        
        deltat_int = tfinal/Nintegrate
        I = Integrator('cvodes',f)
        I.setOption('tf',deltat_int)
        I.setOption('abstol',1e-4)
        I.setOption('reltol',1e-4)
        I.init()
        return I
    

    def solve(self):
        I = self.InitIntegrate(self.MPC.plant_model,self.MPC.t_plant)
        if self.Estimator is not None:
            est_P = NP.reshape(self.Estimator.P00,self.MPC.nx*self.MPC.nx)
        else:
            est_P = 0.0
        self.x_final[:,0] = self.MPC.x0        
        x_plant = self.MPC.x0        
        self.est = EstimationRoutine(self)
        for self.k in range(self.MPC.N_sampling):
            i = self.k            
            print i
            print self.MPC.N_sampling
            self.MPC.x_con.init(x_plant)
            self.MPC.x_init.setPVals(self.x_final[:,i])
            #print self.MPC.x_init.getPVals() #TRY
            self.solveOCP(self.MPC.prob)
            self.u_final[:,i] = self.MPC.u_con(0,True).flatten()
            I.setInput(self.u_final[:,i],'p')
            #print self.u_final[:,i]
            xint0 = x_plant
            for j in range(self.Nintegrate):
                I.setInput(xint0,'x0')
                I.evaluate()
                xint0 = I.getOutput('xf')
            x_plant = NP.array(xint0).flatten()

            x_meas = self.est.getMeasurements(x_plant)
            self.x_measured[:,i+1] = x_meas
            
            self.x_final[:,i+1],est_P = self.est.estimate(x_meas,est_P,self.x_final[:,i],self.u_final[:,i],self.MPC.t_plant)
         
            #raw_input('TTT')
        if self.save is not None:
            NP.savetxt(self.save+'_states',self.x_final)
            NP.savetxt(self.save+'_controls',self.u_final)        
        
    def plotStates(self,n=None,save=None):
        t = NP.linspace(0.0,self.MPC.total_plant,self.MPC.N_sampling+1)
        if n is None:
            np = NP.ceil(NP.sqrt(self.MPC.nx))
            for i in range(self.MPC.nx):
                plt.subplot(np,np,i+1)
                plt.plot(t,self.x_final[i,:])
                plt.xlabel('t')
                plt.ylabel('x'+str(i))
            if save is not None:
                plt.savefig(save[0]+'_states',format=save[1])    
            plt.show()
        elif isinstance(n,list) is True:
            if len(n)>self.MPC.nx:
                raise IOError('List of states too long')
            np = NP.ceil(NP.sqrt(NP.abs(len(n))))
            j = 1
            for i in n:
                plt.subplot(np,np,j)
                plt.plot(t,self.x_final[i,:])
                plt.xlabel('t')
                plt.ylabel('x'+str(i))
                j+=1
            if save is not None:
                plt.savefig(save[0]+'_states',format=save[1])    
            plt.show()
        elif isinstance(n,int) is True:
                plt.plot(t,self.x_final[n,:])
                plt.xlabel('t')
                plt.ylabel('x'+str(n))
                if save is not None:
                    plt.savefig(save[0]+'_states',format=save[1])    
                plt.show()
        else:
            raise IOError('\'n\' should be a list or integer')

    def plotControls(self,n=None,save=None):
        t = NP.linspace(0.0,self.MPC.total_plant-self.MPC.t_plant,self.MPC.N_sampling)
        if n is None:
            np = NP.ceil(NP.sqrt(NP.abs(self.MPC.nu)))
            for i in range(self.MPC.nu):
                plt.subplot(np,np,i+1)
                plt.step(t,self.u_final[i,:])
                plt.xlabel('t')
                plt.ylabel('u'+str(i))
            if save is not None:
                plt.savefig(save[0]+'controls',format=save[1])    
            plt.show()
        elif isinstance(n,list) is True:
            if len(n)>self.MPC.nu:
                raise IOError('List of controls too long')
            np = NP.ceil(NP.sqrt(NP.abs(len(n))))
            j = 1
            for i in n:
                plt.subplot(np,np,j)
                plt.plot(t,self.u_final[i,:])
                plt.xlabel('t')
                plt.ylabel('u'+str(i))
                j+=1
            if save is not None:
                plt.savefig(save[0]+'_controls',format=save[1])    
            plt.show()
        elif isinstance(n,int) is True:
                plt.plot(t,self.u_final[n,:])
                plt.xlabel('t')
                plt.ylabel('u'+str(n))
                if save is not None:
                    plt.savefig(save[0]+'_controls',format=save[1])    
                plt.show()
        else:
            raise IOError('\'n\' should be a list or integer')
    def plotMeasurements(self):
        t = NP.linspace(0.0,self.MPC.total_plant,self.MPC.N_sampling+1)
        i = 0        
        for j in self.nmx:
            plt.plot(t,self.x_measured[i,:])
            plt.plot(t,self.x_final[j,:])
            i+=1
            plt.show()