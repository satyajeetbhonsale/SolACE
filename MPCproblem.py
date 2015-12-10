# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:18:44 2015

@author: Satyajeet
"""

import numpy as NP
from pomodoro.problem.problem import Problem
from casadi import *
from SolACE.MultipleShootingProblem import MultipleShootingProblem
import matplotlib.pyplot as plt

class MPCproblem(object):
    ## Initializing the controller. Time Ranges etc.
    def __init__(self,N_sampling,t_control=None,total_plant=None,dis='Collocation'):
        self.N_sampling = N_sampling
        self.dis = dis
        if t_control == None and total_plant == None:
            raise IOError('Define either control horizon or total plant run time')
        if t_control is not None:
            self.t_plant = t_control/10.0 #10 Discretizations fixed
            self.t_control = t_control
            self.total_plant = self.t_plant * self.N_sampling
        if total_plant is not None:
            self.total_plant = total_plant
            self.t_plant = total_plant/self.N_sampling
            self.t_control = self.t_plant*10 #10 Discretizations fixed
        if total_plant is not None and t_control is not None:
            if total_plant/self.N_sampling > t_control/10:
                raise IOError('Check control horizon and total plant run time')
        if dis is not 'MultipleShooting':
            self.prob = Problem()
            
        else:
            print "Hello"
            self.prob = MultipleShootingProblem()
        self.prob.setTimeRange(0.0,self.t_control)
        self.t = SX.sym('t')
        self.nmx = None
    def addControllerStates(self,n,xmin,xmax,xstart=None,method=['LagrangeColl',10],constr=True):
        if self.dis is not 'MultipleShooting':
            self.x_con = self.prob.addStates(n,xmin,xmax,xstart,method=['LagrangeColl',10])
            self.x_init = self.prob.addFixedParameters(n)
            self.prob.addConstraints(self.x_con(0)-self.x_init,0.0)
            self.nx = n            
        else:
            self.x_con = self.prob.addStates(n,xmin,xmax,xstart,method=['LagrangeColl',10])
        return self.x_con
    def addControllerInputs(self,n,umin,umax,ustart=None,method=['PiecewiseConstant',10]):
        self.u_con = self.prob.addControls(n,umin,umax,ustart=None,method=['PiecewiseConstant',10])
        self.nu = n        
        return self.u_con
    def addControllerODEs(self,diff,u,rhs,colsel=None):
        self.rhs_con = rhs
        if self.dis is not 'MultipleShooting':
            self.prob.addOde(self.x_con,self.rhs_con)
        else:
            self.prob.addOde(self.x_con,self.u_con,self.rhs_con)
            
    def addControllerConstraints(self,c,cmin,cmax1=None):
        self.prob.addConstraints(c,cmin,cmax=cmax1)
    def addControlObjective(self,f):
        self.prob.addObjective(f)
    ### Initialize the plant
    def addPlantStates(self,n):
        self.x_plant = SX.sym('x',n)
        return self.x_plant
    def addPlantInputs(self,n):
        self.u_plant = SX.sym('u',n)
        return self.u_plant
    def addPlantODEs(self,x,rhs):
        self.plant_model = SXFunction(daeIn(x=self.x_plant,p=self.u_plant,t=self.t),daeOut(ode=rhs))
        self.plant_model.setOption('name','Integrator')
    def setInitCondition(self,xint):
        if issubclass(type(xint),numpy.ndarray) is False:
            raise IOError('Initial Condition should be an array')
        self.x0 = xint
    def addMeasurementNoise(self,nmx,std):
        self.nmx = nmx
        self.std = std
        if len(nmx) != len(std):
            raise IOError('Standard Deviation For All Measurements')

        