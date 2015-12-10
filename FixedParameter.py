# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:48:30 2015

@author: Satyajeet
"""

from casadi import *
import numpy as NP
import copy
from SolACE.caller import caller

class FixedParameters(caller):
    
    def initialize(self,obj,pvals):
        super(FixedParameters,self).initialize(obj)
        self.np = self.size()
        self.pvals = pvals
        self.pmin = pvals
        self.pmax = pvals
        self.p = MX.sym("F",self.np)
            
#    def evalp(self):
#        return self.p
        
    def setPVals(self,pvals):
        if isinstance(self.pvals,float):
            self.pvals = pvals
        elif isinstance(self.pvals,list):
            self.pvals = pvals
        elif pvals.shape[0] != self.pvals.shape[0]:
            self.pvals[0:pvals.shape[0]] = pvals
        else:
            self.pvals = pvals
        self.pmin = self.pvals
        self.pmax = self.pvals
        
    def getPVals(self):
        return self.pvals
        
    def getBounds(self):
        return self.pmin,self.pmax
        
#TypeWrapper(FixedParameters,SX)
