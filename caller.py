# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:12:09 2015

@author: user
"""

from casadi import *
import numpy as NP
import matplotlib.pyplot as plt
import copy
def TypeWrapper(cl,t):
    for k in dir(cl):
        if k in ["__Csetitem__"]:
            def wrapper(original):
                def wrapped(self,*args,**kwargs):
                    if isinstance(args[1],int):
                        k = 2
                    else:
                        k = 1
                    een = False
                    twee = False
                    if hasattr(self,'pas'):
                        een = True
                    if hasattr(args[k],'pas'):
                        twee = True
                    if een and twee:
                        if len(self.pas.obj) > len(args[k].pas.obj):
                            pas = self.pas
                        else:
                            pas = args[k].pas
                    elif een and not twee:
                        pas = self.pas
                    elif twee and not een:
                        pas = args[k].pas
                    else:
                        pas = []
                    ret = original(self,*args,**kwargs)
                    ret = self
                    if isinstance(ret,t) and not(isinstance(ret,cl)):
                        out = cl(ret)
                        out.initialize(pas)
                    else:
                        out = ret
                        out.initialize(pas)
                    return out
                return wrapped
            setattr(cl,k,wrapper(getattr(cl,k)))
        elif k in ["__Cgetitem__","__mul__","__add__","trans","mul","__neg__","__rmul__","__sub__","__div__","__truediv__","__pow__","__constpow__","__rpow__","__radd__","__rsub__","__rdiv__","__rtruediv__","fabs"]:
            def wrapper(original):
                def wrapped(self,*args,**kwargs):
                    een = False
                    twee = False
                    if hasattr(self,'pas'):
                        een = True
                    if len(args) > 0:
                        if hasattr(args[0],'pas'):
                            twee = True
                    if een and twee:
                        if len(self.pas.obj) > len(args[0].pas.obj):
                            pas = self.pas
                        else:
                            pas = args[0].pas
                    elif een and not twee:
                        pas = self.pas
                    elif twee and not een:
                        pas = args[0].pas
                    else:
                        pas = []
                    ret = original(self,*args,**kwargs)
                    if isinstance(ret,t) and not(isinstance(ret,cl)):
                        out = cl(ret)
                        out.initialize(pas)
                    else:
                        out = ret
                        out.initialize(pas)
                    return out
                return wrapped
            setattr(cl,k,wrapper(getattr(cl,k)))
class caller(SX):
    def initialize(self,pas):
        self.pas = pas
        self.f = SXFunction(self.pas.obj,[self])
        self.f.init()
    def ff(self,member,i):
        print type(member)
        print member
        if member.shape[1] > 1:
            return member[:,i]
        else:
            return member
    def __call__(self,j=None,num=False):
        if j == 0:
            nn = self.f.getNumInputs()
            print nn
            ns = self.shape[0]
            for i in range(nn):
                x = self.pas.obj[i].evalAtStart()
            if num:
                xp = self.pas.obj[i].getPVals()
                x = xp.reshape((xp.shape[0]/self.pas.obj[i].shape[0],self.pas.obj[i].shape[0]))[0,:].T
        if j == 'control':
            nn = self.f.getNumInputs()
            ns = self.shape[0]
            for i in range(nn):
                x = self.pas.obj[i].evalAtControls()
            if num:
                xp = self.pas.obj[i].getPVals()
                x = xp.reshape((xp.shape[0]/self.pas.obj[i].shape[0],self.pas.obj[i].shape[0])).T
        if j < 0:
            nn = self.f.getNumInputs()
            for i in range(nn):
                x = self.pas.obj[i].evalAtEnd()             
            if num:
                xp = self.pas.obj[i].getPVals()
                x = xp.reshape((xp.shape[0]/self.pas.obj[i].shape[0],self.pas.obj[i].shape[0]))[-1,:].T
        return x
#TypeWrapper(caller,SX)