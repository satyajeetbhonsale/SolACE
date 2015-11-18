# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:46:43 2015

@author: user
"""
class Passer(object):
    
    def __init__(self):
        self.obj = []
        self.trange = []
        
    def addObj(self,obj):
        self.obj.append(obj)
    
    def setTRange(self,trange):
        self.trange = trange