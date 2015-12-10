# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:27:15 2015

@author: Satyajeet
"""

import numpy as NP
from casadi import *
import matplotlib.pyplot as plt
from pomodoro.dataset import DataSet
from pomodoro.problem import Problem,MOproblem
import multiprocessing as mp
import pickle
import scipy

class SolveMS(object):
    
    def __init__(self,prob,hessian=True,printlevel=0,max_iter=4000,tol=1e-8,linsolver='ma27',inf= 'no',inf_ytol= 1e+8, inf_ctol=0.001, solver='ipopt'):
        self.prob = prob
        self.hessian = hessian
        self.printlevel = printlevel
        self.max_iter = max_iter
        self.tol = tol
        self.linsolver = linsolver
        self.inf= inf
        self.inf_ytol= inf_ytol
        self.inf_ctol= inf_ctol
        self.solv = solver
        
    def HackyMXFunction(self,ins,outs):
      # Obtain all symbolic primitives present in the inputs
      Vs = getSymbols(ins)
    
      # Construct a helper function with these symbols as inputs
      h = MXFunction(Vs, ins)
      h.init()
    
      # Assert dense inputs
      for i in ins:
        assert i.isDense(), "input must be dense"
      #print "Dag!!"
      # Obtain sparsity pattern of the helper function Jacobian
      # It should be a permutation of the unit matrix
    #  H = blockcat([[Sparsity(h.jacSparsity(i,j)) for i in range(h.getNumInputs())] for j in range(h.getNumOutputs())])
    #  
    #  assert H.size1()==H.size2(), "input must be one-to-one"
    #  assert H.colind()==range(H.size1()+1), "input must be one-to-one"
    #  assert H.T.colind()==range(H.size1()+1), "input must be one-to-one"
      
      
      # Missing assertion: check that only transpose, reshape, vertcat, slice, .. are used
    
      # Construct new MX symbols on which the original inputs will depend
      Vns = [MX.sym("V",i.size()) for i in ins]
    
      # Use symbolic adjoint mode to transform the new MX symbols
      res = h.callDerivative([DMatrix.zeros(i.shape) for i in Vs],[[[]]],[Vns],True,False)
      
      # Substitute the original inputs
      f = MXFunction(Vns,substitute(outs,Vs,res[2][0]))
      f.init()
      
      return f        
    def Init(self):
        f = self.prob.objective
        g = self.prob.constraints
        V = self.prob.V
        nlp = self.HackyMXFunction(nlpIn(x=V),nlpOut(f=f,g=g))
        self.solver = NlpSolver(self.solv,nlp)
        
        if self.solv == 'ipopt':
            if not self.hessian:
                self.solver.setOption("hessian_approximation","limited-memory")
 #           self.solver.setOption("print_level",self.printlevel)
            self.solver.setOption("max_iter",self.max_iter)
            self.solver.setOption("print_time",False)
            self.solver.setOption("tol",self.tol)
            self.solver.setOption("linear_solver",self.linsolver)
            #self.solver.setOption("expect_infeasible_problem",self.inf)
            #self.solver.setOption("expect_infeasible_problem_ytol",self.inf_ytol)
            #self.solver.setOption("expect_infeasible_problem_ctol",self.inf_ctol)
        elif self.solv == 'worhp':
            self.solver.setOption("print_time",False)
            self.solver.setOption("NLPprint",5)
        elif self.solv == 'sqpmethod':
            self.solver.setOption("print_time",False)
            self.solver.setOption("qp_solver",'qpoases')
        print "voor"
        self.solver.init()
        print "na"
        n = V.shape[0]
        Xinit = NP.zeros(n)
        self.solver.setInput(Xinit, "x0")
        self.solver.setInput(self.prob.cmin,"lbg")
        self.solver.setInput(self.prob.cmax,"ubg")
        lb, ub = self.prob.getBounds()
#        lb[0:4] = self.prob.x_init
#        ub[0:4] = self.prob.x_init
        self.solver.setInput(lb,"lbx")
        self.solver.setInput(ub,"ubx")
    def solve(self):
        self.Init()
        self.solver.evaluate()
        self.prob.saveOptResults(NP.array(self.solver.output("x")).flatten())

#        print self.f
       
        