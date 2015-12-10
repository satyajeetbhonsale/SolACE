# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:55:16 2015

@author: satyajeet
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
from SolACE.Estimator import Estimator

#Vapor heat capacity [kJ kg-1 C-1]
cp_vap_A = 14.6;
cp_vap_B = 2.04;
cp_vap_C = 1.05;
cp_vap_D = 1.85;
cp_vap_E = 1.87;
cp_vap_F = 2.02;
cp_vap_G = 0.712;
cp_vap_H = 0.628;

## Liquid heat capacity [kJ kg-1 C-1]
cp_liq_A = 0;  ## A-C not condensable
cp_liq_B = 0;
cp_liq_C = 0;
cp_liq_D = 7.66;
cp_liq_E = 4.17;
cp_liq_F = 4.45;
cp_liq_G = 2.55;
cp_liq_H = 2.45;

## Heat of vaporization [kJ kg-1]
H_vap_A = 0;
H_vap_B = 0;
H_vap_C = 0;
H_vap_D = 202;
H_vap_E = 372;
H_vap_F = 372;
H_vap_G = 523;
H_vap_H = 486;

#Reference Temperature
Tref = 273.15;
Tstr = 338.85;

RkJ   = 8.31451e0;
Rkcal = 1.987e0;

## Cooling water
cp_cw     = 4.18e0;
#Antione Constants
A_D = 20.81; B_D = -1444.0; C_D = 259.0;
A_E = 21.24; B_E = -2114.0; C_E = 265.5;
A_F = 21.24; B_F = -2144.0; C_F = 265.5;
A_G = 21.32; B_G = -2748.0; C_G = 232.9;
A_H = 22.10; B_H = -3318.0; C_H = 249.6;

'''# Feed streams #

// Temperatures
// Tabel 1 - Downs and Vogel
'''
stream_1_T = 45 + Tref; ## 318.15 K
stream_2_T = 45 + Tref;
stream_3_T = 45 + Tref;
stream_4_T = 45 + Tref;
stream_8_T = 102.9 + Tref;

#// Composition of Feed Stream A
stream_1_conc_A = 0.99990;
stream_1_conc_B = 0.00010;
stream_1_conc_C = 0.00000;
stream_1_conc_D = 0.00000;
stream_1_conc_E = 0.00000;
stream_1_conc_F = 0.00000;
stream_1_conc_G = 0.00000;
stream_1_conc_H = 0.00000;

#// Composition of Feed Stream D
stream_2_conc_A = 0.00000;
stream_2_conc_B = 0.00010;
stream_2_conc_C = 0.00000;
stream_2_conc_D = 0.99990;
stream_2_conc_E = 0.00000;
stream_2_conc_F = 0.00000;
stream_2_conc_G = 0.00000;
stream_2_conc_H = 0.00000;

#// Composition of Feed Stream E
stream_3_conc_A = 0.00000;
stream_3_conc_B = 0.00000;
stream_3_conc_C = 0.00000;
stream_3_conc_D = 0.00000;
stream_3_conc_E = 0.99990;
stream_3_conc_F = 0.00010;
stream_3_conc_G = 0.00000;
stream_3_conc_H = 0.00000;

#// Composition of Feed Stream C
stream_4_conc_A = 0.48500;
stream_4_conc_B = 0.00500;
stream_4_conc_C = 0.51000;
stream_4_conc_D = 0.00000;
stream_4_conc_E = 0.00000;
stream_4_conc_F = 0.00000;
stream_4_conc_G = 0.00000;
stream_4_conc_H = 0.00000;

stream_1_cp = (stream_1_conc_A * cp_vap_A + stream_1_conc_B * cp_vap_B + 
               stream_1_conc_C * cp_vap_C + stream_1_conc_D * cp_vap_D +
               stream_1_conc_E * cp_vap_E + stream_1_conc_F * cp_vap_F +
               stream_1_conc_G * cp_vap_G + stream_1_conc_H * cp_vap_H);

stream_2_cp = (stream_2_conc_A * cp_vap_A + stream_2_conc_B * cp_vap_B +
               stream_2_conc_C * cp_vap_C + stream_2_conc_D * cp_vap_D +
               stream_2_conc_E * cp_vap_E + stream_2_conc_F * cp_vap_F +
               stream_2_conc_G * cp_vap_G + stream_2_conc_H * cp_vap_H);

stream_3_cp = (stream_3_conc_A * cp_vap_A + stream_3_conc_B * cp_vap_B +
               stream_3_conc_C * cp_vap_C + stream_3_conc_D * cp_vap_D +
               stream_3_conc_E * cp_vap_E + stream_3_conc_F * cp_vap_F +
               stream_3_conc_G * cp_vap_G + stream_3_conc_H * cp_vap_H);

stream_4_cp = (stream_4_conc_A * cp_vap_A + stream_4_conc_B * cp_vap_B +
               stream_4_conc_C * cp_vap_C + stream_4_conc_D * cp_vap_D +
               stream_4_conc_E * cp_vap_E + stream_4_conc_F * cp_vap_F +
               stream_4_conc_G * cp_vap_G + stream_4_conc_H * cp_vap_H);
               
Vm = 141.53e0;    # Volume mixing zone

stream_1_flow = 11.2/3600.0;
stream_2_flow = 114.4/3600.0;
stream_3_flow = 98.0/3600.0;
stream_4_flow = 417.5/3600.0;
stream_5_flow = 465.7/3600.0;
# stream_6_flow = 1890.8/3600.0;
# stream_7_flow = 1476.0/3600.0;
stream_8_flow = 1201.5/3600.0;
stream_9_flow = 15.1/3600.0;


stream_5_conc_A  = 0.43263;
stream_5_conc_B  = 0.00444;
stream_5_conc_C  = 0.45264;
stream_5_conc_D  = 0.00116;
stream_5_conc_E  = 0.07256;
stream_5_conc_F  = 0.00885;
stream_5_conc_G  = 0.01964;
stream_5_conc_H  = 0.00808;

stream_8_conc_A  = 0.32958;
stream_8_conc_B  = 0.13823;
stream_8_conc_C  = 0.23978;
stream_8_conc_D  = 0.01257;
stream_8_conc_E  = 0.18579;
stream_8_conc_F  = 0.02263;
stream_8_conc_G  = 0.04844;
stream_8_conc_H  = 0.02299;

gamma_D_r = 0.996011e0;
gamma_E_r = 1;
gamma_F_r = 1.078e0;
gamma_G_r = 0.999e0;
gamma_H_r = 0.999e0;

## Reactor constants
kAr = 129.4e0;

## Volume Reactor
Vr = 36.8117791e0;
Vliq = 11.8;
# Vvap = 0;



## Densities
rho_liq_reactor  = 9.337145754e0;

Tm = 359.25;#start
Tr = 393.55;#start

# # Eq. 5.6
alpha_1 = 1.0399157e0;
alpha_2 = 1.011373129e0;
alpha_3 = 1;

## Eq. 5.17  (A, B, C not condensable)
reactor_x_A = 0.0;
reactor_x_B = 0.0;
reactor_x_C = 0.0;

## Heat exchange with cooling water
T_CWSr_in = 0.308000000000000e+03;

## Assumption - from measurement data
T_CWSr_out = 0.367599000000000e+03;

UA =  127.6;

lx = [-0.10]*18
ux = [1000]*18

lu = [0.0]*4
uu = [227.10,1.0,1.0,1.0]
xs = [4.883796012e+01,1.3581698782601494e01,4.003019454e01,9.7240317507928289e00,2.7443144213409383e01,2.5287892794899194e00,5.4376801699256117e00,2.5340908634231365e00,3.626444078091077e02,5.031608878009897e00,2.1296567956657944e00,3.7868352943264569e00,1.5154871302534503e-01,9.6154000218439766e00,1.3071231152493332e00,6.328731837338335e01,6.9378697766081004e01,393.55]
mue1 = [-1,0,-1,-1,0,0,1,0]
mue2 = [-1,0,-1,0,-1,0,0,1]
mue3 = [-1,0,0,-3,-1,3,0,0]

MPC = MPCproblem(1000,total_plant=10000.0)
x = MPC.addControllerStates(18,lx,ux,xs)
u = MPC.addControllerInputs(4,lu,uu)


mixing = x[0:9] #mixing(8) is temperature
reactor = x[9:18] #reactor(8) is temperature

mixing_zone_N = (mixing[0] + mixing[1] + mixing[2] + mixing[3] + mixing[4] + mixing[5] + mixing[6] + mixing[7])

pm_MPa = mixing_zone_N * (RkJ * mixing[8] / Vm) / 1000.0

stream_6_conc = SX.zeros(8)
for i in range(8):
    stream_6_conc[i] = mixing[i]/mixing_zone_N
stream_6_conc = MPC.prob.makeExpression(stream_6_conc)

# Energy balance for the mixing zone
mixing_zone_Ncp =  (mixing[0]*cp_vap_A + mixing[1]*cp_vap_B + mixing[2]*cp_vap_C
                  + mixing[3]*cp_vap_D + mixing[4]*cp_vap_E + mixing[5]*cp_vap_F
                  + mixing[6]*cp_vap_G + mixing[7]*cp_vap_H);
                  
#// Stream 5
stream_5_cp = (stream_5_conc_A * cp_vap_A + stream_5_conc_B * cp_vap_B +
               stream_5_conc_C * cp_vap_C + stream_5_conc_D * cp_vap_D +
               stream_5_conc_E * cp_vap_E + stream_5_conc_F * cp_vap_F +
               stream_5_conc_G * cp_vap_G + stream_5_conc_H * cp_vap_H);

#// Stream 8
stream_8_cp = (stream_8_conc_A * cp_vap_A + stream_8_conc_B * cp_vap_B +
               stream_8_conc_C * cp_vap_C + stream_8_conc_D * cp_vap_D +
               stream_8_conc_E * cp_vap_E + stream_8_conc_F * cp_vap_F +
               stream_8_conc_G * cp_vap_G + stream_8_conc_H * cp_vap_H);

reactor_x_D  = reactor[3]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_E  = reactor[4]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_F  = reactor[5]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_G  = reactor[6]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_H  = reactor[7]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);

Vr_liq =   (reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]) / rho_liq_reactor

Vvap = Vr - Vr_liq

pr_sat_D = 1.0e-6 * exp(A_D + (B_D / (C_D + reactor[8] - Tref) ) ) ;
pr_sat_E = 1.0e-6 * exp(A_E + (B_E / (C_E + reactor[8] - Tref) ) ) ;
pr_sat_F = 1.0e-6 * exp(A_F + (B_F / (C_F + reactor[8] - Tref) ) ) ;
pr_sat_G = 1.0e-6 * exp(A_G + (B_G / (C_G + reactor[8] - Tref) ) ) ;
pr_sat_H = 1.0e-6 * exp(A_H + (B_H / (C_H + reactor[8] - Tref) ) ) ;

p_A_r = (reactor[0] * RkJ * reactor[8] / 1000.0)/Vvap#/*10.0*/);
p_B_r = (reactor[1] * RkJ * reactor[8] / 1000.0)/Vvap#/*10.0*/);
p_C_r = (reactor[2] * RkJ * reactor[8] / 1000.0)/Vvap#/*10.0*/);/**/
p_D_r = gamma_D_r * reactor_x_D  * pr_sat_D#s/*/10.0*/;
p_E_r = gamma_E_r * reactor_x_E  * pr_sat_E#/*/10.0*/;
p_F_r = gamma_F_r * reactor_x_F  * pr_sat_F#/*/10.0*/;
p_G_r = gamma_G_r * reactor_x_G  * pr_sat_G#/*/10.0*/;
p_H_r = gamma_H_r * reactor_x_H  * pr_sat_H#/*/10.0*/;


reactor_R = SX.zeros(3)
reactor_R[0] = alpha_1 * Vvap* exp(44.06 - (42600.0/(Rkcal * reactor[8]))  ) * (pow((p_A_r*1.0e03),1.080)) * (pow((p_C_r*1.0e03),0.311)) * (pow((p_D_r*1.0e03),0.874)) / 3600.0;
reactor_R[1] = alpha_2 * Vvap* exp(10.27 - (19500.0/(Rkcal * reactor[8]))  ) * (pow((p_A_r*1.0e03),1.150)) * (pow((p_C_r*1.0e03),0.370)) * (p_E_r*1.0e03)/ 3600.0;
reactor_R[2] = alpha_3 * Vvap* exp(59.50 - (59500.0/(Rkcal * reactor[8]))  ) * (p_A_r*1.0e03) * ( 0.77 * (p_D_r*1.0e03) + (p_E_r*1.0e03)) / 3600.0;
reactor_R = MPC.prob.makeExpression(reactor_R)

#///  Reactor intermediate states
reactor_conv_rate = SX.zeros(8)
reactor_conv_rate[0] = mue1[0] * reactor_R[0] + mue2[0] * reactor_R[1] + mue3[0] * reactor_R[2];
reactor_conv_rate[1] = 0;
reactor_conv_rate[2] = mue1[2] * reactor_R[0] + mue2[2] * reactor_R[1] ;
reactor_conv_rate[3] = mue1[3] * reactor_R[0] + mue3[3] * reactor_R[2];
reactor_conv_rate[4] = mue2[4] * reactor_R[1] + mue3[4] * reactor_R[2];
reactor_conv_rate[5] = mue3[5] * reactor_R[2];
reactor_conv_rate[6] = mue1[6] * reactor_R[0];
reactor_conv_rate[7] = mue2[7] * reactor_R[1];
reactor_conv_rate = MPC.prob.makeExpression(reactor_conv_rate)

reactor_Ncp  =  (reactor[0] * cp_vap_A + reactor[1] * cp_vap_B
               + reactor[2] * cp_vap_C + reactor[3] * cp_liq_D
               + reactor[4] * cp_liq_E + reactor[5] * cp_liq_F
               + reactor[6] * cp_liq_G + reactor[7] * cp_liq_H);


stream_6_cp = ((stream_6_conc[0]) * cp_vap_A + (stream_6_conc[1]) * cp_vap_B +
               (stream_6_conc[2]) * cp_vap_C + (stream_6_conc[3]) * cp_vap_D +
               (stream_6_conc[4]) * cp_vap_E + (stream_6_conc[5]) * cp_vap_F +
               (stream_6_conc[6]) * cp_vap_G + (stream_6_conc[7]) * cp_vap_H);

delt_Hr = SX.zeros(3)
delt_Hr[0] = (mue1[0] * cp_vap_A * (reactor[8] - Tref) + mue1[2] * cp_vap_C * (reactor[8] - Tref) + mue1[3] * cp_vap_D * (reactor[8] - Tref) + mue1[6] * cp_vap_G * (reactor[8]- Tref) - 136033.04e0) / 1000.0;
delt_Hr[1] = (mue2[0] * cp_vap_A * (reactor[8] - Tref) + mue2[2] * cp_vap_C * (reactor[8] - Tref) + mue2[4] * cp_vap_E * (reactor[8] - Tref) + mue2[7] * cp_vap_H * (reactor[8]- Tref) - 93337.9616e0) / 1000.0;
delt_Hr[2] = (mue3[0] * cp_vap_A * (reactor[8] - Tref) + mue3[3] * cp_vap_D * (reactor[8] - Tref) + mue3[4] * cp_vap_E * (reactor[8] - Tref) + mue3[5] * cp_vap_F * (reactor[8] - Tref) + 0) / 1000.0;
delt_Hr = MPC.prob.makeExpression(delt_Hr)

reactor_exoth_heat = reactor_R[0] * delt_Hr[0] + reactor_R[1]* delt_Hr[1] + reactor_R[2] * delt_Hr[2];

pr_MPa = (p_A_r + p_B_r + p_C_r + p_D_r + p_E_r + p_F_r + p_G_r + p_H_r);

# Eq. 5.15
stream_7_conc = SX.zeros(8)
stream_7_conc[0] = p_A_r/pr_MPa;
stream_7_conc[1] = p_B_r/pr_MPa;
stream_7_conc[2] = p_C_r/pr_MPa;
stream_7_conc[3] = p_D_r/pr_MPa;
stream_7_conc[4] = p_E_r/pr_MPa;
stream_7_conc[5] = p_F_r/pr_MPa;
stream_7_conc[6] = p_G_r/pr_MPa;
stream_7_conc[7] = p_H_r/pr_MPa;
stream_7_conc = MPC.prob.makeExpression(stream_7_conc)

Qr = u[0] * cp_cw * (T_CWSr_out - T_CWSr_in)/1000.0

press_m_r_diff = sqrt(pm_MPa  - pr_MPa)
stream_6_flow = 0.8333711713 * (press_m_r_diff)

ps_MPa = 2.70

press_r_s_diff = sqrt(pr_MPa - ps_MPa);
stream_7_flow = 1.53546206685993 * press_r_s_diff;

rhs = Expression(SX.zeros(18))
rhs[0] = u[3] * stream_1_conc_A + stream_5_flow * stream_5_conc_A + stream_8_flow * stream_8_conc_A - stream_6_flow * (mixing[0]/mixing_zone_N)
rhs[1] = u[3] * stream_1_conc_B + u[1] * stream_2_conc_B + stream_5_flow * stream_5_conc_B + stream_8_flow * stream_8_conc_B - stream_6_flow * (mixing[1]/mixing_zone_N);
rhs[2] = stream_5_flow * stream_5_conc_C + stream_8_flow * stream_8_conc_C - stream_6_flow * (mixing[2]/mixing_zone_N);
rhs[3] = u[1] * stream_2_conc_D + stream_5_flow * stream_5_conc_D + stream_8_flow * stream_8_conc_D - stream_6_flow * (mixing[3]/mixing_zone_N)
rhs[4] = u[2] * stream_3_conc_E + stream_5_flow * stream_5_conc_E + stream_8_flow * stream_8_conc_E - stream_6_flow * (mixing[4]/mixing_zone_N)
rhs[5] = u[2] * stream_3_conc_F + stream_5_flow * stream_5_conc_F + stream_8_flow * stream_8_conc_F - stream_6_flow * (mixing[5]/mixing_zone_N)
rhs[6] = stream_5_flow * stream_5_conc_G + stream_8_flow * stream_8_conc_G - stream_6_flow * (mixing[6]/mixing_zone_N)
rhs[7] = stream_5_flow * stream_5_conc_H + stream_8_flow * stream_8_conc_H - stream_6_flow * (mixing[7]/mixing_zone_N)
rhs[8] = (u[3] * stream_1_cp * (stream_1_T - mixing[8]) + u[1] * stream_2_cp * (stream_2_T - mixing[8]) + u[2] * stream_3_cp * (stream_3_T - mixing[8]) + stream_5_flow * stream_5_cp * (Tstr - mixing[8]) + stream_8_flow * stream_8_cp * (stream_8_T - mixing[8]))/mixing_zone_Ncp;

rhs[9] = stream_6_flow * (stream_6_conc[0]) - stream_7_flow * (stream_7_conc[0]) + reactor_conv_rate[0];
rhs[10] = stream_6_flow * (stream_6_conc[1]) - stream_7_flow * (stream_7_conc[1]) + reactor_conv_rate[1];
rhs[11] = stream_6_flow * (stream_6_conc[2]) - stream_7_flow * (stream_7_conc[2]) + reactor_conv_rate[2];
rhs[12] = stream_6_flow * (stream_6_conc[3]) - stream_7_flow * (stream_7_conc[3]) + reactor_conv_rate[3];
rhs[13] = stream_6_flow * (stream_6_conc[4]) - stream_7_flow * (stream_7_conc[4]) + reactor_conv_rate[4];
rhs[14] = stream_6_flow * (stream_6_conc[5]) - stream_7_flow * (stream_7_conc[5]) + reactor_conv_rate[5];
rhs[15] = stream_6_flow * (stream_6_conc[6]) - stream_7_flow * (stream_7_conc[6]) + reactor_conv_rate[6];
rhs[16] = stream_6_flow * (stream_6_conc[7]) - stream_7_flow * (stream_7_conc[7]) + reactor_conv_rate[7];
rhs[17] = ((1/1000.0) * stream_6_flow * stream_6_cp * (mixing[8] - reactor[8]) - Qr - reactor_exoth_heat)/reactor_Ncp;

MPC.addControllerODEs(x,rhs)

r = NP.array([393.5,2.745,25.93,114.5/3600.0,98.0/3600.0,11.2/3600.0,])
h = NP.array([reactor[8],pr_MPa,u[0],u[1],u[2],u[3]])
H = h - r
S = [9.8657122193156304e-01,0.1428778068436973e-01,0.01,0.01,0.01,0.01]

ff = sum(S[0]*H[0]('coll')*H[0]('coll')+S[1]*H[1]('coll')*H[1]('coll')+S[2]*H[2]('coll')*H[2]('coll')+S[3]*H[3]('coll')*H[3]('coll')+S[4]*H[4]('coll')*H[4]('coll')+S[5]*H[5]('coll')*H[5]('coll')) 
MPC.addControlObjective(ff)

x1 = MPC.addPlantStates(18)
u1 = MPC.addPlantInputs(4)

mixing = x1[0:9] #mixing(8) is temperature
reactor = x1[9:18] #reactor(8) is temperature

mixing_zone_N = (mixing[0] + mixing[1] + mixing[2] + mixing[3] + mixing[4] + mixing[5] + mixing[6] + mixing[7])

pm_MPa = mixing_zone_N * (RkJ * mixing[8] / Vm) / 1000.0

stream_6_conc = SX.zeros(8)
for i in range(8):
    stream_6_conc[i] = mixing[i]/mixing_zone_N

# Energy balance for the mixing zone
mixing_zone_Ncp =  (mixing[0]*cp_vap_A + mixing[1]*cp_vap_B + mixing[2]*cp_vap_C
                  + mixing[3]*cp_vap_D + mixing[4]*cp_vap_E + mixing[5]*cp_vap_F
                  + mixing[6]*cp_vap_G + mixing[7]*cp_vap_H);
                  
#// Stream 5
stream_5_cp = (stream_5_conc_A * cp_vap_A + stream_5_conc_B * cp_vap_B +
               stream_5_conc_C * cp_vap_C + stream_5_conc_D * cp_vap_D +
               stream_5_conc_E * cp_vap_E + stream_5_conc_F * cp_vap_F +
               stream_5_conc_G * cp_vap_G + stream_5_conc_H * cp_vap_H);

#// Stream 8
stream_8_cp = (stream_8_conc_A * cp_vap_A + stream_8_conc_B * cp_vap_B +
               stream_8_conc_C * cp_vap_C + stream_8_conc_D * cp_vap_D +
               stream_8_conc_E * cp_vap_E + stream_8_conc_F * cp_vap_F +
               stream_8_conc_G * cp_vap_G + stream_8_conc_H * cp_vap_H);

reactor_x_D  = reactor[3]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_E  = reactor[4]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_F  = reactor[5]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_G  = reactor[6]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);
reactor_x_H  = reactor[7]/(reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]);

Vr_liq =   (reactor[3] + reactor[4] + reactor[5] + reactor[6] + reactor[7]) / rho_liq_reactor

Vvap = Vr - Vr_liq

pr_sat_D = 1.0e-6 * exp(A_D + (B_D / (C_D + reactor[8] - Tref) ) ) ;
pr_sat_E = 1.0e-6 * exp(A_E + (B_E / (C_E + reactor[8] - Tref) ) ) ;
pr_sat_F = 1.0e-6 * exp(A_F + (B_F / (C_F + reactor[8] - Tref) ) ) ;
pr_sat_G = 1.0e-6 * exp(A_G + (B_G / (C_G + reactor[8] - Tref) ) ) ;
pr_sat_H = 1.0e-6 * exp(A_H + (B_H / (C_H + reactor[8] - Tref) ) ) ;

p_A_r = (reactor[0] * RkJ * reactor[8] / 1000.0)/Vvap#/*10.0*/);
p_B_r = (reactor[1] * RkJ * reactor[8] / 1000.0)/Vvap#/*10.0*/);
p_C_r = (reactor[2] * RkJ * reactor[8] / 1000.0)/Vvap#/*10.0*/);/**/
p_D_r = gamma_D_r * reactor_x_D  * pr_sat_D#s/*/10.0*/;
p_E_r = gamma_E_r * reactor_x_E  * pr_sat_E#/*/10.0*/;
p_F_r = gamma_F_r * reactor_x_F  * pr_sat_F#/*/10.0*/;
p_G_r = gamma_G_r * reactor_x_G  * pr_sat_G#/*/10.0*/;
p_H_r = gamma_H_r * reactor_x_H  * pr_sat_H#/*/10.0*/;


reactor_R = SX.zeros(3)
reactor_R[0] = alpha_1 * Vvap* exp(44.06 - (42600.0/(Rkcal * reactor[8]))  ) * (pow((p_A_r*1.0e03),1.080)) * (pow((p_C_r*1.0e03),0.311)) * (pow((p_D_r*1.0e03),0.874)) / 3600.0;
reactor_R[1] = alpha_2 * Vvap* exp(10.27 - (19500.0/(Rkcal * reactor[8]))  ) * (pow((p_A_r*1.0e03),1.150)) * (pow((p_C_r*1.0e03),0.370)) * (p_E_r*1.0e03)/ 3600.0;
reactor_R[2] = alpha_3 * Vvap* exp(59.50 - (59500.0/(Rkcal * reactor[8]))  ) * (p_A_r*1.0e03) * ( 0.77 * (p_D_r*1.0e03) + (p_E_r*1.0e03)) / 3600.0;

#///  Reactor intermediate states
reactor_conv_rate = SX.zeros(8)
reactor_conv_rate[0] = mue1[0] * reactor_R[0] + mue2[0] * reactor_R[1] + mue3[0] * reactor_R[2];
reactor_conv_rate[1] = 0;
reactor_conv_rate[2] = mue1[2] * reactor_R[0] + mue2[2] * reactor_R[1] ;
reactor_conv_rate[3] = mue1[3] * reactor_R[0] + mue3[3] * reactor_R[2];
reactor_conv_rate[4] = mue2[4] * reactor_R[1] + mue3[4] * reactor_R[2];
reactor_conv_rate[5] = mue3[5] * reactor_R[2];
reactor_conv_rate[6] = mue1[6] * reactor_R[0];
reactor_conv_rate[7] = mue2[7] * reactor_R[1];

reactor_Ncp  =  (reactor[0] * cp_vap_A + reactor[1] * cp_vap_B
               + reactor[2] * cp_vap_C + reactor[3] * cp_liq_D
               + reactor[4] * cp_liq_E + reactor[5] * cp_liq_F
               + reactor[6] * cp_liq_G + reactor[7] * cp_liq_H);


stream_6_cp = ((stream_6_conc[0]) * cp_vap_A + (stream_6_conc[1]) * cp_vap_B +
               (stream_6_conc[2]) * cp_vap_C + (stream_6_conc[3]) * cp_vap_D +
               (stream_6_conc[4]) * cp_vap_E + (stream_6_conc[5]) * cp_vap_F +
               (stream_6_conc[6]) * cp_vap_G + (stream_6_conc[7]) * cp_vap_H);

delt_Hr = SX.zeros(3)
delt_Hr[0] = (mue1[0] * cp_vap_A * (reactor[8] - Tref) + mue1[2] * cp_vap_C * (reactor[8] - Tref) + mue1[3] * cp_vap_D * (reactor[8] - Tref) + mue1[6] * cp_vap_G * (reactor[8]- Tref) - 136033.04e0) / 1000.0;
delt_Hr[1] = (mue2[0] * cp_vap_A * (reactor[8] - Tref) + mue2[2] * cp_vap_C * (reactor[8] - Tref) + mue2[4] * cp_vap_E * (reactor[8] - Tref) + mue2[7] * cp_vap_H * (reactor[8]- Tref) - 93337.9616e0) / 1000.0;
delt_Hr[2] = (mue3[0] * cp_vap_A * (reactor[8] - Tref) + mue3[3] * cp_vap_D * (reactor[8] - Tref) + mue3[4] * cp_vap_E * (reactor[8] - Tref) + mue3[5] * cp_vap_F * (reactor[8] - Tref) + 0) / 1000.0;

reactor_exoth_heat = reactor_R[0] * delt_Hr[0] + reactor_R[1]* delt_Hr[1] + reactor_R[2] * delt_Hr[2];

pr_MPa = (p_A_r + p_B_r + p_C_r + p_D_r + p_E_r + p_F_r + p_G_r + p_H_r);

# Eq. 5.15
stream_7_conc = SX.zeros(8)
stream_7_conc[0] = p_A_r/pr_MPa;
stream_7_conc[1] = p_B_r/pr_MPa;
stream_7_conc[2] = p_C_r/pr_MPa;
stream_7_conc[3] = p_D_r/pr_MPa;
stream_7_conc[4] = p_E_r/pr_MPa;
stream_7_conc[5] = p_F_r/pr_MPa;
stream_7_conc[6] = p_G_r/pr_MPa;
stream_7_conc[7] = p_H_r/pr_MPa;

Qr = u1[0] * cp_cw * (T_CWSr_out - T_CWSr_in)/1000.0

press_m_r_diff = sqrt(pm_MPa  - pr_MPa)
stream_6_flow = 0.8333711713 * (press_m_r_diff)

ps_MPa = 2.70

press_r_s_diff = sqrt(pr_MPa - ps_MPa);
stream_7_flow = 1.53546206685993 * press_r_s_diff;

rhs1 = SX.zeros(18)
rhs1[0] = u1[3] * stream_1_conc_A + stream_5_flow * stream_5_conc_A + stream_8_flow * stream_8_conc_A - stream_6_flow * (mixing[0]/mixing_zone_N)
rhs1[1] = u1[3] * stream_1_conc_B + u1[1] * stream_2_conc_B + stream_5_flow * stream_5_conc_B + stream_8_flow * stream_8_conc_B - stream_6_flow * (mixing[1]/mixing_zone_N);
rhs1[2] = stream_5_flow * stream_5_conc_C + stream_8_flow * stream_8_conc_C - stream_6_flow * (mixing[2]/mixing_zone_N);
rhs1[3] = u1[1] * stream_2_conc_D + stream_5_flow * stream_5_conc_D + stream_8_flow * stream_8_conc_D - stream_6_flow * (mixing[3]/mixing_zone_N)
rhs1[4] = u1[2] * stream_3_conc_E + stream_5_flow * stream_5_conc_E + stream_8_flow * stream_8_conc_E - stream_6_flow * (mixing[4]/mixing_zone_N)
rhs1[5] = u1[2] * stream_3_conc_F + stream_5_flow * stream_5_conc_F + stream_8_flow * stream_8_conc_F - stream_6_flow * (mixing[5]/mixing_zone_N)
rhs1[6] = stream_5_flow * stream_5_conc_G + stream_8_flow * stream_8_conc_G - stream_6_flow * (mixing[6]/mixing_zone_N)
rhs1[7] = stream_5_flow * stream_5_conc_H + stream_8_flow * stream_8_conc_H - stream_6_flow * (mixing[7]/mixing_zone_N)
rhs1[8] = (u1[3] * stream_1_cp * (stream_1_T - mixing[8]) + u1[1] * stream_2_cp * (stream_2_T - mixing[8]) + u1[2] * stream_3_cp * (stream_3_T - mixing[8]) + stream_5_flow * stream_5_cp * (Tstr - mixing[8]) + stream_8_flow * stream_8_cp * (stream_8_T - mixing[8]))/mixing_zone_Ncp;

rhs1[9] = stream_6_flow * (stream_6_conc[0]) - stream_7_flow * (stream_7_conc[0]) + reactor_conv_rate[0];
rhs1[10] = stream_6_flow * (stream_6_conc[1]) - stream_7_flow * (stream_7_conc[1]) + reactor_conv_rate[1];
rhs1[11] = stream_6_flow * (stream_6_conc[2]) - stream_7_flow * (stream_7_conc[2]) + reactor_conv_rate[2];
rhs1[12] = stream_6_flow * (stream_6_conc[3]) - stream_7_flow * (stream_7_conc[3]) + reactor_conv_rate[3];
rhs1[13] = stream_6_flow * (stream_6_conc[4]) - stream_7_flow * (stream_7_conc[4]) + reactor_conv_rate[4];
rhs1[14] = stream_6_flow * (stream_6_conc[5]) - stream_7_flow * (stream_7_conc[5]) + reactor_conv_rate[5];
rhs1[15] = stream_6_flow * (stream_6_conc[6]) - stream_7_flow * (stream_7_conc[6]) + reactor_conv_rate[6];
rhs1[16] = stream_6_flow * (stream_6_conc[7]) - stream_7_flow * (stream_7_conc[7]) + reactor_conv_rate[7];
rhs1[17] = ((1/1000.0) * stream_6_flow * stream_6_cp * (mixing[8] - reactor[8]) - Qr - reactor_exoth_heat)/reactor_Ncp;

MPC.addPlantODEs(x1,rhs1)
x0 = NP.array([4.883796012e+01,1.3581698782601494e01,4.003019454e01,9.7240317507928289e00,2.7443144213409383e01,2.5287892794899194e00,5.4376801699256117e00,2.5340908634231365e00,3.626444078091077e02,5.031608878009897e00,2.1296567956657944e00,3.7868352943264569e00,1.5154871302534503e-01,9.6154000218439766e00,1.3071231152493332e00,6.328731837338335e01,6.9378697766081004e01,393.55])
MPC.setInitCondition(x0)
solver = MPCsolve(MPC,printlevel=0)
solver.solve()
solver.plotStates()#
solver.plotControls()#
###
x = NP.loadtxt('xTE')
mixing = x[0:9,:] #mixing(8) is temperature
reactor = x[9:18,:] #reactor(8) is temperature

mixing_zone_N = (mixing[0,:] + mixing[1,:] + mixing[2,:] + mixing[3,:] + mixing[4,:] + mixing[5,:] + mixing[6,:] + mixing[7,:])
pm_MPa = mixing_zone_N * (RkJ * mixing[8] / Vm) / 1000.0
t = NP.linspace(0,108000,301)
plt.plot(t,pm_MPa)
plt.show()