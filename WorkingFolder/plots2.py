# Plot file to plot 2 saved solutions from the same vehicle model. 
# %%
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from casadi import *
from enum import Enum
from vehicleModels import linearST
import scipy.io as syio
from matplotlib.patches import Ellipse

N = 100
#Parameters to model
class Params(Enum):
    m = 3000
    lf = 1.3
    lr = 1.5
    g = 9.82
    l = lf + lr
    w = 1.8
    Izz = m*((l+2)**2)/12
    h = 0.5

    """
            #Dry
    muxf = 1.2
    muxr = 1.2
    Bxf = 11.7
    Bxr = 11.1
    Cxf = 1.69
    Cxr = 1.69
    Exf = 0.377
    Exr = 0.362
    muyf = 0.935
    muyr = 0.961
    Byf = 8.86
    Byr = 9.3
    Cyf = 1.19
    Cyr = 1.19
    Eyf = -1.21
    Eyr = -1.11
    """

    

    
               #Wet
    muxf = 1.06
    muxr = 1.07
    Bxf = 12.0
    Bxr = 11.5
    Cxf = 1.8
    Cxr = 1.8
    Exf = 0.313
    Exr = 0.300
    muyf = 0.885
    muyr = 0.911
    Byf = 10.7
    Byr = 11.3
    Cyf = 1.07
    Cyr = 1.07
    Eyf = -2.14
    Eyr = -1.97
    


    #Time constant
    TdFxf = 0.1
    TdFxr = 0.1

#Elipse constants
R1 = 52 #diff from obstacle 1 center in x
R2 = 1 #height of obstacle 1
R3 = 60 #diff from obstacle 2 center in x
R4 = 4 #height of obstacle 2
obstacle_center_x = 0 #OBSTACLE CENTER X
obstacle_center_y = 0 #OBSTACLE CENTER Y
road_width = R4-R2
middle_of_road = obstacle_center_y+R2+road_width/2
upper_corner_of_road = obstacle_center_y+R4-(Params.w.value/2)-0.1

Fzf = Params.m.value*Params.g.value*Params.lr.value/(Params.lf.value+Params.lr.value)
Fzr = Params.m.value*Params.g.value*Params.lf.value/(Params.lf.value+Params.lr.value)
Dxf = Params.muxf.value*Fzf 
Dxr = Params.muxr.value*Fzr
Dyf = Params.muyf.value*Fzf 
Dyr = Params.muyr.value*Fzr

pos_x_opt = syio.loadmat('LinearSTData2/pos_x.mat')['pos_x_data'][0]
pos_y_opt = syio.loadmat('LinearSTData2/pos_y.mat')['pos_y_data'][0]
vel_x_opt = syio.loadmat('LinearSTData2/vel_x.mat')['vel_x_data'][0]
vel_y_opt = syio.loadmat('LinearSTData2/vel_y.mat')['vel_y_data'][0]
phi_opt = syio.loadmat('LinearSTData2/phi.mat')['phi_data'][0]
r_opt = syio.loadmat('LinearSTData2/r.mat')['r_data'][0]
delta_opt = syio.loadmat('LinearSTData2/delta.mat')['delta_data'][0]
delta_d_opt = syio.loadmat('LinearSTData2/delta_d.mat')['delta_d_data'][0]
Fxf_opt = syio.loadmat('LinearSTData2/fxf.mat')['fxf_data'][0]
Fxr_opt = syio.loadmat('LinearSTData2/fxr.mat')['fxr_data'][0]
T_opt = syio.loadmat('LinearSTData2/t.mat')['t_data'][0][0]
alphaf_opt = syio.loadmat('LinearSTData2/alphaf.mat')['alphaf_data'][0]
alphar_opt = syio.loadmat('LinearSTData2/alphar.mat')['alphar_data'][0]
Fyf_opt = syio.loadmat('LinearSTData2/fyf.mat')['fyf_data'][0]
Fyr_opt = syio.loadmat('LinearSTData2/fyr.mat')['fyr_data'][0]
Fzf_opt = syio.loadmat('LinearSTData2/fzf.mat')['fzf_data'][0][0]
Fzr_opt = syio.loadmat('LinearSTData2/fzr.mat')['fzr_data'][0][0]

pos_x_opt4 = syio.loadmat('LinearSTData3/pos_x.mat')['pos_x_data'][0]
pos_y_opt4 = syio.loadmat('LinearSTData3/pos_y.mat')['pos_y_data'][0]
vel_x_opt4 = syio.loadmat('LinearSTData3/vel_x.mat')['vel_x_data'][0]
vel_y_opt4 = syio.loadmat('LinearSTData3/vel_y.mat')['vel_y_data'][0]
phi_opt4 = syio.loadmat('LinearSTData3/phi.mat')['phi_data'][0]
r_opt4 = syio.loadmat('LinearSTData3/r.mat')['r_data'][0]
delta_opt4 = syio.loadmat('LinearSTData3/delta.mat')['delta_data'][0]
delta_d_opt4 = syio.loadmat('LinearSTData3/delta_d.mat')['delta_d_data'][0]
Fxf_opt4 = syio.loadmat('LinearSTData3/fxf.mat')['fxf_data'][0]
Fxr_opt4 = syio.loadmat('LinearSTData3/fxr.mat')['fxr_data'][0]
T_opt4 = syio.loadmat('LinearSTData3/t.mat')['t_data'][0][0]
alphaf_opt4 = syio.loadmat('LinearSTData3/alphaf.mat')['alphaf_data'][0]
alphar_opt4 = syio.loadmat('LinearSTData3/alphar.mat')['alphar_data'][0]
Fyf_opt4 = syio.loadmat('LinearSTData3/fyf.mat')['fyf_data'][0]
Fyr_opt4 = syio.loadmat('LinearSTData3/fyr.mat')['fyr_data'][0]
Fzf_opt4 = syio.loadmat('LinearSTData3/fzf.mat')['fzf_data'][0][0]
Fzr_opt4 = syio.loadmat('LinearSTData3/fzr.mat')['fzr_data'][0][0]

pos_x_opt2 = syio.loadmat('MagicSTData2/pos_x.mat')['pos_x_data'][0]
pos_y_opt2 = syio.loadmat('MagicSTData2/pos_y.mat')['pos_y_data'][0]
vel_x_opt2 = syio.loadmat('MagicSTData2/vel_x.mat')['vel_x_data'][0]
vel_y_opt2 = syio.loadmat('MagicSTData2/vel_y.mat')['vel_y_data'][0]
phi_opt2 = syio.loadmat('MagicSTData2/phi.mat')['phi_data'][0]
r_opt2 = syio.loadmat('MagicSTData2/r.mat')['r_data'][0]
delta_opt2 = syio.loadmat('MagicSTData2/delta.mat')['delta_data'][0]
delta_d_opt2 = syio.loadmat('MagicSTData2/delta_d.mat')['delta_d_data'][0]
Fxf_opt2 = syio.loadmat('MagicSTData2/fxf.mat')['fxf_data'][0]
Fxr_opt2 = syio.loadmat('MagicSTData2/fxr.mat')['fxr_data'][0]
T_opt2 = syio.loadmat('MagicSTData2/t.mat')['t_data'][0][0]
alphaf_opt2 = syio.loadmat('MagicSTData2/alphaf.mat')['alphaf_data'][0]
alphar_opt2 = syio.loadmat('MagicSTData2/alphar.mat')['alphar_data'][0]
Fyf_opt2 = syio.loadmat('MagicSTData2/fyf.mat')['fyf_data'][0]
Fyr_opt2 = syio.loadmat('MagicSTData2/fyr.mat')['fyr_data'][0]
Fzf_opt2 = syio.loadmat('MagicSTData2/fzf.mat')['fzf_data'][0][0]
Fzr_opt2 = syio.loadmat('MagicSTData2/fzr.mat')['fzr_data'][0][0]

pos_x_opt5 = syio.loadmat('MagicSTData3/pos_x.mat')['pos_x_data'][0]
pos_y_opt5 = syio.loadmat('MagicSTData3/pos_y.mat')['pos_y_data'][0]
vel_x_opt5 = syio.loadmat('MagicSTData3/vel_x.mat')['vel_x_data'][0]
vel_y_opt5 = syio.loadmat('MagicSTData3/vel_y.mat')['vel_y_data'][0]
phi_opt5 = syio.loadmat('MagicSTData3/phi.mat')['phi_data'][0]
r_opt5 = syio.loadmat('MagicSTData3/r.mat')['r_data'][0]
delta_opt5 = syio.loadmat('MagicSTData3/delta.mat')['delta_data'][0]
delta_d_opt5 = syio.loadmat('MagicSTData3/delta_d.mat')['delta_d_data'][0]
Fxf_opt5 = syio.loadmat('MagicSTData3/fxf.mat')['fxf_data'][0]
Fxr_opt5 = syio.loadmat('MagicSTData3/fxr.mat')['fxr_data'][0]
T_opt5 = syio.loadmat('MagicSTData3/t.mat')['t_data'][0][0]
alphaf_opt5 = syio.loadmat('MagicSTData3/alphaf.mat')['alphaf_data'][0]
alphar_opt5 = syio.loadmat('MagicSTData3/alphar.mat')['alphar_data'][0]
Fyf_opt5 = syio.loadmat('MagicSTData3/fyf.mat')['fyf_data'][0]
Fyr_opt5 = syio.loadmat('MagicSTData3/fyr.mat')['fyr_data'][0]
Fzf_opt5 = syio.loadmat('MagicSTData3/fzf.mat')['fzf_data'][0][0]
Fzr_opt5 = syio.loadmat('MagicSTData3/fzr.mat')['fzr_data'][0][0]


pos_x_opt3 = syio.loadmat('MagicTimeConstSTData2/pos_x.mat')['pos_x_data'][0]
pos_y_opt3 = syio.loadmat('MagicTimeConstSTData2/pos_y.mat')['pos_y_data'][0]
vel_x_opt3 = syio.loadmat('MagicTimeConstSTData2/vel_x.mat')['vel_x_data'][0]
vel_y_opt3 = syio.loadmat('MagicTimeConstSTData2/vel_y.mat')['vel_y_data'][0]
phi_opt3 = syio.loadmat('MagicTimeConstSTData2/phi.mat')['phi_data'][0]
r_opt3 = syio.loadmat('MagicTimeConstSTData2/r.mat')['r_data'][0]
delta_opt3 = syio.loadmat('MagicTimeConstSTData2/delta.mat')['delta_data'][0]
delta_d_opt3 = syio.loadmat('MagicTimeConstSTData2/delta_d.mat')['delta_d_data'][0]
Fxf_opt3 = syio.loadmat('MagicTimeConstSTData2/fxf.mat')['fxf_data'][0]
Fxr_opt3 = syio.loadmat('MagicTimeConstSTData2/fxr.mat')['fxr_data'][0]
T_opt3 = syio.loadmat('MagicTimeConstSTData2/t.mat')['t_data'][0][0]
alphaf_opt3 = syio.loadmat('MagicTimeConstSTData2/alphaf.mat')['alphaf_data'][0]
alphar_opt3 = syio.loadmat('MagicTimeConstSTData2/alphar.mat')['alphar_data'][0]
Fyf_opt3 = syio.loadmat('MagicTimeConstSTData2/fyf.mat')['fyf_data'][0]
Fyr_opt3 = syio.loadmat('MagicTimeConstSTData2/fyr.mat')['fyr_data'][0]
Fzf_opt3 = syio.loadmat('MagicTimeConstSTData2/fzf.mat')['fzf_data'][0][0]
Fzr_opt3 = syio.loadmat('MagicTimeConstSTData2/fzr.mat')['fzr_data'][0][0]

pos_x_opt6 = syio.loadmat('MagicTimeConstSTData3/pos_x.mat')['pos_x_data'][0]
pos_y_opt6 = syio.loadmat('MagicTimeConstSTData3/pos_y.mat')['pos_y_data'][0]
vel_x_opt6 = syio.loadmat('MagicTimeConstSTData3/vel_x.mat')['vel_x_data'][0]
vel_y_opt6 = syio.loadmat('MagicTimeConstSTData3/vel_y.mat')['vel_y_data'][0]
phi_opt6 = syio.loadmat('MagicTimeConstSTData3/phi.mat')['phi_data'][0]
r_opt6 = syio.loadmat('MagicTimeConstSTData3/r.mat')['r_data'][0]
delta_opt6 = syio.loadmat('MagicTimeConstSTData3/delta.mat')['delta_data'][0]
delta_d_opt6 = syio.loadmat('MagicTimeConstSTData3/delta_d.mat')['delta_d_data'][0]
Fxf_opt6 = syio.loadmat('MagicTimeConstSTData3/fxf.mat')['fxf_data'][0]
Fxr_opt6 = syio.loadmat('MagicTimeConstSTData3/fxr.mat')['fxr_data'][0]
T_opt6 = syio.loadmat('MagicTimeConstSTData3/t.mat')['t_data'][0][0]
alphaf_opt6 = syio.loadmat('MagicTimeConstSTData3/alphaf.mat')['alphaf_data'][0]
alphar_opt6 = syio.loadmat('MagicTimeConstSTData3/alphar.mat')['alphar_data'][0]
Fyf_opt6 = syio.loadmat('MagicTimeConstSTData3/fyf.mat')['fyf_data'][0]
Fyr_opt6 = syio.loadmat('MagicTimeConstSTData3/fyr.mat')['fyr_data'][0]
Fzf_opt6 = syio.loadmat('MagicTimeConstSTData3/fzf.mat')['fzf_data'][0][0]
Fzr_opt6 = syio.loadmat('MagicTimeConstSTData3/fzr.mat')['fzr_data'][0][0]

#Plot position every 5 timeframe small
tt = np.linspace(0, T_opt, len(pos_x_opt))
ttt = np.linspace(0, T_opt, N+1)
tt2 = np.linspace(0, T_opt4, len(pos_x_opt4))
ttt2 = np.linspace(0, T_opt4, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt, pos_y_opt, zorder=1, color = 'blue')
ax.plot(pos_x_opt4, pos_y_opt4, zorder=1, color = 'red')
ax.legend(['Linear', 'LinearMassWet'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
#add rectangle
for p in range(0, len(ttt), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt, phi_opt)
    posxi = np.interp(i, ttt, pos_x_opt)
    posyi = np.interp(i, ttt, pos_y_opt)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'blue', zorder=2)
    ax.add_patch(rect)
for p in range(0, len(ttt2), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt2, phi_opt4)
    posxi = np.interp(i, ttt2, pos_x_opt4)
    posyi = np.interp(i, ttt2, pos_y_opt4)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-7, 7))
plt.grid(True)
plt.title('Paths of Linear and Linear with mass and wet')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')


#Plot position every second small
tt = np.linspace(0, T_opt, len(pos_x_opt))
ttt = np.linspace(0, T_opt, N+1)
tt2 = np.linspace(0, T_opt4, len(pos_x_opt4))
ttt2 = np.linspace(0, T_opt4, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt, pos_y_opt, zorder=1, color = 'blue')
ax.plot(pos_x_opt4, pos_y_opt4, zorder=1, color = 'red')
ax.legend(['Linear', 'LinearMassWet'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
#add rectangle
for i in range(0, int(T_opt+1), 1):
    yawi = np.interp(i, ttt, phi_opt)
    posxi = np.interp(i, ttt, pos_x_opt)
    posyi = np.interp(i, ttt, pos_y_opt)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'blue', zorder=2)
    ax.add_patch(rect)
for i in range(0, int(T_opt2+1), 1):
    yawi = np.interp(i, ttt2, phi_opt4)
    posxi = np.interp(i, ttt2, pos_x_opt4)
    posyi = np.interp(i, ttt2, pos_y_opt4)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-7, 7))
plt.grid(True)
plt.title('Paths of Linear and Linear with mass and wet')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')

#Fxf, Fyf
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt[1:], Fxf_opt, color = 'blue')
plt.plot(tt2[1:], Fxf_opt4, color = 'red')
plt.hlines(y = Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxf [N]')
plt.subplot(322)
plt.plot(tt, Fyf_opt, color = 'blue')
plt.plot(tt2, Fyf_opt4, color = 'red')
plt.hlines(y = Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyf [N]')
plt.subplot(312)
plt.plot(tt[1:], sqrt((Fxf_opt)**2 + (Fyf_opt[1:])**2), color = 'blue')
plt.plot(tt2[1:], sqrt((Fxf_opt4)**2 + (Fyf_opt4[1:])**2), color = 'red')
plt.hlines(y = sqrt((Dxf)**2+(Dyf)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.legend(['Linear', 'LinearMassWet'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Ff [N]')
plt.subplot(313)
plt.plot(Fxf_opt, Fyf_opt[1:], color = 'blue', zorder = 1)
plt.plot(Fxf_opt4, Fyf_opt4[1:], color = 'red', zorder = 1)
ax = plt.gca()
ellipse = Ellipse(xy=(0, 0), width=2*Dxf, height=2*Dyf, edgecolor='black', fc='None', lw=2, linestyle='--', zorder=2)
ax.add_patch(ellipse)
plt.xlim(-Dxf-2000, Dxf+2000)
plt.ylim(-Dyf-10000, Dyf+10000)
plt.grid(True)
plt.xlabel('Fxf [N]')
plt.ylabel('Fyf [N]')

#Fxr, Fyr
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt[1:], Fxr_opt, color = 'blue')
plt.plot(tt2[1:], Fxr_opt4, color = 'red')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxr [N]')
plt.subplot(322)
plt.plot(tt, Fyr_opt, color = 'blue')
plt.plot(tt2, Fyr_opt4, color = 'red')
plt.hlines(y = Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyr [N]')
plt.subplot(312)
plt.plot(tt[1:], sqrt((Fxr_opt)**2 + (Fyr_opt[1:])**2), color = 'blue')
plt.plot(tt2[1:], sqrt((Fxr_opt4)**2 + (Fyr_opt4[1:])**2), color = 'red')
plt.hlines(y = sqrt((Dxr)**2+(Dyr)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.legend(['Linear', 'LinearMassWet'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fr [N]')
plt.subplot(313)
plt.plot(Fxr_opt, Fyr_opt[1:], color = 'blue', zorder = 1)
plt.plot(Fxr_opt2, Fyr_opt4[1:], color = 'red', zorder = 1)
plt.vlines(x = 0, ymin = -Dyr, ymax = Dyr, color = 'black', zorder = 2, linestyle='--')
ax = plt.gca()
ellipse = Ellipse(xy=(0, 0), width=2*Dxr, height=2*Dyr, edgecolor='black', fc='None', lw=2, linestyle='--', zorder=2)
ax.add_patch(ellipse)
plt.xlim(-Dxr-2000, 2000)
plt.ylim(-Dyr-10000, Dyr+10000)
plt.grid(True)
plt.xlabel('Fxr [N]')
plt.ylabel('Fyr [N]')


#Phi, r, delta and delta_d
plt.figure(figsize=(10,4.0))
plt.subplot(411)
plt.plot(tt, phi_opt*180/pi, color = 'blue')
plt.plot(tt2, phi_opt4*180/pi, color = 'red')
plt.legend(['Linear', 'LinearMassWet'])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
plt.grid(True)
plt.title('phi, r, delta and delta_d (yaw, yaw rate, steering angle and steering rate)')
plt.xlabel('time [s]')
plt.ylabel('phi x [deg]')
plt.subplot(412)
plt.plot(tt, r_opt*180/pi, color = 'blue')
plt.plot(tt2, r_opt4*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('r [deg/s]')
plt.subplot(413)
plt.plot(tt, delta_opt*180/pi, color = 'blue')
plt.plot(tt2, delta_opt4*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta [deg]')
plt.subplot(414)
plt.plot(tt[1:], delta_d_opt*180/pi, color = 'blue')
plt.plot(tt2[1:], delta_d_opt4*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta_d [deg/s]')

#alphaf and alphar
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, list(np.array(alphaf_opt)*180/pi), color = 'blue')
plt.plot(tt2, list(np.array(alphaf_opt4)*180/pi), color = 'red')
plt.legend(['Linear', 'LinearMassWet'])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
plt.grid(True)
plt.title('alphaf and alphar')
plt.xlabel('time [s]')
plt.ylabel('alphaf [deg]')
plt.subplot(212)
plt.plot(tt, list(np.array(alphar_opt)*180/pi), color = 'blue')
plt.plot(tt2, list(np.array(alphar_opt4)*180/pi), color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('alphar [deg]')








#Plot position every 5 timeframe small
tt = np.linspace(0, T_opt2, len(pos_x_opt2))
ttt = np.linspace(0, T_opt2, N+1)
tt2 = np.linspace(0, T_opt5, len(pos_x_opt5))
ttt2 = np.linspace(0, T_opt5, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt2, pos_y_opt2, zorder=1, color = 'blue')
ax.plot(pos_x_opt5, pos_y_opt5, zorder=1, color = 'red')
ax.legend(['Magic', 'MagicMassWet'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
#add rectangle
for p in range(0, len(ttt), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt, phi_opt2)
    posxi = np.interp(i, ttt, pos_x_opt2)
    posyi = np.interp(i, ttt, pos_y_opt2)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'blue', zorder=2)
    ax.add_patch(rect)
for p in range(0, len(ttt2), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt2, phi_opt5)
    posxi = np.interp(i, ttt2, pos_x_opt5)
    posyi = np.interp(i, ttt2, pos_y_opt5)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-7, 7))
plt.grid(True)
plt.title('Paths of Magic and Magic with mass and wet')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')


#Plot position every second small
tt = np.linspace(0, T_opt2, len(pos_x_opt2))
ttt = np.linspace(0, T_opt2, N+1)
tt2 = np.linspace(0, T_opt5, len(pos_x_opt5))
ttt2 = np.linspace(0, T_opt5, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt2, pos_y_opt2, zorder=1, color = 'blue')
ax.plot(pos_x_opt5, pos_y_opt5, zorder=1, color = 'red')
ax.legend(['Magic', 'MagicMassWet'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
#add rectangle
for i in range(0, int(T_opt+1), 1):
    yawi = np.interp(i, ttt, phi_opt2)
    posxi = np.interp(i, ttt, pos_x_opt2)
    posyi = np.interp(i, ttt, pos_y_opt2)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'blue', zorder=2)
    ax.add_patch(rect)
for i in range(0, int(T_opt2+1), 1):
    yawi = np.interp(i, ttt2, phi_opt5)
    posxi = np.interp(i, ttt2, pos_x_opt5)
    posyi = np.interp(i, ttt2, pos_y_opt5)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-7, 7))
plt.grid(True)
plt.title('Paths of Magic and Magic with mass and wet')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')

#Fxf, Fyf
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt[1:], Fxf_opt2, color = 'blue')
plt.plot(tt2[1:], Fxf_opt5, color = 'red')
plt.hlines(y = Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxf [N]')
plt.subplot(322)
plt.plot(tt, Fyf_opt2, color = 'blue')
plt.plot(tt2, Fyf_opt5, color = 'red')
plt.hlines(y = Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyf [N]')
plt.subplot(312)
plt.plot(tt[1:], sqrt((Fxf_opt2)**2 + (Fyf_opt2[1:])**2), color = 'blue')
plt.plot(tt2[1:], sqrt((Fxf_opt5)**2 + (Fyf_opt5[1:])**2), color = 'red')
plt.hlines(y = sqrt((Dxf)**2+(Dyf)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.legend(['Magic', 'MagicMassWet'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Ff [N]')
plt.subplot(313)
plt.plot(Fxf_opt2, Fyf_opt2[1:], color = 'blue', zorder = 1)
plt.plot(Fxf_opt5, Fyf_opt5[1:], color = 'red', zorder = 1)
ax = plt.gca()
ellipse = Ellipse(xy=(0, 0), width=2*Dxf, height=2*Dyf, edgecolor='black', fc='None', lw=2, linestyle='--', zorder=2)
ax.add_patch(ellipse)
plt.xlim(-Dxf-2000, Dxf+2000)
plt.ylim(-Dyf-10000, Dyf+10000)
plt.grid(True)
plt.xlabel('Fxf [N]')
plt.ylabel('Fyf [N]')

#Fxr, Fyr
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt[1:], Fxr_opt2, color = 'blue')
plt.plot(tt2[1:], Fxr_opt5, color = 'red')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxr [N]')
plt.subplot(322)
plt.plot(tt, Fyr_opt2, color = 'blue')
plt.plot(tt2, Fyr_opt5, color = 'red')
plt.hlines(y = Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyr [N]')
plt.subplot(312)
plt.plot(tt[1:], sqrt((Fxr_opt2)**2 + (Fyr_opt2[1:])**2), color = 'blue')
plt.plot(tt2[1:], sqrt((Fxr_opt5)**2 + (Fyr_opt5[1:])**2), color = 'red')
plt.hlines(y = sqrt((Dxr)**2+(Dyr)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.legend(['Magic', 'MagicMassWet'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fr [N]')
plt.subplot(313)
plt.plot(Fxr_opt2, Fyr_opt2[1:], color = 'blue', zorder = 1)
plt.plot(Fxr_opt5, Fyr_opt5[1:], color = 'red', zorder = 1)
plt.vlines(x = 0, ymin = -Dyr, ymax = Dyr, color = 'black', zorder = 2, linestyle='--')
ax = plt.gca()
ellipse = Ellipse(xy=(0, 0), width=2*Dxr, height=2*Dyr, edgecolor='black', fc='None', lw=2, linestyle='--', zorder=2)
ax.add_patch(ellipse)
plt.xlim(-Dxr-2000, 2000)
plt.ylim(-Dyr-10000, Dyr+10000)
plt.grid(True)
plt.xlabel('Fxr [N]')
plt.ylabel('Fyr [N]')


#Phi, r, delta and delta_d
plt.figure(figsize=(10,4.0))
plt.subplot(411)
plt.plot(tt, phi_opt2*180/pi, color = 'blue')
plt.plot(tt2, phi_opt5*180/pi, color = 'red')
plt.legend(['Magic', 'MagicMassWet'])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
plt.grid(True)
plt.title('phi, r, delta and delta_d (yaw, yaw rate, steering angle and steering rate)')
plt.xlabel('time [s]')
plt.ylabel('phi x [deg]')
plt.subplot(412)
plt.plot(tt, r_opt2*180/pi, color = 'blue')
plt.plot(tt2, r_opt5*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('r [deg/s]')
plt.subplot(413)
plt.plot(tt, delta_opt2*180/pi, color = 'blue')
plt.plot(tt2, delta_opt5*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta [deg]')
plt.subplot(414)
plt.plot(tt[1:], delta_d_opt2*180/pi, color = 'blue')
plt.plot(tt2[1:], delta_d_opt5*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta_d [deg/s]')

#alphaf and alphar
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, list(np.array(alphaf_opt2)*180/pi), color = 'blue')
plt.plot(tt2, list(np.array(alphaf_opt5)*180/pi), color = 'red')
plt.legend(['Magic', 'MagicMassWet'])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
plt.grid(True)
plt.title('alphaf and alphar')
plt.xlabel('time [s]')
plt.ylabel('alphaf [deg]')
plt.subplot(212)
plt.plot(tt, list(np.array(alphar_opt2)*180/pi), color = 'blue')
plt.plot(tt2, list(np.array(alphar_opt5)*180/pi), color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('alphar [deg]')







#Plot position every 5 timeframe small
tt = np.linspace(0, T_opt2, len(pos_x_opt2))
ttt = np.linspace(0, T_opt2, N+1)
tt2 = np.linspace(0, T_opt5, len(pos_x_opt5))
ttt2 = np.linspace(0, T_opt5, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt2, pos_y_opt2, zorder=1, color = 'blue')
ax.plot(pos_x_opt5, pos_y_opt5, zorder=1, color = 'red')
ax.legend(['FilterMagic', 'FilterMagicWet'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
#add rectangle
for p in range(0, len(ttt), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt, phi_opt2)
    posxi = np.interp(i, ttt, pos_x_opt2)
    posyi = np.interp(i, ttt, pos_y_opt2)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'blue', zorder=2)
    ax.add_patch(rect)
for p in range(0, len(ttt2), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt2, phi_opt5)
    posxi = np.interp(i, ttt2, pos_x_opt5)
    posyi = np.interp(i, ttt2, pos_y_opt5)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-7, 7))
plt.grid(True)
plt.title('Paths of FilterMagic and FilterMagic with mass and wet')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')


#Plot position every second small
tt = np.linspace(0, T_opt3, len(pos_x_opt3))
ttt = np.linspace(0, T_opt3, N+1)
tt2 = np.linspace(0, T_opt6, len(pos_x_opt6))
ttt2 = np.linspace(0, T_opt6, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt3, pos_y_opt3, zorder=1, color = 'blue')
ax.plot(pos_x_opt6, pos_y_opt6, zorder=1, color = 'red')
ax.legend(['FilterMagic', 'FilterMagicMassWet'])
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
#add rectangle
for i in range(0, int(T_opt+1), 1):
    yawi = np.interp(i, ttt, phi_opt3)
    posxi = np.interp(i, ttt, pos_x_opt3)
    posyi = np.interp(i, ttt, pos_y_opt3)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'blue', zorder=2)
    ax.add_patch(rect)
for i in range(0, int(T_opt2+1), 1):
    yawi = np.interp(i, ttt2, phi_opt6)
    posxi = np.interp(i, ttt2, pos_x_opt6)
    posyi = np.interp(i, ttt2, pos_y_opt6)
    w = 0.2
    opposite_x = posxi + (Params.l.value+2)*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + (Params.l.value+2)*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), Params.l.value+2, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-7, 7))
plt.grid(True)
plt.title('Paths of FilterMagic and FilterMagic with mass')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')

#Fxf, Fyf
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt, Fxf_opt3, color = 'blue')
plt.plot(tt2, Fxf_opt6, color = 'red')
plt.hlines(y = Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxf [N]')
plt.subplot(322)
plt.plot(tt, Fyf_opt3, color = 'blue')
plt.plot(tt2, Fyf_opt6, color = 'red')
plt.hlines(y = Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyf [N]')
plt.subplot(312)
plt.plot(tt, sqrt((Fxf_opt3)**2 + (Fyf_opt3)**2), color = 'blue')
plt.plot(tt2, sqrt((Fxf_opt6)**2 + (Fyf_opt6)**2), color = 'red')
plt.hlines(y = sqrt((Dxf)**2+(Dyf)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.legend(['FilterMagic', 'FilterMagicMassWet'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Ff [N]')
plt.subplot(313)
plt.plot(Fxf_opt3, Fyf_opt3, color = 'blue', zorder = 1)
plt.plot(Fxf_opt6, Fyf_opt6, color = 'red', zorder = 1)
ax = plt.gca()
ellipse = Ellipse(xy=(0, 0), width=2*Dxf, height=2*Dyf, edgecolor='black', fc='None', lw=2, linestyle='--', zorder=2)
ax.add_patch(ellipse)
plt.xlim(-Dxf-2000, Dxf+2000)
plt.ylim(-Dyf-10000, Dyf+10000)
plt.grid(True)
plt.xlabel('Fxf [N]')
plt.ylabel('Fyf [N]')

#Fxr, Fyr
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt, Fxr_opt3, color = 'blue')
plt.plot(tt2, Fxr_opt6, color = 'red')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxr [N]')
plt.subplot(322)
plt.plot(tt, Fyr_opt3, color = 'blue')
plt.plot(tt2, Fyr_opt6, color = 'red')
plt.hlines(y = Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyr [N]')
plt.subplot(312)
plt.plot(tt, sqrt((Fxr_opt3)**2 + (Fyr_opt3)**2), color = 'blue')
plt.plot(tt2, sqrt((Fxr_opt6)**2 + (Fyr_opt6)**2), color = 'red')
plt.hlines(y = sqrt((Dxr)**2+(Dyr)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.legend(['FilterMagic', 'FilterMagicMassWet'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fr [N]')
plt.subplot(313)
plt.plot(Fxr_opt3, Fyr_opt3, color = 'blue', zorder = 1)
plt.plot(Fxr_opt6, Fyr_opt6, color = 'red', zorder = 1)
plt.vlines(x = 0, ymin = -Dyr, ymax = Dyr, color = 'black', zorder = 2, linestyle='--')
ax = plt.gca()
ellipse = Ellipse(xy=(0, 0), width=2*Dxr, height=2*Dyr, edgecolor='black', fc='None', lw=2, linestyle='--', zorder=2)
ax.add_patch(ellipse)
plt.xlim(-Dxr-2000, 2000)
plt.ylim(-Dyr-10000, Dyr+10000)
plt.grid(True)
plt.xlabel('Fxr [N]')
plt.ylabel('Fyr [N]')


#Phi, r, delta and delta_d
plt.figure(figsize=(10,4.0))
plt.subplot(411)
plt.plot(tt, phi_opt3*180/pi, color = 'blue')
plt.plot(tt2, phi_opt6*180/pi, color = 'red')
plt.legend(['FilterMagic', 'FilterMagicMassWet'])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
plt.grid(True)
plt.title('phi, r, delta and delta_d (yaw, yaw rate, steering angle and steering rate)')
plt.xlabel('time [s]')
plt.ylabel('phi x [deg]')
plt.subplot(412)
plt.plot(tt, r_opt3*180/pi, color = 'blue')
plt.plot(tt2, r_opt6*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('r [deg/s]')
plt.subplot(413)
plt.plot(tt, delta_opt3*180/pi, color = 'blue')
plt.plot(tt2, delta_opt6*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta [deg]')
plt.subplot(414)
plt.plot(tt[1:], delta_d_opt3*180/pi, color = 'blue')
plt.plot(tt2[1:], delta_d_opt6*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta_d [deg/s]')

#alphaf and alphar
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, list(np.array(alphaf_opt3)*180/pi), color = 'blue')
plt.plot(tt2, list(np.array(alphaf_opt6)*180/pi), color = 'red')
plt.legend(['FilterMagic', 'FilterMagicMassWet'])
ax = plt.gca()
leg = ax.get_legend()
leg.legend_handles[0].set_color('blue')
leg.legend_handles[1].set_color('red')
plt.grid(True)
plt.title('alphaf and alphar')
plt.xlabel('time [s]')
plt.ylabel('alphaf [deg]')
plt.subplot(212)
plt.plot(tt, list(np.array(alphar_opt3)*180/pi), color = 'blue')
plt.plot(tt2, list(np.array(alphar_opt6)*180/pi), color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('alphar [deg]')

"""
#Velocities
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, vel_x_opt*3.6)
plt.plot(tt2, vel_x_opt2*3.6)
plt.plot(tt3, vel_x_opt3*3.6)
plt.legend(['Linear', 'Magic', 'MagicTimeConst'])
plt.grid(True)
plt.title('Speed of all models')
plt.xlabel('Time [s]')
plt.ylabel('Speed x [km/h]')
plt.subplot(212)
plt.plot(tt, vel_y_opt*3.6)
plt.plot(tt2, vel_y_opt2*3.6)
plt.plot(tt3, vel_y_opt3*3.6)
plt.legend(['Linear', 'Magic', 'MagicTimeConst'])
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Speed y [km/h]')
"""

"""
#Fxf, Fxr, Fyf and Fyr
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt[1:], Fxf_opt)
plt.plot(tt[1:], Fxr_opt)
plt.plot(tt2[1:], Fxf_opt2)
plt.plot(tt2[1:], Fxr_opt2)
plt.plot(tt3, Fxf_opt3)
plt.plot(tt3, Fxr_opt3)
plt.legend(['Fxf_linear', 'Fxr_linear', 'Fxf_magic', 'Fxr_magic', 'Fxf_timeconstmagic', 'Fxr_timeconstmagic'])
plt.grid(True)
plt.title('Fxf, Fxr, Fyf and Fyr')
plt.xlabel('time [s]')
plt.ylabel('Fxf, Fxr [N]')
plt.subplot(212)
plt.plot(tt, Fyf_opt)
plt.plot(tt, Fyr_opt)
plt.plot(tt2, Fyf_opt2)
plt.plot(tt2, Fyr_opt2)
plt.plot(tt3, Fyf_opt3)
plt.plot(tt3, Fyr_opt3)
plt.legend(['Fyf_linear', 'Fyr_linear', 'Fyf_magic', 'Fyr_magic', 'Fyf_timeconstmagic', 'Fyr_timeconstmagic'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyf, Fyr [N]')
"""

"""
#Velocity 
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, sqrt((vel_x_opt**2)+vel_y_opt**2)*3.6)
plt.plot(tt2, sqrt((vel_x_opt2**2)+vel_y_opt2**2)*3.6)
plt.plot(tt3, sqrt((vel_x_opt3**2)+vel_y_opt3**2)*3.6)
plt.legend(['Linear', 'Magic', 'MagicTimeConst'])
plt.grid(True)
plt.title('velocity')
plt.xlabel('time [s]')
plt.ylabel('v [km/h]')
"""

"""
#Tire force Fyf/Fzf and Fyr/Fzr 
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(list(np.array(alphaf_opt)*180/pi), list(np.array(Fyf_opt)/Fzf_opt))
plt.plot(list(np.array(alphaf_opt2)*180/pi), list(np.array(Fyf_opt2)/Fzf_opt2))
plt.plot(list(np.array(alphaf_opt3)*180/pi), list(np.array(Fyf_opt3)/Fzf_opt3))
plt.legend(['Linear', 'Magic', 'MagicTimeConst'])
plt.grid(True)
plt.title('Fyf/Fzf and Fyr/Fzr')
plt.xlabel('alphaf [deg]')
plt.ylabel('Fyf/Fzf [N]')
plt.subplot(212)
plt.plot(list(np.array(alphar_opt)*180/pi), list(np.array(Fyr_opt)/Fzr_opt))
plt.plot(list(np.array(alphar_opt2)*180/pi), list(np.array(Fyr_opt2)/Fzr_opt2))
plt.plot(list(np.array(alphar_opt3)*180/pi), list(np.array(Fyr_opt3)/Fzr_opt3))
plt.legend(['Linear', 'Magic', 'MagicTimeConst'])
plt.grid(True)
plt.xlabel('alphar [deg]')
plt.ylabel('Fyr/Fzr [N]')
"""


plt.show()

# %%