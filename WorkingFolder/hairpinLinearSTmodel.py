#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


# Run file for the linear tire model. Use the read_from_file variable to be able to use the saved
# as a guess for current problem. Use the actual_car_size variable to use the set width and length of
# the car, otherwise a point at the center of mass is used. Set the friction variable to the wanted 
# friction. 

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


read_from_file = True #Read previous solutions from file to help solver
actual_car_size = True #Use the car length and width otherwise it is seen as a point
friction = "dry" #Friction used

#Elipse constants
R1 = 10 #diff from obstacle 1 center in x
R2 = 5 #height of obstacle 1
R3 = 60 #diff from obstacle 2 center in x
R4 = 10 #height of obstacle 2
obstacle_center_x = 0 #OBSTACLE CENTER X
obstacle_center_y = 0 #OBSTACLE CENTER Y
road_width = R4-R2
middle_of_road = obstacle_center_y+R2+road_width/2

# ---- Parameters to model ----
match friction:
    case "dry":
        class Params(Enum):
            m = 2100
            lf = 1.3
            lr = 1.5
            g = 9.82
            l = lf + lr
            w = 1.8
            Izz = m*((l+2)**2)/12
            h = 0.5

            #Dry
            muxf = 1.2
            muxr = 1.2
            muyf = 0.935
            muyr = 0.961
    case "wet":
        class Params(Enum):
            m = 2100
            lf = 1.3
            lr = 1.5
            g = 9.82
            l = lf + lr
            w = 1.8
            Izz = m*((l+2)**2)/12
            h = 0.5

            #Wet
            muxf = 1.06
            muxr = 1.07
            muyf = 0.885
            muyr = 0.911
    case "snow":
        class Params(Enum):
            m = 2100
            lf = 1.3
            lr = 1.5
            g = 9.82
            l = lf + lr
            w = 1.8
            Izz = m*((l+2)**2)/12
            h = 0.5

            #Snow
            muxf = 0.407
            muxr = 0.409
            muyf = 0.383
            muyr = 0.394
    case "snowi":
        class Params(Enum):
            m = 2100
            lf = 1.3
            lr = 1.5
            g = 9.82
            l = lf + lr
            w = 1.8
            Izz = m*((l+2)**2)/12
            h = 0.5

            #Snow
            muxf = 0.707
            muxr = 0.709
            muyf = 0.683
            muyr = 0.694
    case "ice":
        class Params(Enum):
            m = 2100
            lf = 1.3
            lr = 1.5
            g = 9.82
            l = lf + lr
            w = 1.8
            Izz = m*((l+2)**2)/12
            h = 0.5

            #Ice
            muxf = 0.172
            muxr = 0.173
            muyf = 0.162
            muyr = 0.167

upper_corner_of_road = obstacle_center_y+R4-(Params.w.value/2)-0.1

# ---- Start values to solver ----
if(read_from_file == False):
    print("normal start values")
    start_pos_x_init = obstacle_center_x+5
    start_pos_y_init = upper_corner_of_road
    start_vel_x_init = 25/3.6
    start_vel_y_init = 0
    start_phi_init = 0
    start_r_init = 0
    start_delta_init = 0
    start_delta_d_init = 0
    start_Fxf_init = 100
    start_Fxr_init = 0
    start_T_init = 2.4
else:
    print("loading from file")
    start_pos_x_init = syio.loadmat('LinearSTData/pos_x.mat')['pos_x_data'][0]
    start_pos_y_init = syio.loadmat('LinearSTData/pos_y.mat')['pos_y_data'][0]
    start_vel_x_init = syio.loadmat('LinearSTData/vel_x.mat')['vel_x_data'][0]
    start_vel_y_init = syio.loadmat('LinearSTData/vel_y.mat')['vel_y_data'][0]
    start_phi_init = syio.loadmat('LinearSTData/phi.mat')['phi_data'][0]
    start_r_init = syio.loadmat('LinearSTData/r.mat')['r_data'][0]
    start_delta_init = syio.loadmat('LinearSTData/delta.mat')['delta_data'][0]
    start_delta_d_init = syio.loadmat('LinearSTData/delta_d.mat')['delta_d_data'][0]
    start_Fxf_init = syio.loadmat('LinearSTData/fxf.mat')['fxf_data'][0]
    start_Fxr_init = syio.loadmat('LinearSTData/fxr.mat')['fxr_data'][0]
    start_T_init = syio.loadmat('LinearSTData/t.mat')['t_data'][0]

# ---- Actual start values ----
start_pos_x = obstacle_center_x+5
start_pos_y = upper_corner_of_road
start_vel_x = 25/3.6
start_vel_y = 0
start_phi = 0
end_phi = -pi
end_r = 0
end_delta = 0
end_delta_d = 0
# ---- End values ----
end_pos_x = start_pos_x+5
end_pos_y = -middle_of_road

# ---- Steering constraints ----
delta_max = pi/6
delta_d_max = pi/4

N = 100
opti = Opti()

# ---- Decision variables ----
X = opti.variable(7, N+1) # state trajectory
pos_x = X[0, :] 
pos_y = X[1, :]
vel_x = X[2, :]
vel_y = X[3, :]
phi = X[4, :] #yaw 
r = X[5, :] #yaw rate
delta = X[6, :] #steering angle

U = opti.variable(3, N)   # control trajectory (steering rate and front wheel force)
delta_d = U[0, :] #steering rate
Fxf = U[1, :] #Long. front tire force
Fxr = U[2, :] #Long. rear tire force
T = opti.variable(1) #final time

# ---- Objective ----
opti.minimize(T) #minimize time
#opti.minimize(-vel_x[-1]) #maximize vel_x

# ---- Dynamic constraints -----
dt = T/N # length of a control interval
for k in range(N): # loop over control intervals 
   # Runge-Kutta 4 integration                      
   k1 = linearST(X[:,k],         U[:,k], Params)[0]
   k2 = linearST(X[:,k]+dt/2*k1, U[:,k], Params)[0]
   k3 = linearST(X[:,k]+dt/2*k2, U[:,k], Params)[0]
   k4 = linearST(X[:,k]+dt*k3,   U[:,k], Params)[0]
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4); 
   opti.subject_to(X[:,k+1]==x_next) # close the gaps
   
# ---- Obstacle constraint ----
for k in range(N+1):
    top_right_x = pos_x[k] + (((Params.lf.value+1)/2)*cos(phi[k])) - ((Params.w.value/2)*sin(phi[k]))
    top_right_y = pos_y[k] + (((Params.lf.value+1)/2)*sin(phi[k])) + ((Params.w.value/2)*cos(phi[k]))
    top_left_x = pos_x[k] - (((Params.lr.value+1)/2)*cos(phi[k])) - ((Params.w.value/2)*sin(phi[k]))
    top_left_y = pos_y[k] - (((Params.lr.value+1)/2)*sin(phi[k])) + ((Params.w.value/2)*cos(phi[k]))
    bottom_left_x = pos_x[k] - (((Params.lr.value+1)/2)*cos(phi[k])) + ((Params.w.value/2)*sin(phi[k]))
    bottom_left_y = pos_y[k] - (((Params.lf.value+1)/2)*sin(phi[k])) - ((Params.w.value/2)*cos(phi[k]))
    bottom_right_x = pos_x[k] + (((Params.lf.value+1)/2)*cos(phi[k])) + ((Params.w.value/2)*sin(phi[k]))
    bottom_right_y = pos_y[k] + (((Params.lf.value+1)/2)*sin(phi[k])) - ((Params.w.value/2)*cos(phi[k]))

    if actual_car_size:
        opti.subject_to(((top_left_x-obstacle_center_x)/R1)**6 + ((top_left_y-obstacle_center_y)/R2)**6 >= 1) #upper left corner
        opti.subject_to(((bottom_left_x-obstacle_center_x)/R1)**6 + ((bottom_left_y-obstacle_center_y)/R2)**6 >= 1) #lower left corner
        opti.subject_to(((top_right_x-obstacle_center_x)/R1)**6 + ((top_right_y-obstacle_center_y)/R2)**6 >= 1) #upper right corner
        opti.subject_to(((bottom_right_x-obstacle_center_x)/R1)**6 + ((bottom_right_y-obstacle_center_y)/R2)**6 >= 1) #lower right corner
        opti.subject_to(((top_left_x-obstacle_center_x)/R3)**6 + ((top_left_y-obstacle_center_y)/R4)**6 <= 1) #upper left corner
        opti.subject_to(((bottom_left_x-obstacle_center_x)/R3)**6 + ((bottom_left_y-obstacle_center_y)/R4)**6 <= 1) #lower left corner
        opti.subject_to(((top_right_x-obstacle_center_x)/R3)**6 + ((top_right_y-obstacle_center_y)/R4)**6 <= 1) #upper right corner
        opti.subject_to(((bottom_right_x-obstacle_center_x)/R3)**6 + ((bottom_right_y-obstacle_center_y)/R4)**6 <= 1) #upper left corner
    else:
        opti.subject_to((((pos_x[k])-obstacle_center_x)/R1)**6 + (((pos_y[k])-obstacle_center_y)/R2)**6 >= 1) #center
        opti.subject_to((((pos_x[k])-obstacle_center_x)/R3)**6 + (((pos_y[k])-obstacle_center_y)/R4)**6 <= 1) #center

# ---- Initial and boundary conditions ----
if not read_from_file:
    opti.subject_to(pos_x[0] == start_pos_x)      
    opti.subject_to(pos_y[0] == start_pos_y)       

    opti.subject_to(pos_x[-1] == end_pos_x) 
    opti.subject_to(pos_y[-1] == end_pos_y)      

    opti.subject_to(vel_x[0] == start_vel_x)  
    opti.subject_to(vel_y[0] == start_vel_y)           

    opti.subject_to(phi[0] == start_phi)        
    #opti.subject_to(phi[-1] == end_phi)           

    #opti.subject_to(r[-1] == end_r)        

    #opti.subject_to(delta[-1] == end_delta)        

    #opti.subject_to(delta_d[-1] == end_delta_d) 
else:
    opti.subject_to(pos_x[0] == start_pos_x)      
    opti.subject_to(pos_y[0] == start_pos_y)       

    opti.subject_to(pos_x[-1] == end_pos_x) 
    opti.subject_to(pos_y[-1] < 0)      

    opti.subject_to(vel_x[0] == start_vel_x)  
    opti.subject_to(vel_y[0] == start_vel_y)           

    opti.subject_to(phi[0] == start_phi)        
    #opti.subject_to(phi[-1] == end_phi)           

    #opti.subject_to(r[-1] == end_r)        

    #opti.subject_to(delta[-1] == end_delta)        

    #opti.subject_to(delta_d[-1] == end_delta_d) 

# ---- Misc. constraints ----
opti.subject_to(T >= 0) # Time must be positive
opti.subject_to(vel_x > 0) # vel_x not be equal 0

# ---- Steering constrains ----
opti.subject_to(opti.bounded(-delta_max,delta,delta_max))    # steering angle limit
opti.subject_to(opti.bounded(-delta_d_max,delta_d,delta_d_max)) # steering rate limit

# ---- Force constrains ----
Fzfs_max = Params.m.value * Params.g.value * Params.lr.value / (Params.lf.value + Params.lr.value)
Fzrs_max = Params.m.value * Params.g.value * Params.lf.value / (Params.lf.value + Params.lr.value)
Dxf = Params.muxf.value * Fzfs_max
Dxr = Params.muxr.value * Fzrs_max  
opti.subject_to(opti.bounded(-Dxf,Fxf,Dxf)) # front wheel driven  
opti.subject_to(opti.bounded(-Dxr,Fxr,0))      

# ---- Initial values for solver ---
opti.set_initial(pos_x, start_pos_x_init)
opti.set_initial(pos_y, start_pos_y_init)

opti.set_initial(vel_x, start_vel_x_init)
opti.set_initial(vel_y, start_vel_y_init)

opti.set_initial(r, start_r_init)

opti.set_initial(delta, start_delta_init)
opti.set_initial(delta_d, start_delta_d_init)

opti.set_initial(phi, start_phi_init)

opti.set_initial(T, start_T_init)

opti.set_initial(Fxf, start_Fxf_init)
opti.set_initial(Fxr, start_Fxr_init)

# ---- Solve NLP ----
opti.solver('ipopt', {'expand': True},
            {'tol': 10**-3, 'print_level': 5, 'max_iter': 3000})
sol = opti.solve()   # actual solve

# ---- Plotting ----
# Extract solution trajectories and store them in the mprim variable
pos_x_opt = sol.value(pos_x)
vel_x_opt = sol.value(vel_x)
pos_y_opt = sol.value(pos_y)
vel_y_opt = sol.value(vel_y)
phi_opt = sol.value(phi)
r_opt = sol.value(r)
delta_opt = sol.value(delta)
u_opt = sol.value(U)
T_opt = sol.value(T)
print("time [s]", T_opt)
print("vel_x_end [km/h]", vel_x_opt[-1]*3.6)
print("highest phi [s]", max(phi_opt)*180/pi)
print("highest r [s]", max(r_opt)*180/pi)
for i in phi_opt:
    print(i*180/pi)
# ---- Extract variables for plotting ----
F_X_opt = []
F_Y_opt = []
M_Z_opt = []
Fyf_opt = []
Fyr_opt = []
alphaf_opt = []
alphar_opt = []
vxf_opt = []
vxr_opt = []
vyf_opt = []
vyr_opt = []
Dyf = Params.muyf.value * Fzfs_max
Dyr = Params.muyr.value * Fzrs_max
for i in range(N+1):
    variable_list = linearST(sol.value(X[:, i]), sol.value(U[:, 0]), Params)[1]
    F_X_opt.append(variable_list[0])
    F_Y_opt.append(variable_list[1])
    M_Z_opt.append(variable_list[2])
    Fyf_opt.append(variable_list[3])
    Fyr_opt.append(variable_list[4])
    alphaf_opt.append(variable_list[5])
    alphar_opt.append(variable_list[6])
    vxf_opt.append(variable_list[7])
    vxr_opt.append(variable_list[8])
    vyf_opt.append(variable_list[9])
    vyr_opt.append(variable_list[10])


# Position every 5 times stamp
tt = np.linspace(0, T_opt, len(pos_x_opt))
ttt = np.linspace(0, T_opt, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt, pos_y_opt, zorder=1)
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
plt.ylim((-5, 5))
plt.grid(True)
plt.title('Scandinavian Flick')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')

#Plot position every second
tt = np.linspace(0, T_opt, len(pos_x_opt))
ttt = np.linspace(0, T_opt, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_opt, pos_y_opt, zorder=1)
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
plt.ylim((-5, 5))
plt.grid(True)
plt.title('Scandinavian Flick')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')

#Fxf, Fyf
plt.figure(figsize=(10,4.0))
plt.subplot(321)
plt.plot(tt[1:], u_opt[1], color = 'red')
plt.hlines(y = Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxf [N]')
plt.subplot(322)
plt.plot(tt, Fyf_opt, color = 'red')
plt.hlines(y = Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyf, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyf [N]')
plt.subplot(312)
plt.plot(tt[1:], sqrt((u_opt[1])**2 + list((np.array(Fyf_opt[1:]))**2)), color = 'red')
plt.hlines(y = sqrt((Dxf)**2+(Dyf)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Ff [N]')
plt.subplot(313)
plt.plot(u_opt[1], Fyf_opt[1:], color = 'red', zorder=1)
#plt.hlines(y = Dyf, xmin = -Dxf, xmax = Dxf, color = 'purple', zorder = 2)
#plt.hlines(y = -Dyf, xmin = -Dxf, xmax = Dxf, color = 'purple', zorder = 2)
#plt.vlines(x = Dxf, ymin = -Dyf, ymax = Dyf, color = 'purple', zorder = 2)
#plt.vlines(x = -Dxf, ymin = -Dyf, ymax = Dyf, color = 'purple', zorder = 2)
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
plt.plot(tt[1:], u_opt[2], color = 'red')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dxr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fxr [N]')
plt.subplot(322)
plt.plot(tt, Fyr_opt, color = 'red')
plt.hlines(y = Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = -Dyr, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fyr [N]')
plt.subplot(312)
plt.plot(tt[1:], sqrt((u_opt[2])**2 + list((np.array(Fyr_opt[1:]))**2)), color = 'red')
plt.hlines(y = sqrt((Dxr)**2+(Dyr)**2), xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.hlines(y = 0, xmin = 0, xmax = tt[-1], color = 'black', zorder = 2, linestyle='--')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('Fr [N]')
plt.subplot(313)
plt.plot(u_opt[2], Fyr_opt[1:], color = 'red', zorder=1)
#plt.hlines(y = Dyr, xmin = -Dxr, xmax = 0, color = 'purple', zorder = 2)
#plt.hlines(y = -Dyr, xmin = -Dxr, xmax = 0, color = 'purple', zorder = 2)
#plt.vlines(x = 0, ymin = -Dyr, ymax = Dyr, color = 'purple', zorder = 2)
#plt.vlines(x = -Dxr, ymin = -Dyr, ymax = Dyr, color = 'purple', zorder = 2)
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
plt.plot(tt, phi_opt*180/pi, color = 'red')
plt.grid(True)
plt.title('phi, r, delta and delta_d (yaw, yaw rate, steering angle and steering rate)')
plt.xlabel('time [s]')
plt.ylabel('phi x [deg]')
plt.subplot(412)
plt.plot(tt, r_opt*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('r [deg/s]')
plt.subplot(413)
plt.plot(tt, delta_opt*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta [deg]')
plt.subplot(414)
plt.plot(tt[1:], u_opt[0]*180/pi, color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('delta_d [deg/s]')

#alphaf and alphar
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, list(np.array(alphaf_opt)*180/pi), color = 'red')
plt.grid(True)
plt.title('alphaf and alphar')
plt.xlabel('time [s]')
plt.ylabel('alphaf [deg]')
plt.subplot(212)
plt.plot(tt, list(np.array(alphar_opt)*180/pi), color = 'red')
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('alphar [deg]')

"""
#Velocities
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, vel_x_opt*3.6)
plt.grid(True)
plt.title('Speed of linear ST Model')
plt.xlabel('Time [s]')
plt.ylabel('Speed x [km/h]')
plt.subplot(212)
plt.plot(tt, vel_y_opt*3.6)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Speed y [km/h]')

#vxf, vxr, vyf and vyr
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, list(np.array(vxf_opt)*3.6))
plt.plot(tt, list(np.array(vxr_opt)*3.6))
plt.legend(['vxf', 'vxr'])
plt.grid(True)
plt.title('vxf, vxr, vyf and vyr')
plt.xlabel('time [s]')
plt.ylabel('vxf, vxr [km/h]')
plt.subplot(212)
plt.plot(tt, list(np.array(vyf_opt)*3.6))
plt.plot(tt, list(np.array(vyr_opt)*3.6))
plt.legend(['vyf', 'vyr'])
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('vyf, vyr [km/h]')

#F_X, F_Y, M_Z
plt.figure(figsize=(10,4.0))
plt.subplot(311)
plt.plot(tt, F_X_opt)
plt.grid(True)
plt.title('F_X, F_Y and M_Z')
plt.xlabel('time [s]')
plt.ylabel('F_X [N]')
plt.subplot(312)
plt.plot(tt, F_Y_opt)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('F_Y [N]')
plt.subplot(313)
plt.plot(tt, M_Z_opt)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('M_Z [Nm]')

#Velocity 
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(tt, sqrt((vel_x_opt**2)+vel_y_opt**2)*3.6)
plt.grid(True)
plt.title('velocity')
plt.xlabel('time [s]')
plt.ylabel('v [km/h]')

#Tire force Fyf/Fzf and Fyr/Fzr 
plt.figure(figsize=(10,4.0))
plt.subplot(211)
plt.plot(list(np.array(alphaf_opt)*180/pi), list(np.array(Fyf_opt)/Fzfs_max))
plt.grid(True)
plt.title('Fyf/Fzf and Fyr/Fzr')
plt.xlabel('alphaf [deg]')
plt.ylabel('Fyf/Fzf [N]')
plt.subplot(212)
plt.plot(list(np.array(alphar_opt)*180/pi), list(np.array(Fyr_opt)/Fzrs_max))
plt.grid(True)
plt.xlabel('alphar [deg]')
plt.ylabel('Fyr/Fzr [N]')

"""

plt.show()

# %%
# ---- Save values to file ----
pos_x_deb = opti.debug.value(pos_x)
vel_x_deb = opti.debug.value(vel_x)
pos_y_deb = opti.debug.value(pos_y)
vel_y_deb = opti.debug.value(vel_y)
phi_deb = opti.debug.value(phi)
r_deb = opti.debug.value(r)
delta_deb = opti.debug.value(delta)
u_deb = opti.debug.value(U)
T_deb = opti.debug.value(T)

syio.savemat('LinearSTData/pos_x.mat', {'pos_x_data': pos_x_deb})
syio.savemat('LinearSTData/pos_y.mat', {'pos_y_data': pos_y_deb})
syio.savemat('LinearSTData/vel_x.mat', {'vel_x_data': vel_x_deb})
syio.savemat('LinearSTData/vel_y.mat', {'vel_y_data': vel_y_deb})
syio.savemat('LinearSTData/phi.mat', {'phi_data': phi_deb})
syio.savemat('LinearSTData/r.mat', {'r_data': r_deb})
syio.savemat('LinearSTData/delta.mat', {'delta_data': delta_deb})
syio.savemat('LinearSTData/delta_d.mat', {'delta_d_data': u_deb[0]})
syio.savemat('LinearSTData/fxf.mat', {'fxf_data': u_deb[1]})
syio.savemat('LinearSTData/fxr.mat', {'fxr_data': u_deb[2]})
syio.savemat('LinearSTData/t.mat', {'t_data': T_deb})
syio.savemat('LinearSTData/alphaf.mat', {'alphaf_data': alphaf_opt})
syio.savemat('LinearSTData/alphar.mat', {'alphar_data': alphar_opt})
syio.savemat('LinearSTData/fyf.mat', {'fyf_data': Fyf_opt})
syio.savemat('LinearSTData/fyr.mat', {'fyr_data': Fyr_opt})
syio.savemat('LinearSTData/fzf.mat', {'fzf_data': Fzfs_max})
syio.savemat('LinearSTData/fzr.mat', {'fzr_data': Fzrs_max})
# %%
# ---- Save values to file 2----
pos_x_deb = opti.debug.value(pos_x)
vel_x_deb = opti.debug.value(vel_x)
pos_y_deb = opti.debug.value(pos_y)
vel_y_deb = opti.debug.value(vel_y)
phi_deb = opti.debug.value(phi)
r_deb = opti.debug.value(r)
delta_deb = opti.debug.value(delta)
u_deb = opti.debug.value(U)
T_deb = opti.debug.value(T)

syio.savemat('LinearSTData2/pos_x.mat', {'pos_x_data': pos_x_deb})
syio.savemat('LinearSTData2/pos_y.mat', {'pos_y_data': pos_y_deb})
syio.savemat('LinearSTData2/vel_x.mat', {'vel_x_data': vel_x_deb})
syio.savemat('LinearSTData2/vel_y.mat', {'vel_y_data': vel_y_deb})
syio.savemat('LinearSTData2/phi.mat', {'phi_data': phi_deb})
syio.savemat('LinearSTData2/r.mat', {'r_data': r_deb})
syio.savemat('LinearSTData2/delta.mat', {'delta_data': delta_deb})
syio.savemat('LinearSTData2/delta_d.mat', {'delta_d_data': u_deb[0]})
syio.savemat('LinearSTData2/fxf.mat', {'fxf_data': u_deb[1]})
syio.savemat('LinearSTData2/fxr.mat', {'fxr_data': u_deb[2]})
syio.savemat('LinearSTData2/t.mat', {'t_data': T_deb})
syio.savemat('LinearSTData2/alphaf.mat', {'alphaf_data': alphaf_opt})
syio.savemat('LinearSTData2/alphar.mat', {'alphar_data': alphar_opt})
syio.savemat('LinearSTData2/fyf.mat', {'fyf_data': Fyf_opt})
syio.savemat('LinearSTData2/fyr.mat', {'fyr_data': Fyr_opt})
syio.savemat('LinearSTData2/fzf.mat', {'fzf_data': Fzfs_max})
syio.savemat('LinearSTData2/fzr.mat', {'fzr_data': Fzrs_max})
# %%
# ---- Save values to file 3----
pos_x_deb = opti.debug.value(pos_x)
vel_x_deb = opti.debug.value(vel_x)
pos_y_deb = opti.debug.value(pos_y)
vel_y_deb = opti.debug.value(vel_y)
phi_deb = opti.debug.value(phi)
r_deb = opti.debug.value(r)
delta_deb = opti.debug.value(delta)
u_deb = opti.debug.value(U)
T_deb = opti.debug.value(T)

syio.savemat('LinearSTData3/pos_x.mat', {'pos_x_data': pos_x_deb})
syio.savemat('LinearSTData3/pos_y.mat', {'pos_y_data': pos_y_deb})
syio.savemat('LinearSTData3/vel_x.mat', {'vel_x_data': vel_x_deb})
syio.savemat('LinearSTData3/vel_y.mat', {'vel_y_data': vel_y_deb})
syio.savemat('LinearSTData3/phi.mat', {'phi_data': phi_deb})
syio.savemat('LinearSTData3/r.mat', {'r_data': r_deb})
syio.savemat('LinearSTData3/delta.mat', {'delta_data': delta_deb})
syio.savemat('LinearSTData3/delta_d.mat', {'delta_d_data': u_deb[0]})
syio.savemat('LinearSTData3/fxf.mat', {'fxf_data': u_deb[1]})
syio.savemat('LinearSTData3/fxr.mat', {'fxr_data': u_deb[2]})
syio.savemat('LinearSTData3/t.mat', {'t_data': T_deb})
syio.savemat('LinearSTData3/alphaf.mat', {'alphaf_data': alphaf_opt})
syio.savemat('LinearSTData3/alphar.mat', {'alphar_data': alphar_opt})
syio.savemat('LinearSTData3/fyf.mat', {'fyf_data': Fyf_opt})
syio.savemat('LinearSTData3/fyr.mat', {'fyr_data': Fyr_opt})
syio.savemat('LinearSTData3/fzf.mat', {'fzf_data': Fzfs_max})
syio.savemat('LinearSTData3/fzr.mat', {'fzr_data': Fzrs_max})
# %%
# ---- Debugging position plot ----

pos_x_deb = opti.debug.value(pos_x)
vel_x_deb = opti.debug.value(vel_x)
pos_y_deb = opti.debug.value(pos_y)
vel_y_deb = opti.debug.value(vel_y)
phi_deb = opti.debug.value(phi)
r_deb = opti.debug.value(r)
delta_deb = opti.debug.value(delta)
u_deb = opti.debug.value(U)
T_deb = opti.debug.value(T)

tt = np.linspace(0, T_deb, 1)
ttt = np.linspace(0, T_deb, N+1)
fig = plt.figure(figsize=(10,4.0))
ax = fig.add_subplot(111)
ax.plot(pos_x_deb, pos_y_deb, zorder=1)
for p in range(0, len(ttt), 5):
    i = ttt[p]
    yawi = np.interp(i, ttt, phi_deb)
    posxi = np.interp(i, ttt, pos_x_deb)
    posyi = np.interp(i, ttt, pos_y_deb)
    l = Params.lr.value+Params.lf.value
    w = 0.1
    opposite_x = posxi + l*cos(yawi) - w*sin(yawi)
    opposite_y = posyi + l*sin(yawi) + w*cos(yawi)
    center_x = posxi - (opposite_x-posxi)/2
    center_y = posyi - (opposite_y-posyi)/2
    rect = matplotlib.patches.Rectangle((center_x, center_y), l, w, angle = yawi*(180/pi), color = 'red', zorder=2)
    ax.add_patch(rect)
xx = np.linspace(obstacle_center_x-R1,obstacle_center_x+R1,1000)
xx2 = np.linspace(obstacle_center_x-R3,obstacle_center_x+R3,1000)
#elipses
plt.plot(xx, (1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx, -(1-((xx-obstacle_center_x)/R1)**6)**(1/6)*R2+obstacle_center_y, 'k--')
plt.plot(xx2, (1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.plot(xx2, -(1-((xx2-obstacle_center_x)/R3)**6)**(1/6)*R4+obstacle_center_y, 'k--')
plt.xlim((0, 65))
plt.ylim((-13, 13))
plt.grid(True)
plt.title('Path of linear ST Model')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')
plt.show()

# %%
