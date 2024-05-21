# %%
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from casadi import *
from enum import Enum
from vehicleModels import magicST
import scipy.io as syio
from matplotlib.patches import Ellipse


read_from_file = True #Read previous solutions from file to help solver
actual_car_size = True #Use the car length and width otherwise it is seen as a point
friction = "dry" #Friction used

#Elipse constants
R1 = 52 #diff from obstacle 1 center in x
R2 = 1 #height of obstacle 1
R3 = 60 #diff from obstacle 2 center in x
R4 = 4 #height of obstacle 2
obstacle_center_x = 0 #OBSTACLE CENTER X
obstacle_center_y = 0 #OBSTACLE CENTER Y
road_width = R4-R2
middle_of_road = obstacle_center_y+R2+road_width/2
match friction:
    case "dry":
        #Parameters to model for dry ground
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

            #Time constant
            TdFxf = 0.1
            TdFxr = 0.1
    case "wet":
        #Parameters to model for wet ground
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
    case "snow":
        #Parameters to model for snow ground
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
            Bxf = 10.2
            Bxr = 9.71
            Cxf = 1.96
            Cxr = 1.96
            Exf = 0.651
            Exr = 0.624
            muyf = 0.383
            muyr = 0.394
            Byf = 19.1
            Byr = 20
            Cyf = 0.550
            Cyr = 0.550
            Eyf = -2.10
            Eyr = -1.93

            #Time constant
            TdFxf = 0.1
            TdFxr = 0.1
    case "ice":
        #Parameters to model for ice ground
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
            Bxf = 31.1
            Bxr = 29.5
            Cxf = 1.77
            Cxr = 1.77
            Exf = 0.710
            Exr = 0.681
            muyf = 0.162
            muyr = 0.167
            Byf = 28.4
            Byr = 30
            Cyf = 1.48
            Cyr = 1.48
            Eyf = -1.18
            Eyr = -1.08

            #Time constant
            TdFxf = 0.1
            TdFxr = 0.1

upper_corner_of_road = obstacle_center_y+R4-(Params.w.value/2)-0.1
upper_corner_of_road_under = obstacle_center_y-R2-(Params.w.value/2)-0.1

# start values to solver
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
    start_pos_x_init = syio.loadmat('MagicSTData/pos_x.mat')['pos_x_data'][0]
    start_pos_y_init = syio.loadmat('MagicSTData/pos_y.mat')['pos_y_data'][0]
    start_vel_x_init = syio.loadmat('MagicSTData/vel_x.mat')['vel_x_data'][0]
    start_vel_y_init = syio.loadmat('MagicSTData/vel_y.mat')['vel_y_data'][0]
    start_phi_init = syio.loadmat('MagicSTData/phi.mat')['phi_data'][0]
    start_r_init = syio.loadmat('MagicSTData/r.mat')['r_data'][0]
    start_delta_init = syio.loadmat('MagicSTData/delta.mat')['delta_data'][0]
    start_delta_d_init = syio.loadmat('MagicSTData/delta_d.mat')['delta_d_data'][0]
    start_Fxf_init = syio.loadmat('MagicSTData/fxf.mat')['fxf_data'][0]
    start_Fxr_init = syio.loadmat('MagicSTData/fxr.mat')['fxr_data'][0]
    start_T_init = syio.loadmat('MagicSTData/t.mat')['t_data'][0]

#actual start values
start_pos_x = obstacle_center_x+5
start_pos_y = upper_corner_of_road
start_vel_x = 25/3.6
start_vel_y = 0
start_phi = 0
end_phi = -pi
end_r = 0
end_delta = 0
end_delta_d = 0
#end values
end_pos_x = start_pos_x+5
end_pos_y = -middle_of_road

# steering constraints
delta_max = pi/6
delta_d_max = pi/4

N = 100
opti = Opti()

# ---- decision variables ---------
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
Fxf = U[1, :]
Fxr = U[2, :]
T = opti.variable(1) #final time

# ---- objective ---------
opti.minimize(T) #minimize time
#opti.minimize(-vel_x[-1]) #maximize vel_x

# ---- dynamic constraints -----
dt = T/N # length of a control interval
for k in range(N): # loop over control intervals 
   # Runge-Kutta 4 integration                      
   k1 = magicST(X[:,k],         U[:,k], Params)[0]
   k2 = magicST(X[:,k]+dt/2*k1, U[:,k], Params)[0]
   k3 = magicST(X[:,k]+dt/2*k2, U[:,k], Params)[0]
   k4 = magicST(X[:,k]+dt*k3,   U[:,k], Params)[0]
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

    #Use the car size in simulation otherwise car is seen as a point
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

# ---- initial and boundary conditions --------
#Usually a good idea to have all condition for the first iteration of the problem for easier build-up       
if not read_from_file:
    opti.subject_to(pos_x[0] == start_pos_x)      
    opti.subject_to(pos_y[0] == start_pos_y)       

    opti.subject_to(pos_x[-1] == end_pos_x) 
    opti.subject_to(pos_y[-1] == end_pos_y)      

    opti.subject_to(vel_x[0] == start_vel_x)  
    opti.subject_to(vel_y[0] == start_vel_y)           

    opti.subject_to(phi[0] == start_phi)        
    opti.subject_to(phi[-1] == end_phi)           

    opti.subject_to(r[-1] == end_r)        

    opti.subject_to(delta[-1] == end_delta)        

    opti.subject_to(delta_d[-1] == end_delta_d)

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

# ---- misc. constraints  ----------
opti.subject_to(T >= 0) # Time must be positive
opti.subject_to(vel_x > 0) # vel_x not be equal 0

# ---- steering constrains ----
opti.subject_to(opti.bounded(-delta_max,delta,delta_max))    # steering angle limit
opti.subject_to(opti.bounded(-delta_d_max,delta_d,delta_d_max)) # steering rate limit

# % ---- force constrains ----
Fzfs_max = Params.m.value * Params.g.value * Params.lr.value / (Params.lf.value + Params.lr.value)
Fzrs_max = Params.m.value * Params.g.value * Params.lf.value / (Params.lf.value + Params.lr.value)
Dxf = Params.muxf.value * Fzfs_max
Dxr = Params.muxr.value * Fzrs_max  
opti.subject_to(opti.bounded(-Dxf,Fxf,Dxf)) #front wheel drive   
opti.subject_to(opti.bounded(-Dxr,Fxr,0))      

#---- initial values for solver ---
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

# ---- solve NLP              ------
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
Fy0f_opt = []
Fy0r_opt = []
Dyf = Params.muyf.value * Fzfs_max
Dyr = Params.muyr.value * Fzrs_max
for i in range(N+1):
    variable_list = magicST(sol.value(X[:, i]), sol.value(U[:, 0]), Params)[1]
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
    Fy0f_opt.append(variable_list[11])
    Fy0r_opt.append(variable_list[12])
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
plt.ylim((-15, 15))
plt.grid(True)
plt.title('Path of magic ST Model')
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
plt.ylim((-15, 15))
plt.grid(True)
plt.title('Path of magic ST Model')
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
plt.plot(u_opt[2], Fyr_opt[:-1], color = 'red', zorder=1)
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
plt.title('Speed of ST Model')
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
plt.plot(list(np.array(alphaf_opt)*180/pi), list(np.array(Fy0f_opt)/Fzfs_max))
plt.grid(True)
plt.title('Fy0f/Fzf and Fy0r/Fzr')
plt.xlabel('alphaf [deg]')
plt.ylabel('Fyf/Fzf [N]')
plt.subplot(212)
plt.plot(list(np.array(alphar_opt)*180/pi), list(np.array(Fy0r_opt)/Fzrs_max))
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

syio.savemat('MagicSTData/pos_x.mat', {'pos_x_data': pos_x_deb})
syio.savemat('MagicSTData/pos_y.mat', {'pos_y_data': pos_y_deb})
syio.savemat('MagicSTData/vel_x.mat', {'vel_x_data': vel_x_deb})
syio.savemat('MagicSTData/vel_y.mat', {'vel_y_data': vel_y_deb})
syio.savemat('MagicSTData/phi.mat', {'phi_data': phi_deb})
syio.savemat('MagicSTData/r.mat', {'r_data': r_deb})
syio.savemat('MagicSTData/delta.mat', {'delta_data': delta_deb})
syio.savemat('MagicSTData/delta_d.mat', {'delta_d_data': u_deb[0]})
syio.savemat('MagicSTData/fxf.mat', {'fxf_data': u_deb[1]})
syio.savemat('MagicSTData/fxr.mat', {'fxr_data': u_deb[2]})
syio.savemat('MagicSTData/t.mat', {'t_data': T_deb})
syio.savemat('MagicSTData/alphaf.mat', {'alphaf_data': alphaf_opt})
syio.savemat('MagicSTData/alphar.mat', {'alphar_data': alphar_opt})
syio.savemat('MagicSTData/fyf.mat', {'fyf_data': Fyf_opt})
syio.savemat('MagicSTData/fyr.mat', {'fyr_data': Fyr_opt})
syio.savemat('MagicSTData/fzf.mat', {'fzf_data': Fzfs_max})
syio.savemat('MagicSTData/fzr.mat', {'fzr_data': Fzrs_max})
# %%
# ---- Save values to file2 ----
pos_x_deb = opti.debug.value(pos_x)
vel_x_deb = opti.debug.value(vel_x)
pos_y_deb = opti.debug.value(pos_y)
vel_y_deb = opti.debug.value(vel_y)
phi_deb = opti.debug.value(phi)
r_deb = opti.debug.value(r)
delta_deb = opti.debug.value(delta)
u_deb = opti.debug.value(U)
T_deb = opti.debug.value(T)

syio.savemat('MagicSTData2/pos_x.mat', {'pos_x_data': pos_x_deb})
syio.savemat('MagicSTData2/pos_y.mat', {'pos_y_data': pos_y_deb})
syio.savemat('MagicSTData2/vel_x.mat', {'vel_x_data': vel_x_deb})
syio.savemat('MagicSTData2/vel_y.mat', {'vel_y_data': vel_y_deb})
syio.savemat('MagicSTData2/phi.mat', {'phi_data': phi_deb})
syio.savemat('MagicSTData2/r.mat', {'r_data': r_deb})
syio.savemat('MagicSTData2/delta.mat', {'delta_data': delta_deb})
syio.savemat('MagicSTData2/delta_d.mat', {'delta_d_data': u_deb[0]})
syio.savemat('MagicSTData2/fxf.mat', {'fxf_data': u_deb[1]})
syio.savemat('MagicSTData2/fxr.mat', {'fxr_data': u_deb[2]})
syio.savemat('MagicSTData2/t.mat', {'t_data': T_deb})
syio.savemat('MagicSTData2/alphaf.mat', {'alphaf_data': alphaf_opt})
syio.savemat('MagicSTData2/alphar.mat', {'alphar_data': alphar_opt})
syio.savemat('MagicSTData2/fyf.mat', {'fyf_data': Fyf_opt})
syio.savemat('MagicSTData2/fyr.mat', {'fyr_data': Fyr_opt})
syio.savemat('MagicSTData2/fzf.mat', {'fzf_data': Fzfs_max})
syio.savemat('MagicSTData2/fzr.mat', {'fzr_data': Fzrs_max})
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

syio.savemat('MagicSTData3/pos_x.mat', {'pos_x_data': pos_x_deb})
syio.savemat('MagicSTData3/pos_y.mat', {'pos_y_data': pos_y_deb})
syio.savemat('MagicSTData3/vel_x.mat', {'vel_x_data': vel_x_deb})
syio.savemat('MagicSTData3/vel_y.mat', {'vel_y_data': vel_y_deb})
syio.savemat('MagicSTData3/phi.mat', {'phi_data': phi_deb})
syio.savemat('MagicSTData3/r.mat', {'r_data': r_deb})
syio.savemat('MagicSTData3/delta.mat', {'delta_data': delta_deb})
syio.savemat('MagicSTData3/delta_d.mat', {'delta_d_data': u_deb[0]})
syio.savemat('MagicSTData3/fxf.mat', {'fxf_data': u_deb[1]})
syio.savemat('MagicSTData3/fxr.mat', {'fxr_data': u_deb[2]})
syio.savemat('MagicSTData3/t.mat', {'t_data': T_deb})
syio.savemat('MagicSTData3/alphaf.mat', {'alphaf_data': alphaf_opt})
syio.savemat('MagicSTData3/alphar.mat', {'alphar_data': alphar_opt})
syio.savemat('MagicSTData3/fyf.mat', {'fyf_data': Fyf_opt})
syio.savemat('MagicSTData3/fyr.mat', {'fyr_data': Fyr_opt})
syio.savemat('MagicSTData3/fzf.mat', {'fzf_data': Fzfs_max})
syio.savemat('MagicSTData3/fzr.mat', {'fzr_data': Fzrs_max})
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
plt.ylim((-20, 20))
plt.grid(True)
plt.title('Path of ST Model')
plt.xlabel('x-coordinate [m]')
plt.ylabel('y-coordinate [m]')
plt.show()

# %%
