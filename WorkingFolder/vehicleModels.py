from casadi import *
from numpy import *
    
# functions and definitions                               
def linearST(x ,u, params):
    # ---- inputs ----
    vel_x = x[2]
    vel_y = x[3]
    phi = x[4]
    r = x[5]
    delta = x[6]
    # control 
    delta_d = u[0]
    Fxf = u[1]
    Fxr = u[2]

    # ---- independent wheel velocities ----
    vxf = vel_x*cos(delta) + sin(delta)*(vel_y + params.lf.value*r)
    vyf = cos(delta)*(vel_y + params.lf.value*r) - vel_x*sin(delta)
    vxr = vel_x
    vyr = vel_y + params.lr.value*r
    # ---- determining the lateral slips ----
    alphaf = -atan(vyf/vxf)
    alphar = -atan(vyr/vxr)
    # ---- determining the lateral forces on the wheels from Fy ----
    Fyf = 120000*alphaf
    Fyr = 103600*alphar
    # ---- the total forces and moments acting on the vehcile ----
    F_X = cos(delta)*Fxf - sin(delta)*Fyf + Fxr
    F_Y = cos(delta)*Fyf + sin(delta)*Fxf + Fyr
    M_Z = params.lf.value*cos(delta)*Fyf + params.lf.value*sin(delta)*Fxf - params.lr.value*Fyr
    # ---- Variables for plots ----
    variables_list = [F_X, F_Y, M_Z, Fyf, Fyr, alphaf, alphar, vxf, vxr, vyf, vyr]

    # X_d:
    return [vertcat(vel_x*cos(phi) - x[3]*sin(phi), 
            vel_x*sin(phi) + x[3]*cos(phi),
            F_X/params.m.value + x[3]*x[5],
            F_Y/params.m.value - x[2]*x[5],
            r,
            M_Z/params.Izz.value,
            delta_d), variables_list]

def magicST(x ,u, params):
    # ---- inputs ----
    vel_x = x[2]
    vel_y = x[3]
    phi = x[4]
    r = x[5]
    delta = x[6]
    # control 
    delta_d = u[0]
    Fxf = u[1]
    Fxr = u[2]

    # ---- independent wheel velocities ----
    vxf = vel_x*cos(delta) + sin(delta)*(vel_y + params.lf.value*r)
    vyf = cos(delta)*(vel_y + params.lf.value*r) - vel_x*sin(delta)
    vxr = vel_x
    vyr = vel_y + params.lr.value*r
    # ---- determining the lateral slips ----
    alphaf = -atan(vyf/vxf)
    alphar = -atan(vyr/vxr)
    # ---- determining the lateral forces on the wheels from Fy ----
    # nominalforces
    Fzf = params.m.value*params.g.value*params.lr.value/(params.lf.value+params.lr.value)
    Fzr = params.m.value*params.g.value*params.lf.value/(params.lf.value+params.lr.value)
    Dxf = params.muxf.value*Fzf 
    Dxr = params.muxr.value*Fzr
    Dyf = params.muyf.value*Fzf 
    Dyr = params.muyr.value*Fzr
    # pure slip
    Fy0f = Dyf*sin(params.Cyf.value*atan(params.Byf.value*alphaf-params.Eyf.value*(params.Byf.value*alphaf - atan(params.Byf.value*alphaf))))
    Fy0r = Dyr*sin(params.Cyr.value*atan(params.Byr.value*alphar-params.Eyr.value*(params.Byr.value*alphar - atan(params.Byr.value*alphar))))
    # combined slip - friction ellipse
    Fyf = Fy0f*sqrt(1.01 - (Fxf/Dxf)**2)
    Fyr = Fy0r*sqrt(1.01 - (Fxr/Dxr)**2)
    # ---- the total forces and moments acting on the vehicle ----
    F_X = cos(delta)*Fxf - sin(delta)*Fyf + Fxr
    F_Y = cos(delta)*Fyf + sin(delta)*Fxf + Fyr
    M_Z = params.lf.value*cos(delta)*Fyf + params.lf.value*sin(delta)*Fxf - params.lr.value*Fyr
        
    # ---- Variables for plots ----
    variables_list = [F_X, F_Y, M_Z, Fyf, Fyr, alphaf, alphar, vxf, vxr, vyf, vyr, Fy0f, Fy0r]

    # X_d:
    return [vertcat(vel_x*cos(phi) - x[3]*sin(phi), 
            vel_x*sin(phi) + x[3]*cos(phi),
            F_X/params.m.value + x[3]*x[5],
            F_Y/params.m.value - x[2]*x[5],
            r,
            M_Z/params.Izz.value,
            delta_d), variables_list]



def magicSTTimeConstant(x ,u, params):
    # ---- inputs ----
    vel_x = x[2]
    vel_y = x[3]
    phi = x[4]
    r = x[5]
    delta = x[6]
    Fxf = x[7]
    Fxr = x[8]
    # control 
    delta_d = u[0]
    Fxfval = u[1]
    Fxrval = u[2]

    # ---- independent wheel velocities ----
    vxf = vel_x*cos(delta) + sin(delta)*(vel_y + params.lf.value*r)
    vyf = cos(delta)*(vel_y + params.lf.value*r) - vel_x*sin(delta)
    vxr = vel_x
    vyr = vel_y + params.lr.value*r
    # ---- determining the lateral slips ----
    alphaf = -atan(vyf/vxf)
    alphar = -atan(vyr/vxr)
    # ---- determining the lateral forces on the wheels from Fy ----
    # nominal forces
    Fzf = params.m.value*params.g.value*params.lr.value/(params.lf.value+params.lr.value)
    Fzr = params.m.value*params.g.value*params.lf.value/(params.lf.value+params.lr.value)
    Dxf = params.muxf.value*Fzf 
    Dxr = params.muxr.value*Fzr
    Dyf = params.muyf.value*Fzf 
    Dyr = params.muyr.value*Fzr
    # pure slip
    Fy0f = Dyf*sin(params.Cyf.value*atan(params.Byf.value*alphaf-params.Eyf.value*(params.Byf.value*alphaf - atan(params.Byf.value*alphaf))))
    Fy0r = Dyr*sin(params.Cyr.value*atan(params.Byr.value*alphar-params.Eyr.value*(params.Byr.value*alphar - atan(params.Byr.value*alphar))))
    # combined slip - friction ellipse
    Fyf = Fy0f*sqrt(1.01 - (Fxf/Dxf)**2)
    Fyr = Fy0r*sqrt(1.01 - (Fxr/Dxr)**2)
    # ---- the total forces and moments acting on the vehcile ----
    F_X = cos(delta)*Fxf - sin(delta)*Fyf + Fxr
    F_Y = cos(delta)*Fyf + sin(delta)*Fxf + Fyr
    M_Z = params.lf.value*cos(delta)*Fyf + params.lf.value*sin(delta)*Fxf - params.lr.value*Fyr
        
    # ---- Variables for plots ----
    variables_list = [F_X, F_Y, M_Z, Fyf, Fyr, alphaf, alphar, vxf, vxr, vyf, vyr, Fy0f, Fy0r]

    # X_d:
    return [vertcat(vel_x*cos(phi) - x[3]*sin(phi), 
            vel_x*sin(phi) + x[3]*cos(phi),
            F_X/params.m.value + x[3]*x[5],
            F_Y/params.m.value - x[2]*x[5],
            r,
            M_Z/params.Izz.value,
            delta_d,
            (Fxfval-Fxf)/params.TdFxf.value,
            (Fxrval-Fxr)/params.TdFxr.value), variables_list]

def magicSTLoad(x ,u, params):
    # ---- inputs ----
    vel_x = x[2]
    vel_y = x[3]
    phi = x[4]
    r = x[5]
    delta = x[6]
    Fxf = x[7]
    Fxr = x[8]
    # control 
    delta_d = u[0]
    Fxfval = u[1]
    Fxrval = u[2]

    # ---- independent wheel velocities ----
    vxf = vel_x*cos(delta) + sin(delta)*(vel_y + params.lf.value*r)
    vyf = cos(delta)*(vel_y + params.lf.value*r) - vel_x*sin(delta)
    vxr = vel_x
    vyr = vel_y + params.lr.value*r
    # ---- determining the lateral slips ----
    alphaf = -atan(vyf/vxf)
    alphar = -atan(vyr/vxr)
    # ---- determining the lateral forces on the wheels from Fy ----
    # nominal forces
    Fzf = (params.m.value*params.g.value*params.lr.value + params.h.value*F_X)/(params.lf.value+params.lr.value)
    Fzr = (params.m.value*params.g.value*params.lf.value - params.h.value*F_X)/(params.lf.value+params.lr.value)
    Dxf = params.muxf.value*Fzf 
    Dxr = params.muxr.value*Fzr
    Dyf = params.muyf.value*Fzf 
    Dyr = params.muyr.value*Fzr
    # pure slip
    Fy0f = Dyf*sin(params.Cyf.value*atan(params.Byf.value*alphaf-params.Eyf.value*(params.Byf.value*alphaf - atan(params.Byf.value*alphaf))))
    Fy0r = Dyr*sin(params.Cyr.value*atan(params.Byr.value*alphar-params.Eyr.value*(params.Byr.value*alphar - atan(params.Byr.value*alphar))))
    # combined slip - friction ellipse
    Fyf = Fy0f*sqrt(1.01 - (Fxf/Dxf)**2)
    Fyr = Fy0r*sqrt(1.01 - (Fxr/Dxr)**2)
    # ---- the total forces and moments acting on the vehcile ----
    F_X = cos(delta)*Fxf - sin(delta)*Fyf + Fxr
    F_Y = cos(delta)*Fyf + sin(delta)*Fxf + Fyr
    M_Z = params.lf.value*cos(delta)*Fyf + params.lf.value*sin(delta)*Fxf - params.lr.value*Fyr
        
    # ---- Variables for plots ----
    variables_list = [F_X, F_Y, M_Z, Fyf, Fyr, alphaf, alphar, vxf, vxr, vyf, vyr, Fy0f, Fy0r]

    # X_d:
    return [vertcat(vel_x*cos(phi) - x[3]*sin(phi), 
            vel_x*sin(phi) + x[3]*cos(phi),
            F_X/params.m.value + x[3]*x[5],
            F_Y/params.m.value - x[2]*x[5],
            r,
            M_Z/params.Izz.value,
            delta_d,
            (Fxfval-Fxf)/params.TdFxf.value,
            (Fxrval-Fxr)/params.TdFxr.value), variables_list]