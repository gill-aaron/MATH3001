# -*- coding: utf-8 -*-

# %matplotlib qt5


# PLANETARY ORBITS

import numpy as np
from math import pi, sin, cos, sqrt, floor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from time import sleep

np.set_printoptions(precision=16)

# r = r_norm * r-unit
# h = Timestep Size
# G = Gravitational Constant
# M = Star Mass
# P = Period of Orbit

#--------------------------------------------------------------------------------------------------------------

# CONSTANTS
# (modifying these is the easiest way to test different orbits)
# Note that e should always be a float

e = 0 #0.15
a = 1
b = a * sqrt(1 - e**2)
GM = 1
P = 2 * pi * sqrt(a**3 / GM)
h_initial = P / 15  #P / 200 #15

orbits = 34

# Affects leapfrog only
default_order = 2

# Sample rate for some plots
rate = 25.11 #1.11

# This makes the plot text readable when added to LaTeX
plt.rcParams.update({"font.size": 20})
plt.rcParams.update({"lines.linewidth": 2})

colours = ["r", "m", "c", "g", "y", "k", "b", "m", "y"]
colours_alt = ["orange", "lime", "brown", "pink", "blue", "orange", "lime", "brown", "pink", "blue"]

disable_analytic = False
colswap = 0

# INITIAL CONDITIONS AND PLANETARY SYSTEMS
# r - initial positions
# v - initial velocities
# GM - proportional mass of bodies
# inner - mass of central body / star (0 if none)


class System:
    def __init__(self, r, v, inner=0, GM=[0]):
        self.r = r
        self.v = v
        self.GM = GM * np.ones(len(r))
        self.inner = inner


r_initial = np.array([a * (1 + e), 0])
v_initial = np.array([0, sqrt((GM * (1 - e)) / (a * 1 + e))])
energy_initial = (v_initial[0]**2 + v_initial[1]**2) - GM / np.linalg.norm(r_initial)
angular_initial = r_initial[0] * v_initial[1] - r_initial[1] * v_initial[0]

r_initial_sys = np.array([[a * (1 + e), 0]])
v_initial_sys = np.array([[0, sqrt((GM * (1 - e)) / (a * (1 + e)))]])

r_initial_sys2 = np.vstack(([a * (1 + e), 0], [2 * a * ((1 + e)), 0]))
v_initial_sys2 = np.vstack(([0, sqrt((GM * (1 - e)) / (a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (2 * a * (1 + e)))]))

r_initial_solution = np.vstack(([a * (1 + e), 0], [-a * ((1 + e)), 0]))
v_initial_solution = np.vstack(([0, sqrt((GM * 1.00025 * (1 - e)) / (a * (1 + e)))], [0, -sqrt((GM * 1.00025 * (1 - e)) / (a * (1 + e)))]))

r_initial_solution2 = np.vstack(([a * (1 + e), 0], [-a * ((1 + e)), 0], [0, a * (1 + e)], [0, -a * ((1 + e))]))
v_initial_solution2 = np.vstack(([0, sqrt((GM * (1 + e)) / (a * (1 + e)))], [0, -sqrt((GM * (1 + e)) / (a * (1 + e)))], [-sqrt((GM * (1 + e)) / (a * (1 + e))), 0], [sqrt((GM * (1 + e)) / (a * (1 + e))), 0]))

r_initial_nbody = np.vstack(([3,4], [-4,3], [0,-1], [2,2]))
v_initial_nbody = np.vstack(([0,0], [0,0], [0,0], [0,0]))

r_initial_nbodyt = np.vstack(([-1, 2], [-2, 1], [0, -1], [1, 1], [0, 3], [1, 2], [0, 4], [5, -1]))
v_initial_nbodyt = np.vstack(([0.8, -0], [0.5, 0], [1, 0.5], [-0.6 , 0.2], [0.8, -0.1], [0.1, 0.1], [0.1, 0], [-0.2 , 0.1]))

r_initial_f8 = np.vstack(([0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0,0]))
v_initial_f8 = np.vstack(([0.93240737 / 2, 0.86473146 / 2], [0.93240737 / 2, 0.86473146 / 2], [-0.93240737, -0.86473146]))

r_jupiter_initial = np.vstack(([a * (1 + e), 0], [2.007294512920982 * a * (1 + e), 0], [4.044089833579969 * a * (1 + e), 0], [9.433417291358841 * a * (1 + e), 0]))
v_jupiter_initial = np.vstack(([0, sqrt((GM * (1 - e)) / (a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (2.007294512920982 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (4.044089833579969 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (9.433417291358841 * a * (1 + e)))]))

r_jupiter_initial2 = np.vstack(([a * (1 + e), 0], [2.007294512920982 * a * (1 + e), 0], [4.044089833579969 * a * (1 + e), 0], [9.433417291358841 * a * (1 + e), 0]))
v_jupiter_initial2 = np.vstack(([0, sqrt((GM * (1 - e)) / (a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (2.007294512920982 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (4.044089833579969 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (9.433417291358841 * a * (1 + e)))]))

r_jupiter_initial3 = np.vstack(([a * (1 + e), 0], [2 * a * (1 + e), 0], [4 * a * (1 + e), 0], [8 * a * (1 + e), 0]))
v_jupiter_initial3 = np.vstack(([0, sqrt((GM * (1 - e)) / (a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (2 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (4 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (8 * a * (1 + e)))]))

r_jupiter_initial4 = np.vstack(([a * (1 + e), 0], [2 * a * (1 + e), 0], [3 * a * (1 + e), 0], [4 * a * (1 + e), 0]))
v_jupiter_initial4 = np.vstack(([0, sqrt((GM * (1 - e)) / (a * (1 + e)))], [0, -sqrt((GM * (1 - e)) / (2 * a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (3 * a * (1 + e)))], [0, -sqrt((GM * (1 - e)) / (4 * a * (1 + e)))]))


r_saturn_initial = np.vstack(([a * (1 + e), 0], [1.025 * a * (1 + e), 0], [1.05 * a * (1 + e), 0], [1.075 * a * (1 + e), 0], [1.1 * a * (1 + e), 0], [1.125 * a * (1 + e), 0], 
                              [0.975 * a * (1 + e), 0], [0.95 * a * (1 + e), 0], [0.925 * a * (1 + e), 0], [0.9 * a * (1 + e), 0], [0.875 * a * (1 + e), 0]))

v_saturn_initial = np.vstack(([0, sqrt((GM * (1 - e)) / (a * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 1.025 * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 1.05 * (1 + e)))],
                              [0, sqrt((GM * (1 - e)) / (a * 1.075 * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 1.1 * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 1.125 * (1 + e)))],
                              [0, sqrt((GM * (1 - e)) / (a * 0.975 * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 0.95 * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 0.925 * (1 + e)))],
                              [0, sqrt((GM * (1 - e)) / (a * 0.9 * (1 + e)))], [0, sqrt((GM * (1 - e)) / (a * 0.875 * (1 + e)))]))


p1 = System(r_initial_sys, v_initial_sys)
p1b = System(r_initial_sys, v_initial_sys, inner=1, GM = np.array([1]))
p2b = System(r_initial_sys2, v_initial_sys2 * sqrt(1), inner=1, GM = np.array([0.001, 0.001]))

p1s = System(r_initial_solution, v_initial_solution, inner=1, GM = np.array([0.0001, 0.0001]))
p2s = System(r_initial_solution2, v_initial_solution2, inner=1, GM = np.array([0.001, 0.01, 0.001, 0.001]))

ptest = System(r_initial_sys2, v_initial_sys2, inner=1, GM = np.array([0.001, 0.001]))
ptest2 = System(r_initial_nbodyt, v_initial_nbodyt, inner=1, GM = np.array([0.1, 0.12, 0.015, 0.1, 0.10, 0.2, 0.01, 0.2]))

nbody1 = System(r_initial_nbody, v_initial_nbody, 20)
figure_eight = System(r_initial_f8, v_initial_f8 * sqrt(2), GM = np.array([2,2,2]))
nbody2 = System(r_initial_nbody, v_initial_nbody, GM = np.array([1,2,4,8]))

jupiter = System(r_jupiter_initial, v_jupiter_initial, inner = 1, GM = [0.000047056840153, 0.000026915256522, 0.000083095247167, 0.000060329426025])
jupiter2 = System(r_jupiter_initial2, v_jupiter_initial2, inner = 1, GM = [0.0047056840153, 0.0026915256522, 0.0083095247167, 0.0060329426025]) 
jupiter3 = System(r_jupiter_initial3, v_jupiter_initial3, inner = 1, GM = [0.0047056840153, 0.0026915256522, 0.0083095247167, 0.0060329426025]) 
jupiter4 = System(r_jupiter_initial4, v_jupiter_initial4, inner = 1, GM = [0.047056840153, 0.026915256522, 0.083095247167, 0.060329426025]) 

saturn = System(r_saturn_initial, v_saturn_initial, inner = 1, GM = [0.00001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#--------------------------------------------------------------------------------------------------------------

# GENERAL PURPOSE FUNCTIONS

def finishingTouches (final, plot_type, initial=r_initial, one_body = True, central = True):
    # The yellow marker represents the position of star
    # The green marker represents the initial position of the planet
    # The cyan marker represents the final position of the planet
    
    if plot_type <= 1 and central:
        if plot_type == 1 and not disable_analytic:
            analyticalPrediction(h_initial)
            
    if one_body:
        if colswap == 0:
            plt.plot(0, 0, "y*", initial[0][0], initial[0][1], "co", final[0][0], final[0][1], "mo", markersize=15)
        elif colswap == 1:
            plt.plot(final[0][0], final[0][1], "ks", markersize=5)
            plt.plot(0, 0, "y*", final[0][0], final[0][1], "mo", markersize=15)
        else:
            plt.plot(final[0][0], final[0][1], "ks", markersize=5)
            plt.plot(0, 0, "y*", initial[0][0], initial[0][1], "co", markersize=15)
            
    
    elif central:
        #print("test")
        plt.plot(0, 0, "y*")
        #for k in range(len(final)):
            #plt.plot(final[k][0], final[k][1], "co")
        
    print(f"Absolute Error: {np.linalg.norm(final - initial)}")

   
def analyticalPrediction(h):
    E = 0
    b = a * sqrt(1 - e**2)
    increment = 2 * pi * h / P 
    r = np.array([a * (1 + e), 0])

    while E < 2 * pi:                                 
        E += increment 
        r_next = np.array([a * (cos(E) + e), b * sin(E)]) 
        plt.plot((r[0], r_next[0]), (r[1], r_next[1]), "k:")
        
        r = r_next


def timeReverse (system, h, orbits, method, reversals=1):
    to_reverse = numerical_method[method]
    w = np.array((to_reverse(system, h, orbits, 0)))
    #first_orbit = np.linalg.norm(w[0] - system.r)
    w_sys = System(w[0], -w[1], system.inner, system.GM)
    plt.clf()
    for i in range(reversals):
        w = to_reverse(w_sys, h, orbits, 1)
        w_sys.r = w[0]
        w_sys.v = -w[1]
        
    print(f"Time reversal error: {np.linalg.norm(w[0] - system.r)}")


#def saturnRing (system, h, orbits, method, spacing=0.99):
    #r_sat = system.r * spacing
    #v_sat = system.v * spacing
    #r_sat[1] = system.r[1] * 1.5
    #v_sat[1] = system.v[1] / sqrt(1.5)
    #newsys = System(r_sat, v_sat, system.inner, system.GM)
    #new_vals = numerical_method[method](newsys, h, 0.1, 1)[0]
    
    
    #to_moon = np.linalg.norm(new_vals[1]) - np.linalg.norm(r_sat[1])
    #loops = 1
    
    #print(new_vals[1])
    #print(r_sat[1])
    #print(to_moon)
    #print("")

    #while to_moon < -0.001: #or to_moon > 10:
        
        
        
        #plt.scatter(np.linalg.norm(r_sat[1] - r_sat[0]), to_moon)
        
        #r_sat = system.r * 1
        #v_sat = system.v * 1
        #r_sat[1] = system.r[1] * spacing**loops
        #v_sat[1] = system.v[1] / sqrt(spacing**loops)
        
        
        
        #newsys = System(r_sat, v_sat, system.inner, system.GM)
        #new_vals = numerical_method[method](newsys, h, 0.1, 1)[0]

        #to_moon = np.linalg.norm(new_vals[1]) - np.linalg.norm(r_sat[1])
        #loops += 1
        
        #print(new_vals[1])
        #print(r_sat[1])
        #print(to_moon)
        #print("")
        

#--------------------------------------------------------------------------------------------------------------

# FUNCTIONS FOR BUILDING METHODS

def F(r, _=0, inner=GM):
    r_norm = np.linalg.norm(r)
    return -inner * r / (r_norm ** 3)

# Testing purposes only
#def nbodyF(r, _=0, inner=0):
    #r_size = r.size // 2
    #f_values = np.zeros((r_size, 2))

    #for i in range(r_size):
        #for j in range(i+1, r_size):
            #r_dif = r[j] - r[i]
            #f_values[i] += GM * r_dif / np.linalg.norm(r_dif)**3
            #f_values[j] += GM * -r_dif / np.linalg.norm(r_dif)**3

    #f_values += -inner * r / np.linalg.norm(r)**3
            
    #return f_values


def nbodyF(r, GM, inner=0):
    r_size = r.size // 2
    f_values = np.zeros((r_size, 2))

    for i in range(r_size):
        for j in range(i+1, r_size):
            r_dif = r[j] - r[i]      
            f_values[i] += GM[j] * r_dif / np.linalg.norm(r_dif)**3
            f_values[j] += -GM[i] * r_dif / np.linalg.norm(r_dif)**3
        f_values[i] += -inner * r[i] / np.linalg.norm(r[i])**3

    return f_values
 
    
f_function = [F, nbodyF, nbodyF]


def solveKepler (r_n, v_n, inner, h_kepler, Y, r_0, v_0):
    # This is the tangent map algorithm
    
    r_ret = r_n * 1
    v_ret = v_n * 1
    r_ini = r_0 * 1
    v_ini = v_0 * 1

    for i in range(len(r_n)):  
        m_0 = np.linalg.norm(r_ini[i])
        #if not m_0 > 0.01:
            #print("collision")
        
        beta = 2 * inner / m_0 - np.dot(v_ini[i], v_ini[i]) 
        n_0 = np.dot(r_ini[i], v_ini[i])
        s_0 = inner - beta * m_0
        
        X = Y[i]
        prevX = X + 0.1
        prevX2 = X + 0.1
        loops = 0
        
        # Newton-Raphson algorithm based on WHFAST implementation
        # Improvements could be made here for edge case scenarios but not needed within scope of this project
        while (X != prevX) and (X != prevX2) and (loops < 20):
            prevX2 = prevX
            prevX = X
            z = abs(beta * X**2)
            G0 = cos(sqrt(z))
            G1 = X * sin(sqrt(z)) / sqrt(z)
            G2 = (1 - G0) / beta
            G3 = (X - G1) / beta
            loops += 1
            X = (X * (n_0 * G1 + s_0 * G2) - n_0 * G2 - s_0 * G3 + h_kepler) / (m_0 + n_0 * G1 + s_0 * G2)
    
        z = abs(beta * X**2)
        G0 = cos(sqrt(z))
        G1 = X * sin(sqrt(z)) / sqrt(z)
        G2 = (1 - G0) / beta
        G3 = (X - G1) / beta
        G4 = (X**2 / 2 - G2) / beta
        G5 = (X**3 / 6 - G3) / beta
    
        r_calc = m_0 + n_0 * G1 + s_0 * G2
    
        f_gauss = 1 - inner * G2 / m_0
        f_dot = -inner * G1 / (m_0 * r_calc)        
        g_gauss = h_kepler - inner * G3
        g_dot = 1 - inner * G2 / r_calc
    
        r_round1 = f_gauss * r_ini + g_gauss * v_ini
        v_round1 = f_dot * r_ini + g_dot * v_ini

    
        #-----------------------------------

        r_change = r_n - r_ini
        v_change = v_n - v_ini
    
        m_d = np.dot(r_ini[i], (r_change / m_0)[i])
        beta_d = -(2 * inner * m_d) / m_0**2 - 2 * np.dot(v_ini[i], v_change[i]) 
        n_d = np.dot(r_change[i], v_ini[i]) + np.dot(r_ini[i], v_change[i])
        s_d = -beta * m_d - beta_d * m_0  
   
        G1B = (G3 - X * G2) / 2
        G2B = (2 * G4 - X * G3) / 2
        G3B = (3 * G5 - X * G4) / 2
    
        tb = n_0 * G2B + s_0 * G3B
        X_new = -1 / r_calc * (X * m_d + G2 * n_d + G3 * s_d + tb * beta_d)

        G1d = G0 * X_new + G1B * beta_d
        G2d = G1 * X_new + G2B * beta_d
        G3d = G2 * X_new + G3B * beta_d
    
        r_calc_d = m_d + n_d * G1 + s_d * G2 + n_0 * G1d + s_0 * G2d


        f_d = inner * G2 * m_d / m_0**2 - inner * G2d / m_0
        f_ddot = inner * G1d / (m_0 * r_calc) + inner * G1 * (m_d / m_0 + r_calc_d / r_calc) / (m_0 * r_calc)

        g_d = -inner * G3d
        g_ddot = inner * G2d / r_calc + inner * G2 * r_calc_d / r_calc**2
    
        #-----------------------------------
    
        r_delta = f_gauss * r_change + g_gauss * v_change + f_d * r_ini + g_d * v_ini
        v_delta = f_dot * r_change + g_dot * v_change + f_ddot * r_ini + g_ddot * v_ini
    
        #print(r_round1[i])
        #print(r_delta[i])
        r_ret[i] = r_round1[i] + r_delta[i]
        #print(r_ret[i])
        #print("")
        v_ret[i] = v_round1[i] + v_delta[i]
        Y[i] = X + X_new
        
    return(r_ret, v_ret, Y)

    
def getTimestep (order, steps):
    if order == 2:
        c = np.array([0.5, 0.5])
        d = np.array([1])
    
    elif order == 4:    
        c = np.array([1 / (2*(2 - 2**(1/3))), (1 - 2**(1/3)) / (2*(2 - 2**(1/3))),
                     (1 - 2**(1/3)) / (2*(2 - 2**(1/3))), 1 / (2*(2 - 2**(1/3)))])
        
        d = np.array([1 / (2 - 2**(1/3)), -2**(1/3) / (2 - 2**(1/3)),
                      1 / (2 - 2**(1/3))])
        
    elif order == 6:    
        c =  np.array([0.392256805238780, 0.510043411918458, -0.471053385409758,
                   0.0687531682525198, 0.0687531682525198, -0.471053385409758,
                   0.510043411918458, 0.392256805238780])
                   
        d =  np.array([0.784513610477560, 0.235573213359357, -1.17767998417887, 
                   1.31518632068391, -1.17767998417887, 0.235573213359357,
                   0.784513610477560])     
        
    else:          
            
        if order == 8.1:
            w = np.array([0, -1.61582374150097, -2.44699182370524, -0.00716989419708120,
                          2.44002732616735, 0.157739928123617, 1.82020630970714, 1.04242620869991, 0])
 
            w[0] = 1 - 2 * np.sum(w)

        if order == 8.2:
            w = np.array([0, -0.00169248587770116, 2.89195744315849, 0.00378039588360192,
                          -2.89688250328827, 2.89105148970595, -2.33864815101035, 1.48819229202922, 0])
 
            w[0] = 1 - 2 * np.sum(w)
        
        if order == 8.3 or order == 8:
            w = np.array([0, 0.311790812418427, -1.55946803821447, -1.67896928259640,
                          1.66335809963315, -1.06458714789183, 1.36934946416871, 0.629030650210433, 0])
 
            w[0] = 1 - 2 * np.sum(w)
        
        if order == 8.4:
            w = np.array([0, 0.102799849391985, -1.96061023297549, 1.93813913762276,
                          -0.158240635368243, -1.44485223686048, 0.253693336566229, 0.914844246229740, 0])
 
            w[0] = 1 - 2 * np.sum(w)
        
        if order == 8.5:
            w = np.array([0, 0.0227738840094906, 2.52778927322839, -0.0719180053552772,
                          0.00536018921307285, -2.04809795887393, 0.107990467703699, 1.30300165760014, 0])
 
            w[0] = 1 - 2 * np.sum(w)
      
        c = np.zeros(steps)
        d = np.zeros(steps-1)
        
        for i in range(floor(steps / 2)):
            c[i] = (w[-i-2] + w[-i-1]) / 2
            c[-i-1] = (w[-i-2] + w[-i-1]) / 2
            d[i] = w[-i-2]
            d[-i-1] = w[-i-2]
    
    return(c, d)


def getSABA (order):
    if order == 1:
            c = np.array([0.5, 0.5])
            d = np.array([1])
   
    elif order == 2:
        a1 = 1/2 - sqrt(3)/6
        a2 = sqrt(3)/3
        b1 = 1/2

        c = np.array([a1, a2, a1])
        d = np.array([b1, b1])

    elif order == 4:
        a1 = 1/2 - sqrt(525 + 70 * sqrt(30))/70
        a2 = (sqrt(525 + 70 * sqrt(30)) - sqrt(525 - 70 * sqrt(30))) / 70
        a3 = sqrt(525 - 70 * sqrt(30)) / 35
        b1 = 1/4 - sqrt(30) / 72
        b2 = 1/4 + sqrt(30) / 72
        
        c = np.array([a1, a2, a3, a2, a1])
        d = np.array([b1, b2, b2, b1])
        
    elif order == 10:
        a1 = 0.013046735741414139961017993957773973
        a2 = 0.054421580914093604672933661830479502
        a3 = 0.092826899194980052248884661654309736
        a4 = 0.123007087084888607717530710974544707
        a5 = 0.142260527573807989957219971018032089
        a6 = 0.148874338981631210884826001129719985  
        b1 = 0.033335672154344068796784404946665896
        b2 = 0.074725674575290296572888169828848666
        b3 = 0.109543181257991021997767467114081596
        b4 = 0.134633359654998177545613460784734677
        b5 = 0.147762112357376435086946497325669165
        
        c = np.array([a1, a2, a3, a4, a5, a6, a5, a4, a3, a2, a1])
        d = np.array([b1, b2, b3, b4, b5, b5, b4, b3, b2, b1])   

    elif order == 10.4: #SABA (10,4)
 
        a1 = 0.04706710064597250612947887637243678556564
        a2 = 0.1847569354170881069247376193702560968574
        a3 = 0.2827060056798362053243616565541452479160
        a4 = -0.01453004174289681837857815229683813033908
        b1 = 0.1188819173681970199453503950853885936957
        b2 = 0.2410504605515015657441667865901651105675
        b3 = -0.2732866667053238060543113981664559460630
        b4 = 0.8267085775712504407295884329818044835997
        
        c = np.array([a1, a2, a3, a4, a4, a3, a2, a1])
        d = np.array([b1, b2, b3, b4, b3, b2, b1])   

    elif order == 8.64: #SABA (8, 6, 4)
 
        a1 = 0.0711334264982231177779387300061549964174
        a2 = 0.241153427956640098736487795326289649618
        a3 = 0.521411761772814789212136078067994229991
        a4 = -0.333698616227678005726562603400438876027
        b1 = 0.183083687472197221961703757166430291072
        b2 = 0.310782859898574869507522291054262796375
        b3 = -0.0265646185119588006972121379164987592663
        b4 = 0.0653961422823734184559721793911134363710
        
        c = np.array([a1, a2, a3, a4, a4, a3, a2, a1])
        d = np.array([b1, b2, b3, b4, b3, b2, b1])  
        
    elif order == 10.64: #SABA (10, 6, 4)
 
        a1 = 0.03809449742241219545697532230863756534060
        a2 = 0.1452987161169137492940200726606637497442
        a3 = 0.2076276957255412507162056113249882065158
        a4 = 0.4359097036515261592231548624010651844006
        a5 = -0.6538612258327867093807117373907094120024
        b1 = 0.09585888083707521061077150377145884776921
        b2 = 0.2044461531429987806805077839164344779763
        b3 = 0.2170703479789911017143385924306336714532
        b4 = -0.01737538195906509300561788011852699719871
        
        c = np.array([a1, a2, a3, a4, a5, a4, a3, a2, a1])
        d = np.array([b1, b2, b3, b4, b4, b3, b2, b1])

    elif order == 10.642: #SABA (10, 6, 4) alt version
 
        a1 = 0.04731908697653382270404371796320813250988
        a2 = 0.2651105235748785159539480036185693201078
        a3 = -0.009976522883811240843267468164812380613143
        a4 = -0.05992919973494155126395247987729676004016
        a5 = 0.2574761120673404534492282264603316880356
        b1 = 0.1196884624585322035312864297489892143852
        b2 = 0.3752955855379374250420128537687503199451
        b3 = -0.4684593418325993783650820409805381740605
        b4 = 0.3351397342755897010393098942949569049275
        b5 = 0.2766711191210800975049457263356834696055
        
        c = np.array([a1, a2, a3, a4, a5, a5, a4, a3, a2, a1])
        d = np.array([b1, b2, b3, b4, b5, b4, b3, b2, b1])
    
    elif order == 18: #WHCK
    
        # jupiter test
        
        #a1 = 8
        #a2 = -8
        #a3 = -8
        #a4 = 8
        #a5 = 24
        #a6 = 24
        #b1 = -1
        #b2 = 1
        #b3 = 6
        #b4 = -1
        #b5 = 1
        
        # p2s test
    
        #a1 = 2/8
        #a2 = -2/8
        #a3 = -1/8
        #a4 = 1/8
        #a5 = 7/8
        #a6 = 1/8
        #b1 = -1/6
        #b2 = 1/6
        #b3 = 1
        #b4 = -1/6
        #b5 = 1/6
        
        
        
        a1 = 5/8
        a2 = -0.25
        a3 = 1/8
        a4 = -1/8
        a5 = 0.25
        a6 = 3/8
        b1 = -1/6
        b2 = 1/6
        b3 = 1
        b4 = -1/6
        b5 = 1/6
     
        #c = np.array([a6, a5, a4, a3, a2, a1])
        #d = np.array([b5, b4, b3, b2, b1])
        
        c = np.array([a1, a2, a3, a4, a5, a6])
        d = np.array([b1, b2, b3, b4, b5])
    
    return(c, d)


def cX(rval, vval, p, q, inner, GM, xY, f_selection):
    xvalues = solveKepler(rval, vval, inner, p, xY, rval, vval) 
    xr_next = xvalues[0] * 1
    xv_next = xvalues[1] * 1
    xY = xvalues[2] * 1

    xv_next = xv_next + q * f_selection(xr_next, GM, 0)
    
    xvalues = solveKepler(xr_next, xv_next, inner, -p, xY, xr_next, xv_next)

    return xvalues
    
  
  
def cZ(rval, vval, p, q, inner, GM, Y, f_selection): 
    
    w = cX(rval, vval, p, q, inner, GM, Y, f_selection)
    w = cX(w[0], w[1], -p, -q, inner, GM, Y, f_selection)

    return (w[0], w[1])


def getCorrector(rval, vval, h_val, inner, GM, Y, f_selection, order=17):
       
    h =  h_val #* 17
    
    # Step size calculations
    
    alpha = sqrt(7/40)
    beta = 1 / (48 * alpha)
    
    # These are mostly for testing purposes
    
    if order == 3: 
        A = np.array([alpha, -alpha])
        B = np.array([-beta / 2, beta / 2])
    
    elif order == 5:
        A = np.array([2 * alpha, alpha, -alpha, -2 * alpha])
        B = np.array([-1 / 6 * beta, 5 / 6 * beta, -5 / 6 * beta, 1 / 6 * beta])
     
        
    # Main choice
    
    elif order == 17:
        A = np.zeros(16)
        B = np.zeros(16)
        B[0] = -0.00008704091947232721
        B[1] = 0.00015348298318361457
        B[2] = -0.012770775246667285
        B[3] = 0.06652968674402478
        B[4] = -0.24239903351841396
        B[5] = 0.6510325862986641
        B[6] = -1.3090623112714728
        B[7] = 1.8685517340134143
        B = B * beta
        
        for i in range(8):
            A[i] = (8-i) * alpha
            A[-1-i] = -A[i]
            B[-1-i] = -B[i]

            
       
    p = h * A
    q = h * B
    
    # zval - standard corrector
    # nzval - inverse corrector
    

    zval = np.vstack([cZ(rval, vval, p[0], q[0], inner, GM, Y, f_selection)])
    nzval = np.vstack([cZ(rval, vval, p[-1], -q[-1], inner, GM, Y, f_selection)])
    
    for j in range(1, len(p)):
        zval = cZ(zval[0], zval[1], p[j], q[j], inner, GM, Y, f_selection)
        nzval = cZ(nzval[0], nzval[1], p[-1-j], -q[-1-j], inner, GM, Y, f_selection)

    #zval = np.vstack([cZ(rval, vval, p[-1], q[-1], inner, GM, Y, f_selection)])
    #nzval = np.vstack([cZ(rval, vval, p[0], -q[0], inner, GM, Y, f_selection)])
        
    #for j in range(1, len(p)):
        #zval = cZ(zval[0], zval[1], p[-1-j], q[-1-j], inner, GM, Y, f_selection)
        #nzval = cZ(nzval[0], nzval[1], p[j], -q[j], inner, GM, Y, f_selection)
        

    
    return(zval[0], zval[1], nzval[0], nzval[1])


#--------------------------------------------------------------------------------------------------------------

# NUMERICAL METHOD FUNCTIONS


def forwardEuler (system, h, orbits, plot_type=1):
    r_current = system.r
    size = r_current.size // 2
    v_current = system.v
    orbits_remaining = orbits
    
    if size == 1:
        f_selection = f_function[0]
        
    elif max(system.GM) == 0:
        f_selection = f_function[1]
        
    else:
        f_selection = f_function[2]
        
    energy_initial = 0
    angular_initial = 0
        
    for i in range(size):
        energy_initial += 0.5 * system.GM[i] * (v_current[i, 0]**2 + v_current[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_current[i])
        angular_initial += (r_current[i, 0] * v_current[i, 1] - r_current[i, 1] * v_current[i, 0]) * system.GM[i]
        for j in range(i+1, size):
            energy_initial += -system.GM[i] * system.GM[j] / np.linalg.norm(r_current[i] - r_current[j])
       
    energy_current = energy_initial
    angular_current = angular_initial
    energy_next = energy_initial
    angular_next = angular_initial
    
    last_orbit = 0
    r_data = np.array([r_current])
    checker = 0

    total = round(P / h * orbits_remaining)
    
    if plot_type == 3:
        plt.xscale("linear")
        plt.yscale("linear")
        #plt.xscale("log")
        #plt.yscale("log")
    else:
        if plot_type == 1 or plot_type == 5:
            plt.axis("equal")
            #plt.xlim(-4, 4)
            #plt.ylim(-4, 4)
            
        plt.xscale("linear")
        plt.yscale("linear")

    for counter in range(total):
         
        r_next = r_current + h * v_current
        v_next = v_current + h * f_selection(r_current, system.GM, system.inner)
        
        if plot_type == 1:
            for i in range(size):
                plt.plot((r_current[i, 0], r_next[i, 0]), (r_current[i, 1], r_next[i, 1]), f"{colours[i]}")
        
        elif (counter >= checker * P / h) and (plot_type == 2):
            checker += rate
            error_distance = 0
            for i in range(size):
                error_distance += np.linalg.norm((r_current[i]-system.r[i]) / size)
            plt.plot((checker-rate, checker), (last_orbit, error_distance), f"{colours[i]}")
            last_orbit = error_distance
        
        elif (counter >= checker * P / h) and (plot_type == 3): 
            checker += rate
            energy_next = 0
            
            for i in range(size):
                energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])

            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}")
            energy_current = energy_next * 1         
        
        
        elif (counter >= checker * P / h) and (plot_type == 3.1):  
            checker += rate
            energy_new = 0
            
            for i in range(size):
                energy_new += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_new += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            
            if abs((energy_new - energy_initial) / energy_new) > abs((energy_next - energy_initial) / energy_initial):
                energy_next = max(energy_next, energy_new) * 1
            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}", linewidth=2)
            energy_current = energy_next * 1
            
     
        elif (counter >= checker * P / h) and (plot_type == 4): 
            checker += rate
            angular_next = 0
            
            for i in range(size):
                angular_next += (r_next[i, 0] * v_next[i, 1] - r_next[i, 1] * v_next[i, 0]) * system.GM[i]
                
            plt.plot((checker-rate, checker), (abs((angular_current - angular_initial) / angular_initial), abs((angular_next - angular_initial) / angular_initial)), f"{colours[i]}")
            angular_current = angular_next * 1
            
        elif (plot_type == 5):
            r_data = np.append(r_data, [r_next], axis=0)
        
        v_current = v_next
        r_current = r_next
       
    if plot_type <= 1:
        if size == 1:
            finishingTouches(r_current, plot_type, system.r)
        elif system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        else:
            finishingTouches(r_current, plot_type, system.r, False, False)
    
    elif plot_type == 2:
        print(f"Error: {error_distance}")
        
    elif plot_type == 3 or plot_type == 3.1:
        energy_next = 0
        for i in range(size):
            energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                
        print(f"Energy Error: {abs((energy_next - energy_initial) / energy_initial)}")
    
    elif plot_type == 4:
        print(f"Angular Error: {abs((angular_next - angular_initial) / angular_initial)}")
    
    elif plot_type == 5: 
        if system.inner != 0:
            finishingTouches(r_current, plot_type, system.r, False)
        global v_tr
        v_tr = v_current
        return(r_data)
    
    return(r_current, v_current)
   
 

def modifiedEuler (system, h, orbits, plot_type=1):
    r_current = system.r
    size = r_current.size // 2
    v_current = system.v
    orbits_remaining = orbits
    
    if size == 1:
        f_selection = f_function[0]
        
    elif max(system.GM)  == 0:
        f_selection = f_function[1]
        
    else:
        f_selection = f_function[2]
 
    energy_initial = 0
    angular_initial = 0
        
    for i in range(size):
        energy_initial += 0.5 * system.GM[i] * (v_current[i, 0]**2 + v_current[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_current[i])
        angular_initial += (r_current[i, 0] * v_current[i, 1] - r_current[i, 1] * v_current[i, 0]) * system.GM[i]
        for j in range(i+1, size):
            energy_initial += -system.GM[i] * system.GM[j] / np.linalg.norm(r_current[i] - r_current[j])
       
    energy_current = energy_initial
    angular_current = angular_initial
    energy_next = energy_initial
    angular_next = angular_initial
    
    r_data = np.array([r_current])
    last_orbit = 0
    checker = 0    
    
    total = round(P / h * orbits_remaining)
    
    if plot_type == 3:
        #plt.xscale("linear")
        #plt.yscale("linear")
        #plt.xscale("log")
        plt.yscale("log")
    else:
        if plot_type == 1 or plot_type == 5:
            plt.axis("equal")
            #plt.xlim(-4, 4)
            #plt.ylim(-4, 4)
            
        plt.xscale("linear")
        plt.yscale("linear")

    for counter in range(total):
         
        r_next = r_current + h * v_current
        v_next = v_current + h * f_selection(r_next, system.GM, system.inner)
        
        if plot_type == 1:
            for i in range(size):
                plt.plot((r_current[i, 0], r_next[i, 0]), (r_current[i, 1], r_next[i, 1]), f"{colours[i]}")
        
        elif (counter >= checker * P / h) and (plot_type == 2):
            checker += rate
            error_distance = 0
            for i in range(size):
                error_distance += np.linalg.norm((r_current[i]-system.r[i]) / size)
            plt.plot((checker-rate, checker), (last_orbit, error_distance), f"{colours[i]}")
            last_orbit = error_distance
        
        elif (counter >= checker * P / h) and (plot_type == 3): 
            checker += rate
            energy_next = 0
            
            for i in range(size):
                energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])

            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}")
            energy_current = energy_next * 1        
        
        
        elif (counter >= checker * P / h) and (plot_type == 3.1):  
            checker += rate
            energy_new = 0
            
            for i in range(size):
                energy_new += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_new += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            
            if abs((energy_new - energy_initial) / energy_new) > abs((energy_next - energy_initial) / energy_initial):
                energy_next = max(energy_next, energy_new) * 1
            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}", linewidth=2)
            energy_current = energy_next * 1
            
     
        elif (counter >= checker * P / h) and (plot_type == 4): 
            checker += rate
            angular_next = 0
            
            for i in range(size):
                angular_next += (r_next[i, 0] * v_next[i, 1] - r_next[i, 1] * v_next[i, 0]) * system.GM[i]
            
            plt.plot((checker-rate, checker), (abs((angular_current - angular_initial) / angular_initial), abs((angular_next - angular_initial) / angular_initial)), f"{colours[i]}")
            angular_current = angular_next * 1
            
        elif (plot_type == 5):
            r_data = np.append(r_data, [r_next], axis=0)
        
        v_current = v_next
        r_current = r_next
       
    if plot_type <= 1:
        if size == 1:
            finishingTouches(r_current, plot_type, system.r)
        elif system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        else:
            finishingTouches(r_current, plot_type, system.r, False, False)
    
    elif plot_type == 2:
        print(f"Error: {error_distance}")
        
    elif plot_type == 3 or plot_type == 3.1:
        energy_next = 0
        for i in range(size):
            energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                
        print(f"Energy Error: {abs((energy_next - energy_initial) / energy_initial)}")

    
    elif plot_type == 4:
        print(f"Angular Error: {abs((angular_next - angular_initial) / angular_initial)}")
    
    elif plot_type == 5: 
        if system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        global v_tr
        v_tr = v_current
        return(r_data)
    
    return(r_current, v_current)


def leapfrog (system, h, orbits, plot_type=1, order=default_order):   
    steps = int(2**(order // 2))
    h_new = h
       
    timesteps = getTimestep(order, steps)
          
    c = timesteps[0] * h_new
    d = timesteps[1] * h_new

    r_current = system.r
    size = r_current.size // 2
    v_current = system.v
    orbits_remaining = orbits
    
    if size == 1:
        f_selection = f_function[0]
        
    elif max(system.GM) == 0:
        f_selection = f_function[1]
        
    else:
        f_selection = f_function[2]
        
    energy_initial = 0
    angular_initial = 0
        
    for i in range(size):
        energy_initial += 0.5 * system.GM[i] * (v_current[i, 0]**2 + v_current[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_current[i])
        angular_initial += (r_current[i, 0] * v_current[i, 1] - r_current[i, 1] * v_current[i, 0]) * system.GM[i]
        for j in range(i+1, size):
            energy_initial += -system.GM[i] * system.GM[j] / np.linalg.norm(r_current[i] - r_current[j])
       
    energy_current = energy_initial
    angular_current = angular_initial
    energy_next = energy_initial
    angular_next = angular_initial   
    
    r_data = np.array([r_current])
    last_orbit = 0
    checker = 0
        
    total = round(P / h * orbits_remaining)
    
    if plot_type == 3:
        plt.xscale("linear")
        plt.yscale("linear")
        #plt.xscale("log")
        #plt.yscale("log")
    else:
        if plot_type == 1 or plot_type == 5:
            plt.axis("equal")
            #plt.xlim(-4, 4)
            #plt.ylim(-4, 4)
            
        plt.xscale("linear")
        plt.yscale("linear")

    for counter in range(total):
        r_next = r_current
        v_next = v_current
            
        for leap in range(steps-1):         
            r_next = r_next + c[leap] * v_next
            v_next = v_next + d[leap] * f_selection(r_next, system.GM, system.inner)
            
        r_next = r_next + c[-1] * v_next
        
        if plot_type == 1:
            for i in range(size):
                plt.plot((r_current[i, 0], r_next[i, 0]), (r_current[i, 1], r_next[i, 1]), f"{colours[i]}")
        
        elif (counter >= checker * P / h) and (plot_type == 2):
            checker += rate
            error_distance = 0
            for i in range(size):
                error_distance += np.linalg.norm((r_current[i]-system.r[i]) / size)
            plt.plot((checker-rate, checker), (last_orbit, error_distance), f"{colours[i]}")
            last_orbit = error_distance
        
        elif (counter >= checker * P / h) and (plot_type == 3): 
            checker += rate
            energy_next = 0
            
            for i in range(size):
                energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])

            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}")
            energy_current = energy_next * 1        
        
        
        elif (counter >= checker * P / h) and (plot_type == 3.1):  
            checker += rate
            energy_new = 0
            
            for i in range(size):
                energy_new += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_new += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            
            if abs((energy_new - energy_initial) / energy_new) > abs((energy_next - energy_initial) / energy_initial):
                energy_next = max(energy_next, energy_new) * 1
            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}", linewidth=2)
            energy_current = energy_next * 1
            
     
        elif (counter >= checker * P / h) and (plot_type == 4): 
            checker += rate
            angular_next = 0
            
            for i in range(size):
                angular_next += (r_next[i, 0] * v_next[i, 1] - r_next[i, 1] * v_next[i, 0]) * system.GM[i]
                
            plt.plot((checker-rate, checker), (abs((angular_current - angular_initial) / angular_initial), abs((angular_next - angular_initial) / angular_initial)), f"{colours[i]}")
            angular_current = angular_next * 1
            
        elif (plot_type == 5):
            r_data = np.append(r_data, [r_next], axis=0)
    
        elif (counter >= checker * P / h) and (plot_type == 6):
            checker += 1
            for i in range(1, size):
                plt.scatter(counter, np.linalg.norm(r_current[i] - r_current[0]), c=f"{colours[i]}")
        
        r_current = r_next
        v_current = v_next
       
    if plot_type <= 1:
        if size == 1:
            finishingTouches(r_current, plot_type, system.r)
        elif system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        else:
            finishingTouches(r_current, plot_type, system.r, False, False)
    
    elif plot_type == 2:
        print(f"Error: {error_distance}")    
        
    elif plot_type == 3 or plot_type == 3.1:
        energy_next = 0
        for i in range(size):
            energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                
        print(f"Energy Error: {abs((energy_next - energy_initial) / energy_initial)}")
    
    elif plot_type == 4:
        print(f"Angular Error: {abs((angular_next - angular_initial) / angular_initial)}")
    
    elif plot_type == 5: 
        if system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        global v_tr
        v_tr = v_current
        return(r_data)
    
    return(r_current, v_current)


def rungeKutta (system, h, orbits, plot_type=1): 
    r_current = system.r
    v_current = system.v
    size = r_current.size // 2
    orbits_remaining = orbits
    
    if size == 1:
        f_selection = f_function[0]
        
    elif max(system.GM)  == 0:
        f_selection = f_function[1]
        
    else:
        f_selection = f_function[2]
    
    energy_initial = 0
    angular_initial = 0
        
    for i in range(size):
        energy_initial += 0.5 * system.GM[i] * (v_current[i, 0]**2 + v_current[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_current[i])
        angular_initial += (r_current[i, 0] * v_current[i, 1] - r_current[i, 1] * v_current[i, 0]) * system.GM[i]
        for j in range(i+1, size):
            energy_initial += -system.GM[i] * system.GM[j] / np.linalg.norm(r_current[i] - r_current[j])
       
    energy_current = energy_initial
    angular_current = angular_initial
    energy_next = energy_initial
    angular_next = angular_initial 
    
    r_data = np.array([r_current])
    last_orbit = 0
    checker = 0

    total = round(P / abs(h) * orbits_remaining)
    
    if plot_type == 3:
        plt.xscale("linear")
        plt.yscale("linear")
        #plt.xscale("log")
        #plt.yscale("log")
    else:
        if plot_type == 1 or plot_type == 5:
            plt.axis("equal")
            #plt.xlim(-4, 4)
            #plt.ylim(-4, 4)
            
        plt.xscale("linear")
        plt.yscale("linear")   

    for counter in range(total):

        a1 = h * v_current
        b1 = h * f_selection(r_current, system.GM, system.inner)
              
        r2 = r_current + 0.5 * a1
        v2 = v_current + 0.5 * b1
        
        a2 = h * v2
        b2 = h * f_selection(r2, system.GM, system.inner)
        
        r3 = r_current + 0.5 * a2
        v3 = v_current + 0.5 * b2
        
        a3 = h * v3
        b3 = h * f_selection(r3, system.GM, system.inner)
      
        r4 = r_current + a3
        v4 = v_current + b3
        
        a4 = h * v4
        b4 = h * f_selection(r4, system.GM, system.inner)   

        r_next = r_current + (a1 + 2 * a2 + 2 * a3 + a4) / 6
        v_next = v_current + (b1 + 2 * b2 + 2 * b3 + b4) / 6
        
        if plot_type == 1:
            for i in range(size):
                plt.plot((r_current[i, 0], r_next[i, 0]), (r_current[i, 1], r_next[i, 1]), f"{colours[i]}")
        
        elif (counter >= checker * P / h) and (plot_type == 2):
            checker += rate
            error_distance = 0
            for i in range(size):
                error_distance += np.linalg.norm((r_current[i]-system.r[i]) / size)
            plt.plot((checker-rate, checker), (last_orbit, error_distance), f"{colours[i]}")
            last_orbit = error_distance
        
        elif (counter >= checker * P / h) and (plot_type == 3): 
            checker += rate
            energy_next = 0
            
            for i in range(size):
                energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])

            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}")
            energy_current = energy_next * 1        
        
        
        elif (counter >= checker * P / h) and (plot_type == 3.1):  
            checker += rate
            energy_new = 0
            
            for i in range(size):
                energy_new += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_new += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            
            if abs((energy_new - energy_initial) / energy_new) > abs((energy_next - energy_initial) / energy_initial):
                energy_next = max(energy_next, energy_new) * 1
            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}", linewidth=2)
            energy_current = energy_next * 1
            
     
        elif (counter >= checker * P / h) and (plot_type == 4): 
            checker += rate
            angular_next = 0
            
            for i in range(size):
                angular_next += (r_next[i, 0] * v_next[i, 1] - r_next[i, 1] * v_next[i, 0]) * system.GM[i]
                
            plt.plot((checker-rate, checker), (abs((angular_current - angular_initial) / angular_initial), abs((angular_next - angular_initial) / angular_initial)), f"{colours[i]}")
            angular_current = angular_next * 1
            
        elif (plot_type == 5):
            r_data = np.append(r_data, [r_next], axis=0)
            
        r_current = r_next
        v_current = v_next
    
    if plot_type <= 1:
        if size == 1:
            finishingTouches(r_current, plot_type, system.r)
        elif system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        else:
            finishingTouches(r_current, plot_type, system.r, False, False)

    elif plot_type == 2:
        print(f"Error: {error_distance}")
        
    elif plot_type == 3 or plot_type == 3.1:
       print(f"Energy Error: {abs((energy_next - energy_initial) / energy_initial)}")
    
    elif plot_type == 4:
        print(f"Angular Error: {abs((angular_next - angular_initial) / angular_initial)}")
    
    elif plot_type == 5: 
        if system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        global v_tr
        v_tr = v_current
        return(r_data)
    
    return(r_current, v_current)


def wisdomHolman (system, h, orbits, plot_type=1, scheme=18):   
    h_new = h
    
    if scheme == 18:
        steps = 6
    elif scheme == 10.642:#
        steps = 10
    elif scheme == 10.64:#
        steps = 9
    elif scheme == 10.4 or scheme == 8.64:#
        steps = 8
    elif scheme == 10:
        steps = 11
    elif scheme == 4:
        steps = 5    
    elif scheme == 2:
        steps = 3
    elif scheme == 1.1: #testing
        steps = 3
    else:
        steps = 2
    
    r_current = system.r
    size = r_current.size // 2
    v_current = system.v
    orbits_remaining = orbits
    
    #if max(system.GM) == 0:
    f_selection = f_function[1]
        
    #else:
        #f_selection = f_function[2]
          
    energy_initial = 0
    angular_initial = 0
        
    for i in range(size):
        energy_initial += 0.5 * system.GM[i] * (v_current[i, 0]**2 + v_current[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_current[i])
        angular_initial += (r_current[i, 0] * v_current[i, 1] - r_current[i, 1] * v_current[i, 0]) * system.GM[i]
        for j in range(i+1, size):
            energy_initial += -system.GM[i] * system.GM[j] / np.linalg.norm(r_current[i] - r_current[j])
       
    energy_current = energy_initial
    angular_current = angular_initial
    energy_next = energy_initial
    angular_next = angular_initial
    energy_store = energy_initial * 1
    
    r_data = np.array([r_current])
    last_orbit = 0
    checker = 0
        
    total = round(P / abs(h_new) * orbits_remaining)

    if plot_type == 3 or plot_type == 6:
        #plt.xscale("linear")
        #plt.yscale("linear")
        #plt.xscale("log")
        plt.yscale("log")
        #plt.ylim(10**-9, 1)
        
        #plt.xlabel("Number of orbits (closest moon)")
        #plt.ylabel("Eccentricity")
        
    else:
        if plot_type == 1 or plot_type == 5:
            plt.axis("equal")
            #plt.xlim(-4, 4)
            #plt.ylim(-4, 4)
            
        plt.xscale("linear")
        plt.yscale("linear")
        
    values = np.array([system.r, system.v])
    r_next = r_current
    v_next = v_current
    Y = 0.4 * np.ones(len(r_current)) 
    
    if scheme == 17 or scheme == 18.1:
        corrector = getCorrector(r_next, v_next, h_new, system.inner, system.GM, Y, f_selection)
        r_next = corrector[0]
        v_next = corrector[1]
        saba = getSABA(1)          
        a = saba[0] * h_new
        b = saba[1] * h_new
        
        if scheme == 17:
            energy_initial = 0
            for i in range(size):
                energy_initial += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_initial += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])

                
    elif scheme == 18:
        corrector = getCorrector(r_next, v_next, h_new, system.inner, system.GM, Y, f_selection)
        r_next = corrector[0]
        v_next = corrector[1] 
        saba = getSABA(18)   
        a = saba[0] * h_new
        b = saba[1] * h_new
        
        #if plot_type == 3:
            #energy_next = 0
            #for i in range(size):
                #energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                #for j in range(i+1, size):
                    #energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            #energy_initial = energy_next * 1
            #energy_current = energy_next * 1
        
    else:
        saba = getSABA(scheme)         
        a = saba[0] * h_new
        b = saba[1] * h_new

    r_max = r_current * 1
    r_min = r_current * 1
    new_e = np.zeros(len(r_current)) 
    old_e = new_e
    previous = [0, 0]
    
     
    for counter in range(total):  

        if scheme < 18.1:
            for leap in range(steps-1):              
                # WH KEPLER STEP 
                values = solveKepler(r_next, v_next, system.inner, a[leap], Y, values[0], values[1])    
                r_next = values[0]
                v_next = values[1]
                Y = values[2]
            
                # WH PERTURBATION STEP (kick step only)
                v_next = v_next + b[leap] * f_selection(r_next, system.GM, 0)
    
                
        else:
            for leap in range(steps-1):
                # WH KEPLER STEP 
                values = solveKepler(r_next, v_next, system.inner, a[leap], Y, values[0], values[1])    
                r_next = values[0]
                v_next = values[1]
                Y = values[2]
            
                # WH LAZY PERTURBATION STEP
                r_leap = r_next + b[leap]**2 * f_selection(r_next, system.GM, 0) / 12
                v_next = v_next + b[leap] * f_selection(r_leap, system.GM, 0)
            
        # WH END STEP 
        values = solveKepler(r_next, v_next, system.inner, a[-1], Y, r_next, v_next)
        r_next = values[0]
        v_next = values[1]
        Y = values[2] 
        
        if plot_type == 1:
            for i in range(size):
                plt.plot((r_current[i, 0], r_next[i, 0]), (r_current[i, 1], r_next[i, 1]), f"{colours[i]}")
                
                
        elif (counter >= checker * P / h) and (plot_type == 2):
            checker += rate
            error_distance = 0
            for i in range(size):
                error_distance += np.linalg.norm((r_current[i]-system.r[i]) / size)
            plt.plot((checker-rate, checker), (last_orbit, error_distance), f"{colours[i]}")
            last_orbit = error_distance
            
            
        elif (counter >= checker * P / h) and (plot_type == 3): 
            checker += rate
            energy_next = 0
            
            for i in range(size):
                energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])

            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}")
            energy_current = energy_next * 1   
            print(counter / total)
        
        
        elif (counter >= checker * P / h) and (plot_type == 3.1):  
            checker += rate
            energy_new = 0
            
            for i in range(size):
                energy_new += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                for j in range(i+1, size):
                    energy_new += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            
            if abs((energy_new - energy_initial) / energy_new) > abs((energy_next - energy_initial) / energy_initial):
                energy_next = max(energy_next, energy_new) * 1
            plt.plot((checker-rate, checker), (abs((energy_current - energy_initial) / energy_initial), abs((energy_next - energy_initial) / energy_initial)), f"{colours[i]}", linewidth=2)
            energy_current = energy_next * 1
            
     
        elif (counter >= checker * P / h) and (plot_type == 4): 
            checker += rate
            angular_next = 0
            
            for i in range(size):
                angular_next += (r_next[i, 0] * v_next[i, 1] - r_next[i, 1] * v_next[i, 0]) * system.GM[i]
                
            plt.plot((checker-rate, checker), (abs((angular_current - angular_initial) / angular_initial), abs((angular_next - angular_initial) / angular_initial)), f"{colours[i]}")
            angular_current = angular_next * 1
        
        elif (plot_type == 5):
            r_data = np.append(r_data, [r_next], axis=0)
            
            
        elif (plot_type == 6) and (counter > checker * rate): 
            for i in range(size):
                length = np.linalg.norm(r_next[i])
                if np.linalg.norm(r_max[i]) < length:
                    r_max[i] = r_next[i] * 1
                    new_e[i] = (np.linalg.norm(r_max[i]) - np.linalg.norm(r_min[i])) / (np.linalg.norm(r_max[i]) + np.linalg.norm(r_min[i]))
                    
                elif length < np.linalg.norm(r_min[i]):
                    r_min[i] = r_next[i] * 1
                    new_e[i] = (np.linalg.norm(r_max[i]) - np.linalg.norm(r_min[i])) / (np.linalg.norm(r_max[i]) + np.linalg.norm(r_min[i]))
            
            if counter > checker * rate * 4:         
                curr_e = np.sum(new_e) / size
                
                current_orbit = orbits_remaining * counter / total
                
                plt.plot((previous[0], current_orbit), (old_e[0], new_e[0]), "lightcoral", "dotted")
                plt.plot((previous[0], current_orbit), (old_e[1], new_e[1]), "mediumaquamarine", "dotted")
                plt.plot((previous[0], current_orbit), (old_e[2], new_e[2]), "skyblue", "dotted")
                plt.plot((previous[0], current_orbit), (old_e[3], new_e[3]), "plum", "dotted")
                plt.plot((previous[0], current_orbit), (previous[1], curr_e),  "k")
                
                old_e = new_e * 1
                
                
                checker += 0.25 * (P / abs(h_new))

                r_max = r_current * 1
                r_min = r_current * 1
                previous = [current_orbit, curr_e]
                print(counter / total)

            
        r_current = 1 * r_next
        
    
    if scheme == 17:# or scheme == 18 or scheme == 18.1:
        corrector = getCorrector(r_next, v_next, h_new, system.inner, system.GM, Y, f_selection)
        r_next = corrector[2]
        v_next = corrector[3]
        
        #if plot_type == 3:
            #energy_next = 0
            #energy_current = 0
            #for i in range(size):
                #energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
                #plt.plot((checker, checker+10), (((energy_current - energy_initial) / energy_initial), ((energy_next - energy_store) / energy_store)), f"{colours[i]}-")       
            #checker += 1
            #energy_current = energy_next
            #print(energy_current)
            #print(energy_next)
    
    r_current = r_next
    v_current = v_next
        
    if plot_type <= 1:
        if size == 1:
            finishingTouches(r_current, plot_type, system.r)
        elif system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        else:
            finishingTouches(r_current, plot_type, system.r, False, False)

    elif plot_type == 2:
        error_distance = 0
        for i in range(size):
            error_distance += np.linalg.norm((r_current[i]-system.r[i]) / size)
        print(f"Error: {error_distance}")
        
    elif plot_type == 3 or plot_type == 3.1:
        energy_current = energy_next
        energy_next = 0
        for i in range(size): 
            energy_next += 0.5 * system.GM[i] * (v_next[i, 0]**2 + v_next[i, 1]**2) - system.inner * system.GM[i] / np.linalg.norm(r_next[i])
            for j in range(i+1, size):
                energy_next += -system.GM[i] * system.GM[j] / np.linalg.norm(r_next[i] - r_next[j])
            
        print(f"Energy Error: {abs((energy_next - energy_store) / energy_store)}")
    
    elif plot_type == 4:
        print(f"Angular Error: {abs((angular_next - angular_initial) / angular_initial)}")
    
    elif plot_type == 5: 
        if system.inner == 0:
            finishingTouches(r_current, plot_type, system.r, False)
        global v_tr
        v_tr = v_current
        return(r_data)
    
    
    
    return(r_current, v_current)


#--------------------------------------------------------------------------------------------------------------

numerical_method = [forwardEuler, modifiedEuler, leapfrog, rungeKutta, wisdomHolman]

#--------------------------------------------------------------------------------------------------------------

# ANIMATION FOR PLOTS


def orbitAnimation(system, method, h=h_initial, tr=False):
    global fig
    global ax
    global orbit_plt
    global endpoint_plt
    fig, ax = plt.subplots()
    ax.axis("equal")
    
    if tr == False:
        results = numerical_method[method](system, h, orbits, 5)
    
    else:
        first = numerical_method[method](system, h, orbits / 2, 5)
        new_sys = system
        new_sys.r = first[-1]
        new_sys.v = -v_tr
        second = numerical_method[method](new_sys, h, orbits / 2, 5)
        results = np.append(first, second, axis=0)

    
    bodies = results.size // results[:, 0].size
    rx = np.zeros((bodies, results[:, 0, 0].size))
    ry = np.zeros((bodies, results[:, 0, 0].size))
    
    orbit_plt = {}
    endpoint_plt = {}
    
    for i in range(bodies):
        rx[i] = results[:, i, 0]
        ry[i] = results[:, i, 1]

        orbit_plt[i] = ax.plot(rx[i], ry[i], color=f"{colours[i-1]}")[0]
        endpoint_plt[i] = ax.plot(rx[i], ry[i], "o", color=f"{colours_alt[i-1]}")[0]
        ax.plot(0, 0, "y*")
        

    def update(frame):    
        for i in range(bodies):
            trail = frame #int(1.5 * P/h) #frame
            
            if frame > (orbits * (1 + P / h_initial)/2) and tr:
                #endpoint_plt[i].set_color(f"{colours_alt[i-3]}")
                orbit_plt[i].set_xdata(rx[i][frame-trail:frame+1])
                orbit_plt[i].set_ydata(ry[i][frame-trail:frame+1])
                endpoint_plt[i].set_xdata(rx[i][frame:frame+1])
                endpoint_plt[i].set_ydata(ry[i][frame:frame+1])
                
            elif frame > trail:
                endpoint_plt[i].set_color(f"{colours_alt[i-1]}")
                orbit_plt[i].set_xdata(rx[i][frame-trail:frame+1])
                orbit_plt[i].set_ydata(ry[i][frame-trail:frame+1])
                endpoint_plt[i].set_xdata(rx[i][frame:frame+1])
                endpoint_plt[i].set_ydata(ry[i][frame:frame+1])
                
            else:
                endpoint_plt[i].set_color(f"{colours_alt[i-1]}")
                orbit_plt[i].set_xdata(rx[i][0:frame+1])
                orbit_plt[i].set_ydata(ry[i][0:frame+1])
                endpoint_plt[i].set_xdata(rx[i][frame:frame+1])
                endpoint_plt[i].set_ydata(ry[i][frame:frame+1])
                
                
                        
        return(ani)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=(orbits * round(1 + P / h_initial)), interval=10)
    
    #ani.save('anim -.gif')  
    return(ani)


#--------------------------------------------------------------------------------------------------------------

# INTEGRATORS
# Forward Euler - forwardEuler
# Modified Euler - modifiedEuler
# Leapfrog - leapfrog (order = 2)
# Yoshida Leapfrog - leapfrog (order = 4 / 6 / 8)
# Runge-Kutta 4 - rungeKutta
# Wisdom-Holman - wisdomHolman (scheme = 1)
# SABA with positive step constraint - wisdomHolman (scheme = 2 / 4 / 10)
# SABA without constraints - wisdomHolman (scheme = 8.64 / 10.4 / 10.64 / 10.642)
# Wisdom-Holman Symplectic Corrector - wisdomHolman (scheme = 17)
# Wisdom-Holman Modified Kernel - wisdomHolman (scheme = 18)
# Wisdom-Holman Lazy Implementer's Kernel - wisdomHolman (scheme = 18.1)


# PLOT TYPES
# 0 - None
# 1 - Position (default)
# 2 - Absolute error (where system returns to start point only)
# 3 - Fractional error in total energy
# 3.1 - Max energy error
# 4 - Fractional error in angular momentum
# 5 - Animated orbit paths (use orbitAnimation function instead)
# 6 - Eccentricity (WH-only)




# FIGURE 1
#Parameters: e = 0.5, h_initial = P / 400, orbits = 10

#rate = 0.11

#colours[0] = "r"
#plt.figure(figsize=(7, 7))
#forwardEuler(p1b, h_initial, orbits, 1)
#plt.title("Forward Euler")
#plt.tight_layout()

#colours[0] = "g"
#plt.figure(figsize=(7, 7))
#modifiedEuler(p1b, h_initial, orbits, 1)
#plt.title("Modified Euler")
#plt.tight_layout()

#colours[0] = "b"
#plt.figure(figsize=(7, 7))
#rungeKutta(p1b, h_initial * 4, orbits, 1)
#plt.title("Runge-Kutta 4")
#plt.tight_layout()


# FIGURE 2
#Parameters: e = 0.5, h_initial = P / 400, orbits = 25000

#rate = 0.11
#plt.figure(figsize=(21, 4.75))
#colours[0] = "g"
#modifiedEuler(p1b, h_initial, orbits, 2)
#colours[0] = "b"
#rungeKutta(p1b, h_initial * 4, orbits, 2)
#plt.xlim(0, orbits)
#plt.ylim(0, 3.5)
#plt.tight_layout()


# FIGURE 3
#Parameters: e = 0.5, h_initial = P / 400, orbits = 10

#rate = 25.11

#plt.figure(figsize=(21, 4.75))
#plt.title("Energy")
#colours[0] = "r"
#forwardEuler(p1b, h_initial, orbits, 3)
#colours[0] = "green"
#modifiedEuler(p1b, h_initial, orbits, 3)
#colours[0] = "olive"
#leapfrog(p1b, h_initial, orbits, 3, 2)
#colours[0] = "b"
#rungeKutta(p1b, h_initial * 4, orbits, 3)
#plt.yscale("log")
#plt.xlim(0, orbits)
#plt.ylim(10**-10, 10)
#plt.tight_layout()

#plt.figure(figsize=(21, 4.75))
#plt.title("Angular Momentum")
#colours[0] = "r"
#forwardEuler(p1b, h_initial, orbits, 4)
#colours[0] = "green"
#modifiedEuler(p1b, h_initial, orbits, 4)
#colours[0] = "olive"
#leapfrog(p1b, h_initial, orbits, 4, 2)
#colours[0] = "b"
#rungeKutta(p1b, h_initial * 4, orbits, 4)
#plt.yscale("log")
#plt.xlim(0, orbits)
#plt.ylim(10**-18, 10)
#plt.tight_layout()


# FIGURE 4
#Parameters: e = 0.5, h_initial = P / 70, orbits = 0.5

#rate = 0.11

#orbits = 0.5
#disable_analytic = True

#plt.figure(figsize=(7, 7))
#colours[0] = "saddlebrown"
#colswap = 1
#timeReverse(p1b, h_initial, orbits, 0)
#plt.title("Forward Euler")

#colours[0] = "r"
#colswap = 2
#forwardEuler(p1b, h_initial, orbits, 1)
#plt.title("Forward Euler")
#plt.xlim(-0.9, 2.8)
#plt.ylim(-1.2, 2.5)
#plt.tight_layout()

#plt.figure(figsize=(7, 7))
#colours[0] = "darkolivegreen"
#colswap = 1
#timeReverse(p1b, h_initial, orbits, 1)

#colours[0] = "g"
#colswap = 2
#modifiedEuler(p1b, h_initial, orbits, 1)
#plt.title("Modified Euler")
#plt.xlim(-0.7, 2)
#plt.ylim(-1, 1.6)
#plt.tight_layout()

#plt.figure(figsize=(7, 7))
#colours[0] = "midnightblue"
#colswap = 1
#timeReverse(p1b, h_initial * 4, orbits, 3)

#colours[0] = "b"
#colswap = 2
#rungeKutta(p1b, h_initial * 4, orbits, 1)
#plt.title("Runge-Kutta 4")
#plt.xlim(-0.7, 2)
#plt.ylim(-1, 1.6)
#plt.tight_layout()

#plt.figure(figsize=(7, 7))
#colours[0] = "darkgoldenrod"
#colswap = 1
#timeReverse(p1b, h_initial, orbits, 2)

#colours[0] = "olive"
#colswap = 2
#leapfrog(p1b, h_initial, orbits, 1, 2)
#plt.title("Leapfrog")
#plt.xlim(-0.7, 2)
#plt.ylim(-1, 1.6)
#plt.tight_layout()


# FIGURE 5

#Parameters: e = 0, h_initial = P / 100 and P / 1000, orbits = 10000

#rate = 9.999

#plt.figure(figsize=(20, 5))
#colours[1] = "orangered"
#colours[3] = "orangered"
#leapfrog(p1s, h_initial * 15, orbits, 3, 8)
#colours[1] = "green"
#colours[3] = "green"
#colours[0] = "green"
#modifiedEuler(p1s, h_initial, orbits, 3)
#colours[1] = "olive"
#colours[3] = "olive"
#leapfrog(p1s, h_initial, orbits, 3, 2)
#colours[1] = "b"
#colours[3] = "b"
#colours[0] = "b"
#rungeKutta(p1s, h_initial * 4, orbits, 3)
#colours[3] = colours[1]
#colours[1] = "m"
#colours[3] = "m"
#wisdomHolman(p1s, h_initial, orbits, 3, 1)
#colours[1] = "teal"
#colours[3] = "teal"
#wisdomHolman(p1s, h_initial * 8, orbits, 3, 10.64)
#colours[1] = "k"
#colours[3] = "k"
#wisdomHolman(p1s, h_initial * 2, orbits, 3, 18.1)
#plt.xlabel("Orbits")
#plt.ylabel("Relative energy error")
#plt.yscale("log")
#plt.xlim(0, orbits * 1.25)
#plt.ylim(5*10**-17, 10)
#plt.title("h = P / 1000")


#t = 0.9
#b = 0.15
#l = 0.07
#r = 0.97
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)



# FIGURE 6

#Parameters: e = 0, h_initial = P / 50, orbits = 100000

#rate = 425.25

#wisdomHolman(jupiter, h_initial, orbits, 6, 18.1)
#plt.ylim(0.000025, 0.0003)
#plt.xlim(0, orbits)
#plt.yscale("linear")


# ANIM 1
#colours[-1] = "c"
#colours_alt = ["cyan", "brown", "purple", "blue"]
#orbitAnimation(jupiter, 4, h_initial)


#--------------------------------------------------------------------------------------------------------------


colours[1] = "green"
colours[3] = "green"
colours[0] = "green"
##modifiedEuler(p1s, h_initial, orbits, 3)
colours[1] = "olive"
colours[3] = "olive"
##leapfrog(p1s, h_initial, orbits, 3, 2)
#leapfrog(p1s, h_initial, orbits, 2, 4)
#leapfrog(p1s, h_initial, orbits, 2, 6)
colours[1] = "orangered"
colours[3] = "orangered"
##leapfrog(p1s, h_initial * 15, orbits, 3, 8)
colours[1] = "b"
colours[3] = "b"
colours[0] = "b"
#rungeKutta(p1s, h_initial * 4, orbits, 3)
colours[3] = colours[1]
colours[1] = "m"
colours[3] = "m"
##wisdomHolman(p1s, h_initial, orbits, 3, 1)
colours[1] = "r"
colours[3] = "r"
##wisdomHolman(p1s, h_initial * 2, orbits, 3, 2)
colours[1] = "gold"
colours[3] = "gold"
##wisdomHolman(p1s, h_initial * 4, orbits, 3, 4)
colours[1] = "sienna"
colours[3] = "sienna"
##wisdomHolman(p1s, h_initial * 10, orbits, 3, 10)
colours[1] = "hotpink"
colours[3] = "hotpink"
##wisdomHolman(p1s, h_initial * 7, orbits, 3, 10.4)
colours[1] = "darkslategray"
colours[3] = "darkslategray"
##wisdomHolman(p1s, h_initial * 8, orbits, 3, 10.64)
colours[1] = "lightseagreen"
colours[3] = "lightseagreen"
#wisdomHolman(p1s, h_initial * 9, orbits, 3, 10.642)
##wisdomHolman(p1s, h_initial, orbits, 3, 17)
colours[1] = "midnightblue"
colours[3] = "midnightblue"
##wisdomHolman(p1s, h_initial * 5, orbits, 3, 18)
colours[1] = "k"
colours[3] = "k"
##wisdomHolman(p1s, h_initial * 2, orbits, 3, 18.1)
#timeReverse(p1b, h_initial, orbits, 2, 1)

#colours = ["grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "white"]
#colours_alt = ["grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "grey", "orangered"]

#colours_alt = ["cyan", "brown", "purple", "orange"]
colours = ["blue", "red", "magenta", "olive"]

#orbitAnimation(p1s, 2, h_initial, True)

#plt.xlim(-8, 8)
#plt.ylim(-8, 8)