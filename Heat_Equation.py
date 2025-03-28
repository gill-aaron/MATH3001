# -*- coding: utf-8 -*-

# %matplotlib qt5

# HEAT EQUATION

import numpy as np
import seaborn as sns
import pandas as pd
from math import sin, pi, e, log
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

plt.rcParams.update({"font.size": 20})
plt.rcParams.update({"lines.linewidth": 2})

#--------------------------------------------------------------------------------------------------------------

# CONSTANTS

K = 1
L = 1
N_init = 30
C_init = 0.3

end_time = 1 #1/16 #3

#h_x = L / (N - 1)
#h_t = C * h_x**2 / K


#--------------------------------------------------------------------------------------------------------------

# GENERAL PURPOSE FUNCTIONS

def getPoints(N, min_point=0):
    # Gets N evenly-distributed points
    x = np.zeros(N)
    for i in range(0, (N+1)//2):
        x[i] = i *  L/(N-1)
        x[N-i-1] = L - i *  L/(N-1)
    return x


def f(x, N, endpoint = 0):
    x_val = x * 1
    for i in range(1, N-1):
        x_val[i] = sin(1 * pi * x_val[i] / L)
        #x_val[i] = abs(sin(2 * pi * x_val[i] / L))**4

    x_val[0] = 0
    x_val[-1] = endpoint
    
    #x_val = np.ones(N)
    #x_val[N//2+1:] *= 0
    
    #x_val = np.ones(N) * 0.1
    return x_val


def plotVals(s, N, p, allow_col=1, scatter=False, plotcol=(-1,-1,-1)):
    col = np.zeros((N, 3))

    col[0] = (min(max(s[0]+1, 0), 1), 0, min(max(1-(s[0]+1)/2, 0), 1))
    col[0] *= allow_col
    if scatter==True:
        if plotcol[0] == -1:
            plt.scatter(p[0], s[0], color=abs(col[0]), zorder=0)
        else:
            plt.scatter(p[0], s[0], color=abs(plotcol), zorder=0)

    for j in range(1, N-1):
        col[j] = (min(max(s[j]+1, 0), 1), 0, min(max(1-(s[j]+1)/2, 0), 1))
        col[j] *= allow_col
        
        if plotcol[0] == -1:
            plt.plot((p[j-1], p[j]), (s[j-1], s[j]), c=abs((col[j]+col[j-1]) / 2), zorder=0)
            if scatter==True:
                plt.scatter(p[j], s[j], color=abs(col[j]), zorder=2)
        else:
            plt.plot((p[j-1], p[j]), (s[j-1], s[j]), c=abs(plotcol), zorder=0)
            if scatter==True:
                plt.scatter(p[j], s[j], color=abs(plotcol), zorder=2)
    
    col[-1] = (min(max(s[-1]+1, 0), 1), 0, min(max(1-(s[-1]+1)/2, 0), 1))
    col[-1] *= allow_col
    
    if plotcol[0] == -1:
        plt.plot((p[j], p[j+1]), (s[j], s[j+1]), c=abs((col[j]+col[j+1]) / 2), zorder=0)
        if scatter==True:
            plt.scatter(p[-1], s[-1], color=abs(col[-1]), zorder=0)
    
    else:
        plt.plot((p[j], p[j+1]), (s[j], s[j+1]), c=abs(plotcol), zorder=0)
        if scatter==True:
            plt.scatter(p[-1], s[-1], color=abs(plotcol), zorder=0)


def heatMap(data, x0=0, x1=L, y0=0, y1=end_time): 
    #plt.imshow(np.log(data), interpolation="sinc", cmap="plasma", origin="lower",
               #aspect="auto", extent=(x0, x1, y0, y1))
    
    #plt.imshow(np.log(data + e), interpolation="none", cmap="plasma", origin="lower",
               #aspect="auto", extent=(x0, x1, y0, y1))
    
    #heat_data = pd.DataFrame(data, np.arange(0,33/32,1/32))
    heat_data = pd.DataFrame(data, getPoints(len(data)))
    sns.heatmap(heat_data, cmap="plasma", vmin=0, vmax=1, xticklabels=4, yticklabels=(len(data)//4))
    #sns.heatmap(heat_data, cmap="plasma", vmin=0, vmax=1, xticklabels=4, yticklabels=4)

def analyticSolution(N, C, end_time, plot_type=1, scatter=True):
    points = getPoints(N)
    s_current = f(points, N)
    values = 0
    
    h_x = L / (N - 1)
    h_t = C * h_x**2 / K
    
    #if plot_type == 1:
        #plotVals(s_current, N, points, 0)
    
    if plot_type == 0:
        values = np.zeros((1 + int(end_time // abs(h_t)), N))
        for k in range(N):
            values[0, k] = s_current[k]

    for counter in range(int(end_time // h_t)): 
        s_next = s_current * 0
        for i in range(1, N-1):
            prev1 = 5
            prev2 = 5
            n = 1
            A = 1
            
            while (s_next[i] != prev1) and (s_next[i] != prev2) and (n < 100):              
                s_next[i] += A * sin((n * pi * points[i]) / L) * e**(-K * (counter+1) * h_t * (n * pi / L)**2)
                prev2 = prev1
                prev1 = s_next[i]
                n += 2
                A = 2 / L * -sin(pi * n) / (pi * (n**2 - 1))

        #if plot_type == 1:
            #plotVals(s_next, N, points, 0)
        if plot_type == 0:
            for k in range(N):
                values[counter+1, k] = s_next[k]
        
        s_current = s_next * 1
    
    if plot_type == 1:
        plotVals(s_current, N, points, 0, scatter)
    
    return values


def findError(N, C, end_time, method, endpoint=0, print_out=1): 
    analytic = analyticSolution(N, C, end_time, plot_type=0)
    numerical = numerical_method[method](N, C, end_time, plot_type=0)
    
    print(analytic)
    print("")
    print(numerical)
    
    square_dif = (analytic[-1] - numerical[-1])**2
    rms_error = 0
    
    #for val in range(len(square_dif)):
    #    for p in range(N-1):
     #       rms_error += 0.5 * (square_dif[val, p] + square_dif[val, p+1]) * 1 / (N-1)
    #rms_error = rms_error**0.5
    
    #rms_error2 = np.sum(np.trapz(square_dif, dx = 1/(N-1)))
    #rms_error2 = rms_error**0.5
    
    #rms_error = np.sqrt(np.mean(square_dif))

    rms_error2 = np.sqrt(np.trapz(square_dif) / (N-1))
    
    
    if print_out == 1:
        print(f"RMS Error: {rms_error2}")
        #print(f"RMS Error 2: {rms_error2}")
        print(f"Norm of Error: {np.linalg.norm(analytic[-1] - numerical[-1])}")
        print(analytic[-1])
        print(numerical[-1])
        print("")
        
    return(rms_error2)

    
def findMultiError(Nmin, Nmax, N_delta, Cmin, Cmax, C_delta, end_time, method, plot=1, col="r", order=2):
    N_current = Nmin * 1
    C_current = Cmin * 1
    err_prev = findError(N_current, C_current, end_time, method)
    N_prev = 0
    C_prev = 0
    err_col = (1, 0, 0)
    
    data = []
    temp_data = []
    
    while C_current <= Cmax:
    
        while N_current <= Nmax:
            err_current = findError(N_current, C_current, end_time, method)
            temp_data.append(err_current)

            if N_current > Nmin and plot == 1:
                plt.plot((N_prev, N_current), (err_prev, err_current), c=col)
                err_prev = err_current * 1
            
            N_prev = N_current * 1
            N_current += N_delta            
        
        if C_current > Cmin and plot == 0:
            if order == 2:
                plt.plot((C_prev**2, C_current**2), (err_prev, err_current), c=col)
            else:
                plt.plot((C_prev, C_current), (err_prev, err_current), c=col)
            
            err_prev = err_current * 1
        
        data.append(temp_data)
        temp_data = []
        C_prev = C_current * 1
        
        if order == 2:
            C_current = np.sqrt(C_current**2 + C_delta)
        
        else:
            C_current += C_delta
        
        N_current = Nmin * 1
        err_col = (1 - C_current, 0, C_current)
        
    
    if plot == 2:
        heatMap(data, Nmin, Nmax, Cmin, Cmax)

    if plot == 1:
        plt.xlabel("N")
        plt.ylabel("RMS Error")
        plt.tight_layout()
    elif plot == 0:
        if order == 2:
            plt.xlabel("C**2")
        else:
            plt.xlabel("C")
        plt.ylabel("RMS Error")
        plt.tight_layout()
    return data


def compareMultiError(Nmin, Nmax, N_delta, Cmin, Cmax, C_delta, end_time, method1, method2):
    method_data_1 = np.array(findMultiError(Nmin, Nmax, N_delta, Cmin, Cmax, C_delta, end_time, method1, 0))
    method_data_2 = np.array(findMultiError(Nmin, Nmax, N_delta, Cmin, Cmax, C_delta, end_time, method2, 0))
    new_data = method_data_1 - method_data_2
    
    heatMap(new_data, Nmin, Nmax, Cmin, Cmax)

          

#--------------------------------------------------------------------------------------------------------------

# NUMERICAL METHOD FUNCTIONS

def FTCS(N, C, end_time, endpoint=0, plot_type=1, eq=0, col=(-1,-1,-1), equation=0):
    points = getPoints(N)
    s_current = f(points, N, endpoint)
    s_next = s_current * 1
    s_old = s_current
    
    h_x = L / (N - 1)
    h_t = C * h_x**2 / K
    
    #if plot_type == 1:
        #plotVals(s_next, N, points)
    
    values = np.zeros((1 + int(end_time // abs(h_t)), N))
    for k in range(N):
        values[0, k] = s_next[k]

    for counter in range(int(end_time // h_t)): 
        for i in range(1, N-1):
            # HEAT
            s_next[i] = s_current[i] + C * (s_current[i+1] + s_current[i-1] - 2 * s_current[i]) 
            
            # WAVE
            #s_next[i] = 2 * s_current[i] - s_old[i] + 5 * C * (s_current[i+1] + s_current[i-1] - 2 * s_current[i])
            
            # TRANSPORT
            #s_next[i] += C/10 * (s_current[i-1] - s_current[i])
        #s_next[-1] += C/10 * (s_current[-2] - s_current[-1])

        # Save the numerical explosion for wave / transport
            
        
        if plot_type == 1:
            pass
            #plotVals(s_next, N, points)
            
        else:
            for k in range(N):
                values[counter+1, k] = s_next[k]
        
        s_old = s_current
        s_current = s_next * 1
    
    if plot_type == 1:
        plotVals(s_current, N, points, col)
        #print(s_current)
        
    elif plot_type == 2:
        heatMap(values)
    
    return values


def FTCS2D(N, C, end_time, endpoint=0, plot_type=1):
    points = getPoints(N)
    s_current = np.zeros((N, N))
    
    for i in range(1, N-1):
        s_current[i] = f(points, N, endpoint)
    
    for i in range(1, N-1):
        s_current[:,i] *= f(points, N, endpoint)
    
    s_next = s_current
    
    h_x = L / (N - 1)
    h_t = C * h_x**2 / K
    
    #if plot_type == 1:
        #plotVals(s_next, N, points)
    
    values = np.zeros((1 + int(end_time // h_t), N, N))

    for u in range(N):
        for v in range(N):
            values[0, u, v] = s_next[u, v]

    for counter in range(int(end_time // h_t)): 
        for i in range(1, N-1):
            for j in range(1, N-1):
                s_next[i,j] = s_current[i,j] + C * ((s_current[i+1,j] + s_current[i-1,j] - 2 * s_current[i,j]) + (s_current[i,j+1] + s_current[i,j-1] - 2 * s_current[i,j]))
           
        
        #if plot_type == 1:
            #plotVals(s_next, N, points)
            
        else:
            for u in range(N):
                for v in range(N):
                    values[counter+1, u, v] = s_next[u, v]
        
        #s_old = s_current
        s_current = s_next * 1
    
    if plot_type == 2:
        heatMap(values[-1])
        
    return values


def BTCS(N, C, end_time, endpoint=0, plot_type=1, eq=0):
    points = getPoints(N)
    s_current = f(points, N, endpoint)
    s_next = s_current * 1
    
    h_x = L / (N - 1)
    h_t = C * h_x**2 / K
    
    a = -C / h_t
    b = (1 + 2 * C) / h_t
    
    
    x = np.zeros(N)
    y = np.zeros(N-1)
    x[0] = 1
    y[0] = 0
    
    for i in range(1, N-1):
        x[i] = b - a * y[i-1]
        y[i] = a / x[i]
    
    x[-1] = 1
  
    d = np.zeros(N)
    w = np.zeros(N)

    
    if plot_type == 1:
        plotVals(s_next, N, points)
    
    values = np.zeros((1 + int(end_time // h_t), N))
    for k in range(N):
        values[0, k] = s_next[k]

    for counter in range(int(end_time // h_t)):
        d[0] = s_current[0]
        d[-1] = s_current[-1]
        
        w[0] = s_current[0] / x[0]
        
        
        for i in range(1, N-1):
            d[i] = 1 / h_t * s_current[i]
            w[i] = (d[i] - a * w[i-1]) / x[i]

        for i in range(1, N-1):
            s_next[-i-1] = w[-i-1] - s_next[-i] * y[-i]
            
        if plot_type == 1:
            plotVals(s_next, N, points)
        else:
            for k in range(N):
                values[counter+1, k] = s_next[k]
    
        s_current = s_next * 1
     
    if plot_type == 2:
        heatMap(values)
        
    return values



def crankNicolson(N, C, end_time, endpoint=0, plot_type=1, equation=0, mu=0.3):
    points = getPoints(N)
    s_current = f(points, N, endpoint)
    s_next = s_current * 1
    
    h_x = L / (N - 1)
    h_t = C * h_x**2 / K
    
    # Heat / RD equation
    if equation == 0 or equation == 1:
        a = -C / 2
        b = (1 + C)
        c = -C / 2
        
    # Transport equation
    #a = -C * h_x / 2 * 50
    #b = 1
    #c = C * h_x / 2 * 50
    
    # Both
    #a = -C * h_x / 2 -C / 2
    #b = 1 + (1 + C)
    #c = C * h_x / 2 -C / 2


    # LU factorisation entries
    p = np.zeros(N)
    q = np.zeros(N-1)
    p[0] = 1
    q[0] = 0
    
    for i in range(1, N-1):
        p[i] = b - a * q[i-1]
        q[i] = c / p[i]
    
    p[-1] = 1
  
    #print(p)
    #print("")
    #print(q)
    
    d = np.zeros(N)
    w = np.zeros(N)

    
    #if plot_type == 1:
        #plotVals(s_next, N, points)
    
    values = np.zeros((1 + int(end_time // h_t), N))
    for k in range(N):
        values[0, k] = s_next[k]

    for counter in range(int(end_time // h_t)):      
        d[0] = s_current[0]
        d[-1] = s_current[-1]
        
        w[0] = s_current[0] / p[0]
        
        for i in range(1, N-1):            
            d[i] = s_current[i] + a * (2 * s_current[i] - s_current[i-1] - s_current[i+1])
            w[i] = (d[i] - a * w[i-1]) / p[i]
            
            
        for i in range(1, N-1):
            s_next[-i-1] = w[-i-1] - s_next[-i] * q[-i]
                
        # Reaction-diffusion  
        if equation == 1:
            #mu = 0.4
            #mu = 0.1
            s_next += 300 * h_t * s_current * (1 - s_current) * (s_current - mu)
            #s_next += 0.5 * (s_current * (1 - s_current) * (s_current - mu) + s_next * (1 - s_next) * (s_next - mu))
        
        
        #s_next *= s_current
        
        #if plot_type == 1:
            #pass
            #plotVals(s_next, N, points)
        #else:
        for k in range(N):
            values[counter+1, k] = s_next[k]

        s_current = s_next
    
    if plot_type == 1:
        plotVals(s_current, N, points)
            
    elif plot_type == 2:
        heatMap(values)
        
    return values



#--------------------------------------------------------------------------------------------------------------



def solverAnimation(N, C, end_time, method, analytic=0, endpoint=0, eq=0):
    global fig
    global ax
    global result_plt
    global solution_plt
    global meanval
    
    fig, ax = plt.subplots()
    
    results = numerical_method[method](N, C, end_time, endpoint, plot_type=0, equation=eq)

    x_vals = getPoints(N)
    
    result_plt = {}
    
    if method == 3:
        #for i in range(len(results)):
            #result_plt[i] = plt.imshow(results[i], interpolation="sinc", cmap="plasma", origin="lower",
                       #aspect="auto", extent=(0, N, 0, N), vmin=-1, vmax=1)
            
        #sns.heatmap(results[0], vmax=1, cbar=True, cmap="plasma")
        def update(frame):
            #for i in range(len(results)):
                #result_plt[i].set = plt.imshow(results[i], interpolation="sinc", cmap="plasma", origin="lower",
                           #aspect="auto", extent=(0, N, 0, N), vmin=-1, vmax=1)
            sns.heatmap(results[frame], vmax=1, cbar=False, cmap="plasma")
                
            return(ani)
    
    
    elif analytic == 1:
        solution = analyticSolution(N, C, end_time, plot_type=0)
        solution_plt = {}   
        
        for i in range(N-1):
            result_plt[i] = ax.plot(x_vals, results[i], zorder=3)[0]
            solution_plt[i] = ax.plot(x_vals, solution[i])[0]
            solution_plt[i].set_color("k")
        
        def update(frame):
            meanval =  np.array((min(max(np.sum(results[frame][:]) / N +1, 0), 1), 0, min(max(1-(np.sum(results[frame][:]) / N +1)/2, 0), 1))) 
            
            # This is here to prevent flashing images when the solution explodes
            if abs(np.sum(results[frame][:])) > N * 5:
                meanval = (0.5, 0, 0.5)
                
            for i in range(N-1):
                result_plt[i].set_color(meanval)
            
                solution_plt[i].set_xdata(x_vals[:])
                solution_plt[i].set_ydata(solution[frame][:])
                result_plt[i].set_xdata(x_vals[:])
                result_plt[i].set_ydata(results[frame][:]) 
                plt.ylim(0, max(solution[frame][N//2], results[frame][N//2]) * 1.1)

            return(ani)
    
    else:
        for i in range(N-1):
            result_plt[i] = ax.plot(x_vals, results[i])[0]

        def update(frame):
            meanval =  np.array((min(max(np.sum(results[frame][:]) / N +1, 0), 1), 0, min(max(1-(np.sum(results[frame][:]) / N +1)/2, 0), 1)))
            
            # This is here to prevent flashing images when the solution explodes
            if abs(np.sum(results[frame][:])) > N * 5:
                meanval = (0.5, 0, 0.5)
            
            for i in range(N-1):
                result_plt[i].set_color(meanval)
                
                result_plt[i].set_xdata(x_vals[:])
                result_plt[i].set_ydata(results[frame][:])   
            
            return(ani)
        
    h_x = L / (N - 1)
    h_t = C * h_x**2 / K
    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(end_time // h_t), interval=10)
    #ani.save('anim rde2.gif') 
    return(ani)

#--------------------------------------------------------------------------------------------------------------

numerical_method = (FTCS, BTCS, crankNicolson, FTCS2D)

#--------------------------------------------------------------------------------------------------------------

# FIGURE 1
#N_init = 5
#C_init = 0.5

#t = 0.9
#b = 0.15
#l = 0.18
#r = 0.99

#analyticSolution(N_init, C_init, 0.25, 1)
#FTCS(N_init, C_init, 0.25, 0, 1)
#plt.title("t = 0.25")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()
#analyticSolution(N_init, C_init, 0.5, 1)
#FTCS(N_init, C_init, 0.5, 0, 1)
#plt.title("t = 0.5")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()
#analyticSolution(N_init, C_init, 0.75, 1)
#FTCS(N_init, C_init, 0.75, 0, 1)
#plt.title("t = 0.75")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()
#analyticSolution(N_init, C_init, 1, 1)
#FTCS(N_init, C_init, 1, 0, 1)
#plt.title("t = 1")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()

#N_init = 15
#C_init = 8/15

#analyticSolution(N_init, C_init, 0.25, 1)
#FTCS(N_init, C_init, 0.25, 0, 1)
#plt.title("t = 0.25")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()
#analyticSolution(N_init, C_init, 0.5, 1)
#FTCS(N_init, C_init, 0.5, 0, 1)
#plt.title("t = 0.5")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()
#analyticSolution(N_init, C_init, 0.75, 1)
#FTCS(N_init, C_init, 0.75, 0, 1)
#plt.title("t = 0.75")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()
#analyticSolution(N_init, C_init, 1, 1)
#FTCS(N_init, C_init, 1, 0, 1)
#plt.title("t = 1")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)


#analyticSolution(N_init, C_init, end_time, 1)
#FTCS(N_init, C_init, end_time, 0, 1)



# FIGURE 2
#end_time = 1
#nstart = 5
#nend = 30
#C_dif = 0.01

#for nval in range(nstart, nend+1):
#    dif = (nval - nstart) / nend
#    findMultiError(nval, nval, 1, C_dif, 1, C_dif, end_time, 0, plot=0, col=((1 - dif), 0, dif))

#plt.xlim(C_dif, 1)
#plt.ylim(0.000000001, 1)
#plt.yscale("log")
#plt.tight_layout()
#plt.figure()


# FIGURE 3

#end_time = 1
#nstart = 5
#nend = 50

#t = 0.9
#b = 0.15
#l = 0.1
#r = 0.99


#C_init = 0.05
#analyticSolution(50, C_init, end_time, 1, scatter=False)

#for nval in range(nstart, nend+1):
#    dif = (nval - nstart) / nend
#    FTCS(nval, C_init, end_time, 0, 1, col=((1 - dif)**4, 0, (dif)**0.25))

#plt.ylim(-0.000001, 0.00008)
#plt.title("C = 0.05")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()

#C_init = 0.18
#analyticSolution(50, C_init, end_time, 1, scatter=False)

#for nval in range(nstart, nend+1):
#    dif = (nval - nstart) / nend
#    FTCS(nval, C_init, end_time, 0, 1, col=((1 - dif)**4, 0, (dif)**0.25))

#plt.ylim(-0.000001, 0.00008)
#plt.title("C = 0.18")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)
#plt.figure()

#C_init = 0.5
#analyticSolution(50, C_init, end_time, 1, scatter=False)

#for nval in range(nstart, nend+1):
#    dif = (nval - nstart) / nend
#    FTCS(nval, C_init, end_time, 0, 1, col=((1 - dif)**4, 0, (dif)**0.25))

#plt.ylim(-0.000001, 0.00008)    
#plt.title("C = 0.5")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)



# FIGURE 4
#end_time = 1
#nstart = 20
#nend = 50
#C_dif = 0.1
#Cend = 30

#for nval in range(nstart, nend+1):
#    dif = (nval - nstart) / (nend - nstart)
#    findMultiError(nval, nval, 1, C_dif, Cend, C_dif, end_time, 2, plot=0, col=((1 - dif), 0, dif))

#plt.xlim(C_dif, Cend)
#plt.ylim(0.0000000001, 1)
#plt.yscale("log")
#plt.tight_layout()


# FIGURE 5
#end_time = 1
#nstart = 100
#nend = 100
#Cstart = 1
#Cend = 300
#C_dif = 50

#for nval in range(nstart, nend+1):
#    dif = 1#(nval - nstart) / (nend - nstart)
#    findMultiError(nval, nval, 1, Cstart, Cend, C_dif, end_time, 2, plot=0, col=((1 - dif), 0, dif), order=2)

#plt.xlim(C_dif, Cend**2)
#plt.ylim(0, 0.000004)
#plt.yscale("linear")
#plt.subplots_adjust(top=0.9, bottom=0.18, left=0.075, right=0.975, wspace=0.2, hspace=0.2)




# FIGURE 6
#N_init = 200
#C_init = 1

#t = 0.9
#b = 0.18
#l = 0.18
#r = 0.935

#plt.rcParams.update({"font.size": 28})

#plt.figure()
#crankNicolson(N_init, C_init, 0.001, 0, 1, 1, mu=0.01)
#plt.title("t = 0.001",)
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.01, 0, 1, 1, mu=0.01)
#plt.title("t = 0.01")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.1, 0, 1, 1, mu=0.01)
#plt.title("t = 0.1")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.3, 0, 1, 1, mu=0.01)
#plt.title("t = 0.3")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.5, 0, 1, 1, mu=0.01)
#plt.title("t = 0.5")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.001, 0, 1, 1, mu=0.49)
#plt.title("t = 0.001")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.01, 0, 1, 1, mu=0.49)
#plt.title("t = 0.01")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.1, 0, 1, 1, mu=0.49)
#plt.title("t = 0.1")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.3, 0, 1, 1, mu=0.49)
#plt.title("t = 0.3")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#crankNicolson(N_init, C_init, 0.5, 0, 1, 1, mu=0.49)
#plt.title("t = 0.5")
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)


# ANIM 1

# Use x_val[i] = abs(sin(2 * pi * x_val[i] / L))**4 for function f.
# Use mu = 0.4, mu = 0.1.

#solverAnimation(N_init, C_init, end_time, method=2, analytic=0, eq=1)




#--------------------------------------------------------------------------------------------------------------





#analyticSolution(N_init, C_init, end_time, 1)
#FTCS(N_init, C_init, end_time, 0, 1)
#BTCS(N_init, C_init, end_time, 0, 2)
#crankNicolson(N_init, C_init, end_time, 0, 1)

#findError(N_init, C_init, end_time, method=2)
#findMultiError(N_init, N_init * 5, 1, C_init, 5, 0.1, end_time, 2, plot=2)
#FTCS2D(N_init, C_init, end_time, 0, 2)


#findMultiError(15, 15, 1, 0.01, 1, 0.01, 1, 2, plot=0)
#findMultiError(N_init, N_init * 5, 1, C_init, C_init, 0.01, end_time, 2, plot=1)

#compareMultiError(N_init, N_init*3, 4, 0.1, 0.5, 0.1, end_time, method1=0, method2=2)

#solverAnimation(N_init, C_init, end_time, method=2, analytic=0, eq=0)


