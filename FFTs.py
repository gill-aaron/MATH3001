# -*- coding: utf-8 -*-

# %matplotlib qt5

# FAST FOURIER TRANSFORM

import numpy as np
import scipy as sp
import seaborn as sns
from numpy import exp, sin, cos
from numpy.fft import fft, ifft, rfft, fft2, ifft2, rfft2, fftn, ifftn
from scipy.fft import fft as scfft
from math import pi, log
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from time import time
import random
from PIL import Image

#from Heat_Equation import getPoints

plt.rcParams.update({"font.size": 20})
plt.rcParams.update({"lines.linewidth": 2})

#--------------------------------------------------------------------------------------------------------------

# CONSTANTS

x_init = [1,2,-1,4]

x2_init = [0,2**(-0.5),1,2**(-0.5),0]
x3_init = [1,2,3,4,7,8,8,8,1,2,9,8,5,6,5,6]


y_init = np.ones(2**8, dtype = "complex_")
for i in range(len(y_init)):
    y_init[i] *= 0.1 * random.randrange(0, 10)# + random.randrange(0, 0)*1j
    
z_init = np.ones(len(y_init))#, dtype = "complex_")
for i in range(len(z_init)):
    z_init[i] *= sin(2 * pi * i // len(z_init))
    


l = 50
lim = np.arange(0, 1, 1/l)
signal = np.zeros(l)
v = 50
for i in range(v):
    signal += i * sin(5.12 * i * pi * lim) // v


# This superposes some number of sinusoids
f = 2**9
t = np.arange(0, 1, 1/f)
sinval = np.zeros(len(t))

# = int((f/2)**0.25)
#for w in range(0, f_root):
#    sinval += 3 / (w+1) * sin(pi * 2 * t * w**4)

for i in range(0, 8):
    sinval += (i+1) * sin(pi * 2 * t * (4 + 1 * i))

for i in range(22, 29):
    sinval += 6 * sin(pi * 4 * t * i)  
 
 

points = 100
heat = sin(pi * 1 * np.linspace(0, 1, points))
 
    
l = 31
heat2 = np.zeros(l)
for i in range(l):
    heat2[i] = sin(pi * i / (l-1) * 3)

heat2 = heat2 * 0 + 4 * np.arctan(exp(np.arange(0, 1, 1/l)))




#img = np.asarray(Image.open("planet test initial.png"))
img = np.asarray(Image.open("fingerprint test.png"))


#--------------------------------------------------------------------------------------------------------------

def plotVals(x, t, col="default", title=""):
    if title:
        plt.figure(figsize=(20, 5))
    else:
        plt.figure(figsize=(20, 3.5))
    
    if col == "default":
        plt.plot(t, x)
    else:
        plt.plot(t, x, col)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1)
    
    if title:
        plt.title(title)
        #plt.subplots_adjust(top=0.9, bottom=0.22, left=0.06, right=0.98, wspace=0.2, hspace=0.2)
        plt.subplots_adjust(top=0.89, bottom=0.21, left=0.08, right=0.975, wspace=0.2, hspace=0.2)
    else:
        plt.subplots_adjust(top=0.99, bottom=0.22, left=0.06, right=0.98, wspace=0.2, hspace=0.2)
    

def fftPlot(x, nq=True, col="r", title="", amp=True):
    if title:
        plt.figure(figsize=(20, 5))
    else:
        plt.figure(figsize=(20, 3.5))
    
    #Plotting is adjusted for nyquist frequency
    if nq == True:
        cutoff = (len(x)+1)//2
        
    else:
        cutoff = len(x)

    if amp:
        x_plot = x[:cutoff] / cutoff
        for i in range(cutoff):
            plt.plot((i,i),(0, (np.linalg.norm(x_plot[i]))), c=col)
    else:
        x_plot = np.array(x[:cutoff] / cutoff, dtype="complex_")
        for i in range(cutoff):
            if x_plot[i].real != 0:
                plt.plot((i,i),(0, np.arctan(x_plot[i].imag / x_plot[i].real)), c=col)
            
    
    plt.plot((0,i),(0, 0), c="r")
    plt.xlabel("Frequency")
    if amp:
        plt.ylabel("Amplitude")
    else:
        plt.ylabel("Phase")
    plt.xlim(0, cutoff)
    plt.ylim(bottom=0)
    if title:
        plt.title(title)
        plt.subplots_adjust(top=0.89, bottom=0.21, left=0.08, right=0.975, wspace=0.2, hspace=0.2)
    else:
        plt.subplots_adjust(top=0.99, bottom=0.22, left=0.06, right=0.98, wspace=0.2, hspace=0.2)


def timeComparison(t_range, a=True, b=True, c=True):
    attempts = 10
    limit = 10
    plt.figure(figsize=(20, 3.5))
    
    duration_a = 0
    duration_b = 0
    duration_c = 0
    duration_d = 0
    
    a_prev = 0
    b_prev = 0
    c_prev = 0
    d_prev = 0
    
    for t in range(t_range+1): 
        
        values = np.random.rand(2**t)

        if a_prev < limit and a:
            time_a = time()
            for _ in range(attempts):
                #ta = time()
                discreteFourierTransform(values, 0, 0)
                #plt.scatter(2**t, time() - ta, c="r")
            duration_a = (time() - time_a) / attempts
            if t >= 1:
                plt.plot((2**(t-1), 2**t), (a_prev, duration_a), c="r")
            
        
        if b_prev < limit and b:
            time_b = time()
            for _ in range(attempts):
                #tb = time()
                tukeyCooleyFFT(values, 0, 0)
                #plt.scatter(2**t, time() - tb, c="b")
            duration_b = (time() - time_b) / attempts
            if t >= 1:
                plt.plot((2**(t-1), 2**t), (b_prev, duration_b), c="b")
    
        if c_prev < limit and c:
            time_c = time()
            for _ in range(attempts):
                fft(values)
            duration_c = (time() - time_c) / attempts
            plt.plot((2**(t-1), 2**t), (c_prev, duration_c), c="c")
        
        if d_prev < limit and c:
            time_d = time()
            for _ in range(attempts):
                scfft(values)
            duration_d = (time() - time_d) / attempts
            plt.plot((2**(t-1), 2**t), (d_prev, duration_d), c="m")
            
        print(t)
        print("")
        #if t >= 1:
            #plt.plot((2**(t-1), 2**t), (a_prev, duration_a), c="r")
            #plt.plot((2**(t-1), 2**t), (b_prev, duration_b), c="b")
            #plt.plot((2**(t-1), 2**t), (c_prev, duration_c), c="c")
            
            
            #plt.plot((2**(t-1), 2**t), (2**t - 2, 2**(t+1) - 2), "r:")
            #if t >=2:
                #plt.plot((2**(t-1), 2**t), ((t-1) * log(t-1, 2), t * log(t, 2)), "k:")

            
        a_prev = duration_a
        b_prev = duration_b
        c_prev = duration_c
        d_prev = duration_d
    
    print(f"DFT: {duration_a}")
    print(f"FFT: {duration_b}")
    print(f"NP: {duration_c}")
    print(f"SP: {duration_d}")
    plt.xlim(100, 1048576)
    plt.ylim(0.01, 10)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Time / s")
    plt.subplots_adjust(top=0.95, bottom=0.22, left=0.06, right=0.98, wspace=0.2, hspace=0.2)




def filterFFT(sig, t):
    plt.rcParams.update({"font.size": 32})
    plotVals(sig, t, title="Initial Signal")
    freq_rep = fft(sig)
    fftPlot(freq_rep, title="Initial Frequency")
    freq_upper = freq_rep * 1
    freq_lower = freq_rep * 1
    
    wavecount = 0
    freq_alt = np.sqrt(freq_rep.real**2 + freq_rep.imag**2)

    for i in range(freq_rep.size):
        if i > 50:
            freq_lower[i] *= 0.1
        else:
            freq_upper[i] *= 0.1
        
        if freq_alt[i] >= 0.01:
            wavecount += 1
    
    freq_mean = np.sum(freq_alt) / wavecount
    for i in range(freq_rep.size):      
        if freq_alt[i] >= freq_mean / 2:
            freq_alt[i] = freq_mean
        else:
            freq_alt[i] = 0
    
    
    plotVals(ifft(freq_lower), t, title="Low-Pass Signal")
    fftPlot(freq_lower, title="Low-Pass Frequency")
    plotVals(ifft(freq_upper), t, title="High-Pass Signal")
    fftPlot(freq_upper, title="High-Pass Frequency")
    plotVals(ifft(freq_alt), t, title="Restoration Signal")
    fftPlot(freq_alt, title="Restoration Frequency")
    

    
 
def rgb(img):
    #First dimension: rows (375)
    #Second dimension: columns (500)
    #third dimension: colours (3), opacity (1)
    
    #img_r = np.array(img[:,:,0]) * 1
    #img_g = np.array(img[:,:,1]) * 1
    #img_b = np.array(img[:,:,2]) * 1
    
    img_r = np.array(img) * 1
    img_g = np.array(img) * 1
    img_b = np.array(img) * 1

    img_r[:,:,1] = 0
    img_r[:,:,2] = 0
    img_g[:,:,0] = 0
    img_g[:,:,2] = 0
    img_b[:,:,0] = 0
    img_b[:,:,1] = 0
    
    plt.figure()
    plt.imshow(img)

    #plt.figure()
    #plt.imshow(img_r)
    
    #plt.figure(figsize=(5, 5.5))
    #imfft_r = fft2(1 - img_r[:,:,0])
    imfft_r = fft2(img_r[:,:,0])
    plotfft = imfft_r.real**2 + imfft_r.imag**2
    #sns.heatmap(plotfft, norm=LogNorm(vmin=0.01, vmax=10**(16)), xticklabels=False, yticklabels=False)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.01, right=0.95, wspace=0.2, hspace=0.2)
    
    #plt.figure()
    #plt.imshow(img_g)
    #plt.figure()
    #imfft_g = fft2(1 - img_g[:,:,1])
    #plotfft = imfft_g.real**2 + imfft_g.imag**2
    imfft_g = fft2(img_g[:,:,1])
    plotfft = imfft_g.real**2 + imfft_g.imag**2
    #sns.heatmap(plotfft, norm=LogNorm(vmin=0.01, vmax=10**(16)), xticklabels=False, yticklabels=False)
    #plt.subplots_adjust(top=0.96, bottom=0.05, left=0.01, right=0.98, wspace=0.2, hspace=0.2)
    
    #plt.figure()
    #plt.imshow(img_b)
    #plt.figure()
    #imfft_b = fft2(1 - img_b[:,:,2])
    #plotfft = imfft_b.real**2 + imfft_b.imag**2
    imfft_b = fft2(img_b[:,:,2])
    plotfft = imfft_b.real**2 + imfft_b.imag**2
    #sns.heatmap(plotfft, norm=LogNorm(vmin=0.01, vmax=10**(16)), xticklabels=False, yticklabels=False)
    #plt.subplots_adjust(top=0.96, bottom=0.05, left=0.01, right=0.98, wspace=0.2, hspace=0.2)
    
    
    final = np.array(img) * 1
    
    result = np.array(filterFFTImage(imfft_r, imfft_g, imfft_b, 1))

    
    final[:,:,0] = np.array(result[0].real)
    final[:,:,1] = np.array(result[1].real)
    final[:,:,2] = np.array(result[2].real)
    
    init = img[:-1,3:] * 1
    final = final[:-1,3:] * 1
    
    
    #plt.figure()
    #plt.imshow(img)
    plt.figure()
    plt.imshow(final[:-1,3:])
    plt.figure(figsize=(5, 5.5))
    v = fft2(final[:,:,0])
    v = v.real**2 + v.imag**2
    sns.heatmap(v, norm=LogNorm(vmin=0.01, vmax=10**(16)), xticklabels=False, yticklabels=False)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.01, right=0.95, wspace=0.2, hspace=0.2)
    
    #plt.imsave("ifingerprint1.png", init)
    #plt.imsave("ifingerprint6.png", final)
    
    




def filterFFTImage(fimr, fimg, fimb, filter_type):
    fftr = np.array(fimr, dtype="complex_") * 1
    fftg = np.array(fimg, dtype="complex_") * 1
    fftb = np.array(fimb, dtype="complex_") * 1
    n = len(fftr)
    m = len(fftr[0])
    #m = 10**4
    
    #limit = 10**5

    
    for row in range(n):
        for col in range(m):  
            #fftr[row, col] = np.round(fftr[row, col] / m, 0) * m
            #fftg[row, col] = np.round(fftg[row, col] / m, 0) * m
            #fftb[row, col] = np.round(fftb[row, col] / m, 0) * m
            
            ampr = fftr[row, col].real**2 + fftr[row, col].imag**2
            ampg = fftg[row, col].real**2 + fftg[row, col].imag**2
            ampb = fftb[row, col].real**2 + fftb[row, col].imag**2
            
            

            #if ampr <= limit:
                #fftr[row, col] = 0
                
            #if ampg <= limit:
                #fftg[row, col] = 0
                    
            #if ampb <= limit:
                #fftb[row, col] = 0      
            
            
            if filter_type == 0:
                #Low pass blur
                # none, 100, 25, 10, 5, 3
                l = 25
                if col >= l and row >= l:
                    fftr[row, col] = 0
                    fftb[row, col] = 0
                    fftg[row, col] = 0

            elif filter_type == 1:
                # none, 5, 50, 250, 265, 270
                # High pass pattern isolation
                l = 250
                if col <= l and row <= l:
                    fftr[row, col] *= 0
                    fftb[row, col] *= 0
                    fftg[row, col] *= 0

            elif filter_type == 2:   
                # Amplitude filter
                if ampr >= 5 * 10**13:
                    fftr[row, col] = 0
                    fftb[row, col] = 0
                    fftg[row, col] = 0
            
            #elif filter_type == 3:
                #l = 500
                # Background removal
                #if col <= l and row <= l:# and (ampr >= 6.2015 * 10**13 and ampr <= 6.202 * 10**13):
                    #fftr[row, col] *= 0.9
                    #fftb[row, col] *= 0.9
                    #fftg[row, col] *= 0.9
                    
            #elif filter_type == 4:   
                # Background colour change
                #if ampr >= 5 * 10**13:
                    #fftr[row, col] *= 0.75
                    #fftb[row, col] *= 0.75
                    #fftg[row, col] = 0
            
                #l = 20
                #if col <= l and row <= l:
                    #fftr[row, col] *= 0.5
                    #fftb[row, col] *= 0.5
                    #fftg[row, col] *= 0.5
                
                
    
     
    if filter_type == 0 or filter_type >= 1:
        return (ifft2(fftr), ifft2(fftg), ifft2(fftb))
    
    else:
        return (1 - ifft2(fftr), 1 - ifft2(fftg), 1 - ifft2(fftb))
    
    
    
    
    
    
    #fft_map = np.array(img[:,:,0:3])

    #for row in range(len(fft_map)):
    #    for col in range(len(fft_map[0])):
    #        fft_map[row, col] = [image[row,col,0], image[row,col,1], image[row,col,2]]
            
    
    #init = fft_map
    #col_map = fft(fft_map)
    #new_map = np.array(img[:,:,0:3] * 0, dtype="complex_")
    
    #sq = 16
    
    
    # col_map[0] = fft(fft_map[0])

    #for row in range(len(col_map)):
    #    for col in range(len(col_map[0])):
    #        for val in range(len(col_map[0,0])):             
    #            if row % sq == 0 and col % sq == 0:
    #                avcol = 0
    #                for x in range(sq):
    #                    for y in range(sq):
    #                        if abs(col_map[row+x, col+y, val].real) >= 5 and abs(col_map[row+x, col+y, val].imag) < 128:
    #                            avcol += col_map[row+x, col+y, val]
    #                avcol /= sq**2
    #                avcol = avcol.real + avcol.imag * 0.75j
    #                avcol = np.round(avcol / 20, 0) * 20
    #                
    #            new_map[row, col, val] = avcol

    #plt.figure(figsize=(7, 7))
    #plt.imshow(col_map.real.astype(np.uint8))
    #plt.imsave("planet test r1.png", col_map.real.astype(np.uint8))
    #plt.figure(figsize=(7, 7))
    #plt.imshow(new_map[::sq, ::sq].real.astype(np.uint8)) 
    #plt.imsave("planet test r2.png", new_map[::sq, ::sq].real.astype(np.uint8))
    #plt.figure(figsize=(7, 7))
    #plt.imshow(col_map.imag.astype(np.uint8))
    #plt.imsave("planet test i1.png", col_map.imag.astype(np.uint8))      
    #plt.figure(figsize=(7, 7))
    #plt.imshow(new_map[::sq, ::sq].imag.astype(np.uint8)) 
    #plt.imsave("planet test i2.png", new_map[::sq, ::sq].imag.astype(np.uint8))  

    
    #output = ifft(new_map[::sq, ::sq]).real.astype(np.uint8)
    #plt.figure(figsize=(7, 7))
    #plt.imshow(init)
    #plt.imsave("rosette.png", init)
    #plt.figure(figsize=(7, 7))
    #plt.imshow(output)
    #plt.imsave("planet test c.png", output)
    
    




#--------------------------------------------------------------------------------------------------------------

def discreteFourierTransform(x_m, inverse=0, plot=1):
    n = len(x_m)
    X_k = np.zeros(n, dtype = "complex_")
    coeff = (-1)**inverse * -2j * pi / n

    for k in range(n):
        for m in range(n):
            X_k[k] += x_m[m] * exp(coeff * m * k)

    if inverse == 1:
        X_k = X_k / n
    
    if plot == 1:
        plotVals(X_k)
        
    return(X_k)


def FFTLoop(x, coeff):
    n = len(x)
    if n == 1:
        return x
    
    else:
        W = exp(coeff * np.arange(n) / n)
        even_component = FFTLoop(x[::2], coeff)
        odd_component = FFTLoop(x[1::2], coeff)

        return np.concatenate([even_component + W[:n//2] * odd_component, even_component + W[n//2:] * odd_component])


def tukeyCooleyFFT(x_m, inverse=0, plot=1):
    # FFT algorithm based on algorithm from (Kong, 2020).
    coeff = -2j * pi * (-1)**inverse
    
    if inverse == 0:
        X_k = FFTLoop(x_m, coeff)
    else:
        X_k = FFTLoop(x_m, coeff) / len(x_m)

    if plot == 1:
        fftPlot(X_k)
        
    return(X_k)


#--------------------------------------------------------------------------------------------------------------

def spectralMethod(init, t, samples=1, plot_type=1, eq=0):
    f_m = np.array(init, dtype = "complex_")#
    f_prev = f_m * 1
    n = len(f_m)

    G_k = np.arange(0, n, 1) / (n-1)

    x = np.linspace(0, 1, n)
    h_x = 1 / (n-1)
    k = np.linspace(1, n-1, n)

    values = []
    values2 = []
    
    #deriv = np.array([], dtype="complex_")
    
    if plot_type == 1:
        loops = int(max(1000 * t, 1))
        h_t = t / loops
    else:
        loops = samples
        h_t = t / (samples-1)
        
    if eq == 1:
        #f_m += 10j
        beta = 1 #0.1
        mu = 0.03125 #0**2
        
        beta *= 4 * pi / n
        mu *= 64 * pi**3 / n**3

        

        

    for l in range(loops):
        if eq == 0:
            F_k = fft(f_m)
            #alpha = 2j * pi * k / n / h_x
            F_k = F_k * exp(-k**2 * h_t)
            f_m = np.array(ifft(F_k), dtype="complex_")
        
        elif eq == 1:
            #F_k = F_k * (exp(-k**3 * h_t) +exp(-k * h_t) * (1 + F_k))
            #F_k = F_k * cos(k * t)
            #for i in range(len(k)):
                #if k[i] != 0:
                    #F_k[i] = F_k[i] * (1 / (exp(k[i]**2 * t) * -k[i]**2) + (1 + 1 / (k[i]**2)))

        
            #c = 1
            #beta = 1
            #mu = 1
            #xi = x - c * t
            
            #f_m = 3 * c / beta / np.cosh(0.5 * np.sqrt(c / mu) * (xi))**2 * f_m
                
                
                #deriv = (-1j * c1 * f_m * ifft(exp(-k * h_t)) + 1j * c2 * ifft(exp(-k**3 * h_t)))
                #f_m = f_m + h_t / 2 * deriv
                #deriv = (-1j * c1 * f_m * ifft(exp(-k * h_t)) + 1j * c2 * ifft(exp(-k**3 * h_t)))
                #f_m = f_m + h_t / 2 * deriv
                
                ##F_k = fft(f_m)
                ##deriv = -1j * beta * f_m * np.array(ifft(k * F_k), dtype="complex_") + 1j * mu * np.array(ifft(k**3 * F_k), dtype="complex_")
                ##f_m = f_m + h_t / 2 * deriv
                ##F_k = fft(f_m)
                ##deriv = -1j * beta * f_m * np.array(ifft(k * F_k), dtype="complex_") + 1j * mu * np.array(ifft(k**3 * F_k), dtype="complex_")
                ##f_m = f_m + h_t / 2 * deriv
                
                F_k = fft(f_m)
                deriv = -1j * beta * f_m * np.array(ifft(k * F_k), dtype="complex_") + 1j * mu * np.array(ifft(k**3 * F_k), dtype="complex_")
                f_leap = f_m + h_t / 2 * deriv
                F_k = fft(f_leap)
                deriv = -1j * beta * f_leap * np.array(ifft(k * F_k), dtype="complex_") + 1j * mu * np.array(ifft(k**3 * F_k), dtype="complex_")
                f_m = f_m + h_t * deriv
        

        # This sets the endpoint to 0
        #f_m -= f_m[0]

        #G_k = ifft(F_k**2)
        values.append(f_m)
        values2.append(G_k)
    
    

        
    #f_m = ifft(F_k)
    
    
    #print(np.max(ans))
    #print(np.max(f_m))
    
    if plot_type == 1:
        #plt.plot(x, init, c="b")
        plt.plot(x, f_m.real, c="r")
        
    
    #plt.plot(fft(ans), c="k")
    #plt.plot(F_k)
    
    #other = 0
    #for m in range(n):
        #other += init[m] * exp(-pi**2 * t - 2 * pi * 1j * m * np.arange(n) / n)
    #plt.plot(other, c="g")
    #plt.plot(fft(init * exp(-pi**2 * t)), c="r")


    return (values, values2)
    
    
    
    
#--------------------------------------------------------------------------------------------------------------



def fftAnimation(init, t, samples, analytic=0, eq=0):
    global fig
    global ax
    global result_plt
    global solution_plt
    global meanval
    

    fig, ax = plt.subplots()
    
    outcome = spectralMethod(init, t, samples, 0, eq=eq)
    results = outcome[0]
    results2 = outcome[1]
    N = len(results)
    #x_vals = np.arange(N) / (N-1)
    
    result_plt = {}
    

    if analytic == 0:
        for i in range(N-1):
            result_plt[i] = ax.plot(results2[i], results[i], c="r")[0]
            #result_plt[i] = ax.plot(x_vals, results[i])[0]

        def update(frame):
            #meanval =  np.array((min(max(np.sum(results[frame][:]) / N +1, 0), 1), 0, min(max(1-(np.sum(results[frame][:]) / N +1)/2, 0), 1)))
            
            # This is here to prevent flashing images when the solution explodes
            #if abs(np.sum(results[frame][:])) > N * 5:
                #meanval = (0.5, 0, 0.5)
            
            for i in range(N-1):
                #result_plt[i].set_color(meanval)
                
                #result_plt[i].set_xdata(x_vals[:])
                result_plt[i].set_xdata(results2[frame][:])
                result_plt[i].set_ydata(results[frame][:])   
            
            return(ani)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=samples, interval=1)
    #plt.ylim(-1.25, 1.25)
    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    #plt.ylim(-0.05, 2.1)
    plt.ylim(-5, 20)
    #ani.save('anim kdv-.gif') 
    return(ani)





#--------------------------------------------------------------------------------------------------------------

# FIGURE 1

#f = 2**9
#t = np.arange(0, 1, 1/f)
#sinval = np.zeros(len(t))

#for i in range(0, 8):
#    sinval += (i+1) * sin(pi * 2 * t * (4 + 8 * i))

#for i in range(37, 46):
#    sinval += 6 * sin(pi * 4 * t * i) 


#plotVals(sinval, t)
#fftPlot(fft(sinval), nq=False)


# FIGURE 2

#f = 2**9
#t = np.arange(0, 1, 1/f)
#sinval = np.zeros(len(t), dtype="complex_")

#for i in range(1, 100):
#    sinval += exp(1.1j * i * t)


#plt.figure(figsize=(20, 3.5))

#plt.plot(t, sinval.real, c="b", label="Real signal component")
#plt.plot(t, sinval.imag, c="r", label="Imaginary signal component")
#plt.xlabel("Time")
#plt.ylabel("Amplitude")
#plt.xlim(0, 1)
#plt.legend()
#plt.subplots_adjust(top=0.99, bottom=0.22, left=0.06, right=0.98, wspace=0.2, hspace=0.2)

#plt.figure(figsize=(20, 3.5))
#plot = fft(sinval).real

#for i in range(f):
#    plt.plot((i,i),(0, np.linalg.norm(fft(sinval)[i])), c="m")

#plt.plot((0,i),(0, 0), c="m")
#plt.xlabel("Frequency")
#plt.ylabel("Amplitude")
#plt.xlim(0, f)
#plt.ylim(bottom=0)
#plt.subplots_adjust(top=0.99, bottom=0.22, left=0.06, right=0.98, wspace=0.2, hspace=0.2)



# FIGURE 3

#np.random.seed(0)
# Limit this to 27 or less to avoid pc freezes
#timeComparison(27)


# FIGURE 4

#random.seed(1)
#f = 2**9
#t = np.arange(0, 1, 1/f)
#sinval2 = np.zeros(len(t))

#for i in range(0, 100):
#    val = random.randint(0, f // 2)
#    val2 = random.randint(1, 1 + (f // 2) - val)**2 / 10000
#    sinval2 += val2 * sin(t * val * 2 * pi)

#filterFFT(sinval2, t)




# FIGURE 5

#random.seed(2)
#f = 2**9
#label = 100
#t = np.arange(0, 1, 1/f)
#sinval3 = np.zeros(len(t))

#sinval_demo = np.zeros((len(t), len(t)))
#sinval2d = np.zeros((len(t), len(t)))

#for i in range(f):
    #for j in range(0, 11):
        #val = random.randint(0, f // 2)
        #sinval3 += 0.01 * j * sin(t * val * 2 * pi)
    #sinval2d[i] = sinval3
    #sinval3 = 0
    

#plt.figure()
#sns.heatmap(sinval2d, cmap="Greys", xticklabels=label, yticklabels=label)
#plt.subplots_adjust(top=0.989, bottom=0.1, left=0.06, right=0.95, wspace=0.2, hspace=0.2)


#plt.figure()
#fsinval2d = rfft2(sinval2d)
#fsinval2d = np.sqrt(fsinval2d.real**2 + fsinval2d.imag**2)
#sns.heatmap(fsinval2d, cmap="plasma", xticklabels=label, yticklabels=label)
#plt.subplots_adjust(top=0.989, bottom=0.1, left=0.06, right=0.95, wspace=0.2, hspace=0.2)


# FIGURE 6
#rgb(img)


# FIGURE 7
#points = 100
#heat = 1 - 500 * (0.5 - np.linspace(0, 1, points))**2
#heat = 1 - 500 * (0.5 * np.linspace(0, 1, points))**2

#for i in range(len(heat)):
#    heat[i] = max(heat[i], 0)
    
#t = 0.9
#b = 0.18
#l = 0.18
#r = 0.935

#plt.rcParams.update({"font.size": 28})

#plt.figure()
#spectralMethod(heat, 0)
#plt.title("t = 0",)
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#spectralMethod(heat, 0.001)
#plt.title("t = 0.001",)
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#spectralMethod(heat, 0.01)
#plt.title("t = 0.01",)
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#spectralMethod(heat, 0.1)
#plt.title("t = 0.1",)
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)

#plt.figure()
#spectralMethod(heat, 1)
#plt.title("t = 1",)
#plt.ylabel("Solution")
#plt.xlabel("x")
#plt.xlim(0, 1)
#plt.ylim(-0.1, 1.1)
#plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)




# FIGURE 8

#points = 1000
#np.set_printoptions(precision=32)

#wave = 1 + sin(np.linspace(0, 1 + 1/points, points) * 10 * pi)
    
#t = 0.9
#b = 0.18
#l = 0.18
#r = 0.935

#plt.rcParams.update({"font.size": 28})

#times = np.linspace(0, 1, 5)

#for T in times:
#    plt.figure()
#    spectralMethod(wave, T * 0.1, eq=1)
#    plt.title(f"t = {np.round(T, 2)}",)
#    plt.ylabel("Solution")
#    plt.xlabel("x")
#    plt.xlim(0, 1)
#    plt.ylim(-0.05, 2.1)
#    plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)


#wave = 1 - (0.5 - np.linspace(0, 1 + 1/points, points))**2 * 3
#wave = (1 + sin(np.linspace(0, 3 * pi + 1/points, points))) / 2
#wave += 1 / np.cosh(np.linspace(0, 3 * pi + 1/points, points))


#for T in times:
#    plt.figure()
#    spectralMethod(wave, T * 0.1, eq=1)
#    plt.title(f"t = {np.round(T, 2)}",)
#    plt.ylabel("Solution")
#    plt.xlabel("x")
#    plt.xlim(0, 1)
#    plt.ylim(-0.05, 2.1)
 #   plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)


#wave =  abs(sin(np.linspace(0, 3 * pi, points)) * 10)

#for T in times:
#    plt.figure()
#    spectralMethod(wave, T * 0.1, eq=1)
#    plt.title(f"t = {np.round(T, 2)}",)
#    plt.ylabel("Solution")
#    plt.xlabel("x")
#    plt.xlim(0, 1)
#    plt.ylim(-0.35, 12.5)
#    plt.subplots_adjust(top=t, bottom=b, left=l, right=r, wspace=0.2, hspace=0.2)


# ANIM 3

#points = 1000

#wave = 1 + sin(np.linspace(0, 1 + 1/points, points) * 10 * pi)
#wave = 1 - (0.5 - np.linspace(0, 1 + 1/points, points))**2 * 3
#wave =  abs(sin(np.linspace(0, 3 * pi, points)) * 10)

#fftAnimation(wave, 1, 100, eq=1)
#fftAnimation(wave, 0.8759374655710950308143, 50, eq=1)


#--------------------------------------------------------------------------------------------------------------



#discreteFourierTransform(x_init, 0)
#tukeyCooleyFFT(x_init)
#fftPlot(fft(fixed_signal))



#filterFFT(sinval, t)
#filterFFTImage(img, t)



#spectralMethod(heat, 0.01)
#fftAnimation(heat, 0.1, 1000)
