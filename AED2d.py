#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:27:10 2022

@author: szechingaudreyfung
"""

import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.constants as constants
import numba
#get_ipython().run_line_magic('matplotlib', 'auto')

plt.rcParams.update({'font.size': 17}) # keep those graph fonts readable!
plt.rcParams['figure.dpi'] = 120

"""
For stability reason, I impose two criteria:
1) dt/dx = 0.5
2) dt2 = 0.25
Therefore dx = sqrt(dt2)/0.25 = 1; dt = 0.5
"""


@numba.jit(nopython=True)
def hat2phy(Ex, Ey, Ez, Bx, By, Bz, axion):
    Ex_phy = (Ex + kappa*axion*Bx)/(1+kappa**2*axion**2)
    Ey_phy = (Ey + kappa*axion*By)/(1+kappa**2*axion**2)
    Ez_phy = (Ez + kappa*axion*Bz)/(1+kappa**2*axion**2)
    Bx_phy = (Bx - kappa*axion*Ex)/(1+kappa**2*axion**2)
    By_phy = (By - kappa*axion*Ey)/(1+kappa**2*axion**2)
    Bz_phy = (Bz - kappa*axion*Ez)/(1+kappa**2*axion**2)
    
    return Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy
@numba.jit(nopython=True)
def phy2hat(Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy, axion):
    Ex = Ex_phy - kappa*axion*Bx_phy
    Ey = Ey_phy - kappa*axion*By_phy
    Ez = Ez_phy - kappa*axion*Bz_phy
    Bx = Bx_phy + kappa*axion*Ex_phy
    By = By_phy + kappa*axion*Ey_phy
    Bz = Bz_phy + kappa*axion*Ez_phy
    
    return Ex, Ey, Ez, Bx, By, Bz

@numba.jit(nopython=True)
def Eupdate2d(Ex, Ey, Ez, Bx, By, Bz):
    for j in range(1,kmax-1):
        for i in range(1,kmax-1):
            Ex[j,i] = Ex[j,i] + 0.5*(Bz[j,i] - Bz[j-1,i])
            Ey[j,i] = Ey[j,i] + 0.5*(Bz[j-1,i] - Bz[j,i])
            Ez[j,i] = Ez[j,i] + 0.5*(By[j,i]-By[j,i-1]+Bx[j-1,i]-Bx[j,i])
    return Ex, Ey, Ez

@numba.jit(nopython=True)
def Bupdate2d(Ex, Ey, Ez, Bx, By, Bz):
    for j in range(0,kmax-1):
        for i in range(0,kmax-1):
            Bx[j,i] = Bx[j,i] + 0.5*(Ez[j,i]-Ez[j+1,i])
            By[j,i] = By[j,i] + 0.5*(Ez[j,i+1]-Ez[j,i])
            Bz[j,i] = Bz[j,i] + 0.5*(Ex[j+1,i] - Ex[j,i]+Ey[j,i]-Ey[j,i+1])
    return Bx, By, Bz

@numba.jit(nopython=True)
def Aupdate2d(Ex, Ey, Ez, Bx, By, Bz, axion, axion_past, Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy):
    for j in range(1, kmax-1):
        for i in range(1,kmax-1):
            axion[j,i] = 2*axion[j,i] - axion_past[j,i] + 0.25*(axion[j,i+1]+axion[j+1,i]-4*axion[j,i]\
            +axion[j,i-1]+axion[j-1,i]) - dt2*kappa* (Ex_phy[j,i]*Bx_phy[j,i] + Ey_phy[j,i]*By_phy[j,i]\
            +Ez_phy[j,i]*Bz_phy[j,i]) - dt2*m**2 * axion[j,i]
        
    return axion


def init(xmax):
    plt.xlim((0, xmax-1))
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_z (eV^2)$')
    plt.show()


def graph(t, E):
    plt.clf()
    ax = fig.add_axes([.25, .25, .6, .6])
    
    img = ax.contourf(E)
    #draw_circle = plt.Circle((jsource, isource), int(sourceRadius), fill=False)
    #ax.add_artist(draw_circle)
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$E_{phy,z}$ (eV^2)')
    ax.set_title('frame time{}'.format(t))
    plt.show()
    plt.pause(0.01)
    
    
#%%

# control
"If plot1d = True, a slice in the y direction will be plot in 1D, otherwise, a 2D densit plot will be plotted"
plot1d = True
source = 'plane waves' # 'pulse'

# axion params
m = 1
kappa =1 

# stability
dt2 = 0.25

# initialization
kmax = 1000
Ex = Ey = Ez = np.zeros([kmax,kmax])
Ex_phy = Ey_phy = Ez_phy = np.zeros([kmax,kmax])
Bx = By = Bz = np.zeros([kmax,kmax])
Bx_phy = By_phy = Bz_phy = np.zeros([kmax,kmax])
axion = np.zeros([kmax,kmax])

# source
isource = int(kmax/2)
jsource = int(kmax/2)
bsource = isource +200
nsteps = 1000



if source == 'pulse':
    # stability requirements
    dx = 20e-9 # ~ nm grid size
    dt_si = dx*0.5/constants.c # ~3.3e-17 s in one grid
    
    # pulse 
    dt_fs = dt_si/constants.femto 
    spread = 2/dt_fs # num of steps in 2 fs ~ 60 grids
    t0 = spread*6 
    freq_in = 2*math.pi* 200* constants.tera 
    w_scale = freq_in*dt_si # 0.042 rad per grid
    
    # set time in grid unit
    
    @numba.jit(nopython=True)
    def get_source(t):
        source = -np.exp(-0.5*(t-t0)**2/spread**2)*np.cos(t*w_scale)
        return source
    
if source == 'plane waves':
    # E source frequency (Energy)
    # Energies of photon ~ 3keV from the Sun
    hbar= 6.582e-16 # eVs
    freq = 3e3/hbar # ~ 4.6e18 Hz
    wavelength = constants.c/freq # 6.58e-11m
    # for 20 grids in one wave:
    ddx = wavelength/400 # ~1e-13 m; sample 400 pts in one wavelength
    ddx_natural = ddx/(hbar*constants.c) # ~1e-6 eV^-1
    
    # Stability: since dx*0.5 > dt
    dt = ddx_natural*0.5 # 8.35e-6 eV^-1
    dtdx=0.5
    dtdx2=0.25
    dt2 = dt**2 # 6.97e-11 eV^-1
    dt_si = dt*hbar # 5.5e-21
    
    angFreq = 2*math.pi*freq # ~2.8e19 rad/s
    w = angFreq*dt_si # 0.157 rad per grid
    # E source (lumininosities)
    # Erms ~ mu_0 * c* S_avg where S is the poynting vector
    
    Savg = 1.4
    Erms = constants.mu_0 * constants.c * Savg # ~527 (SI)
    elemC = 5.2909e-19
    J2eV = 6.242e18
    E0_natural = Erms * elemC * constants.c**2 *  J2eV*hbar # 103041 eV^2
    
    scale2 = 200000 # eV^2
    scale = scale2**0.5
    E0 = E0_natural/scale2 # say 3000 eV^2 = 0.5 grid units
    
    # B source
    Bext_si = 9 #Tesla
    Bext_natural = Bext_si * elemC * constants.c**2 * J2eV * hbar # eV^2
    Bext = Bext_natural/scale2
    #Bext =  0 # amplitude of B_external
    
    def get_source(t):
        source = E0*np.sin(w*t)
        return source



#%%

"Main Code"

cycle = 100
if plot1d == False:
    plt.clf()
    fig = plt.figure(figsize=(8,6))

if plot1d == True:
    plt.clf()
    plt.close()
    cycle = 100
    lw=2
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    xrange = np.linspace(0,kmax, kmax)
    [im] = ax.plot(xrange,Ez[int(kmax/2),:],linewidth=lw)
    [im2] = ax.plot(xrange,By[int(kmax/2),:],linewidth=lw)
    [im3] = ax.plot(xrange,Bx[int(kmax/2),:],linewidth=lw)
    im.set_color('orange')
    im2.set_color('blue')
    im3.set_color('red')
    init(kmax)
    plt.legend(['Ez', 'By', 'Bx'])
    plt.ylim(-1, 1) # use this for plotting E or B field
    #plt.ylim(-1e-12,1e12) # use this for plotting axion



for t in range(nsteps+1):
    pulse = get_source(t)
    
    if t == 0:
        axion_past = np.zeros([kmax, kmax])
    
    # update E
    Ex, Ey, Ez = Eupdate2d(Ex, Ey, Ez, Bx, By, Bz)
    # update physical fields
    Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy = hat2phy(Ex, Ey, Ez, Bx, By, Bz, axion)
    # inject E source
    Ez_phy[:, isource] = Ez_phy[:, isource] + pulse
    # update hat fields
    Ex, Ey, Ez, Bx, By, Bz = phy2hat(Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy, axion)
    
    # update B 
    Bx, By, Bz = Bupdate2d(Ex, Ey, Ez, Bx, By, Bz)
    
    if t == 300:
        # update physical fields
        Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy = hat2phy(Ex, Ey, Ez, Bx, By, Bz, axion)
        # inject B source
        Bz_phy[:,bsource:] = Bz_phy[:,bsource:] + 5e-5*np.ones_like(Bz_phy[:,bsource:])
        # update hat fields
        Ex, Ey, Ez, Bx, By, Bz = phy2hat(Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy, axion)
    
    # update physical fields
    Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy = hat2phy(Ex, Ey, Ez, Bx, By, Bz, axion)
    # update A
    axion = Aupdate2d(Ex, Ey, Ez, Bx, By, Bz, axion, axion_past, Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy)
    
    if t == 0:
        axion_current = np.zeros([kmax,kmax])
    axion_past = axion_current
    axion_current = axion
    
    if t % cycle == 0:
        Ex_phy, Ey_phy, Ez_phy, Bx_phy, By_phy, Bz_phy = hat2phy(Ex, Ey, Ez, Bx, By, Bz, axion)
        if plot1d == False:
            graph(t, Ez_phy)
        if plot1d == True:
            im.set_ydata(Ez_phy[int(kmax/2),:]) # blue
            ax.set_title("frame time {}".format(t))
            plt.show()
            plt.pause(0.05)
print('done')
    