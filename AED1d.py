#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:15:38 2022

@author: szechingaudreyfung
"""

import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.constants as constants
import numba
#get_ipython().run_line_magic('matplotlib', 'auto')



#%%

"""
Everyting here is in natural units, eg. [E] = [B] = [eV^2]; [dt] = [eV^-1].
"""


"""
For stability reason, I impose two criteria:
1) dt/dx = 0.5
2) dt2 = 0.25
Therefore dx is constrained instead of dt
"""


def init(xmax):
    plt.xlim((0, xmax-1))
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    plt.show()


# conversion between hat fields and physical fields

@numba.jit(nopython=True)
def hat2phy(Ex, Ey, Bx, By, axion):
    Ex_phy = (Ex + kappa*axion*Bx)/(1+kappa**2*axion**2)
    Ey_phy = (Ey + kappa*axion*By)/(1+kappa**2*axion**2)
    Bx_phy = (Bx - kappa*axion*Ex)/(1+kappa**2*axion**2)
    By_phy = (By - kappa*axion*Ey)/(1+kappa**2*axion**2)
    
    return Ex_phy, Ey_phy, Bx_phy, By_phy
@numba.jit(nopython=True)
def phy2hat(Ex_phy, Ey_phy, Bx_phy, By_phy, axion):
    Ex = Ex_phy - kappa*axion*Bx_phy
    Ey = Ey_phy - kappa*axion*By_phy
    Bx = Bx_phy + kappa*axion*Ex_phy
    By = By_phy + kappa*axion*Ey_phy
    
    return Ex, Ey, Bx, By

# update equations

@numba.jit(nopython=True)
def Eupdate1d(Ex, Ey, Bx, By):
    for k in range(1,kmax-1):
        Ex[k] = Ex[k] + 0.5 * (By[k-1] - By[k])
        Ey[k] = Ey[k] + 0.5 * (Bx[k] - Bx[k-1])
    return Ex, Ey
@numba.jit(nopython=True)
def Bupdate1d(Ex, Ey, Bx, By):
    for k in range(0,kmax-1):
        Bx[k] = Bx[k] + 0.5 * (Ey[k+1] - Ey[k])
        By[k] = By[k] + 0.5 * (Ex[k] - Ex[k+1])
    return Bx, By
@numba.jit(nopython=True)
def Aupdate1d(Ex, Ey, Bx, By, axion, axion_past, Ex_phy, Ey_phy, Bx_phy, By_phy):
    for k in range(1,kmax-1):
        axion[k] = 2*axion[k] - axion_past[k] + 0.25*(axion[k+1] - 2*axion[k] + axion[k-1])\
        - dt2*kappa*(Ex_phy[k]*Bx_phy[k] + Ey_phy[k]*By_phy[k]) - dt2*(m**2*axion[k])
        
    return axion

#%%

# intial set-up

"Control"
nsteps = 6000
kmax = 4000
source = 'plane waves' # 'pulse'

# initial values
Ex_phy = Ey_phy = np.zeros(kmax, float)
By_phy = np.zeros(kmax, float)
Bx_phy = np.zeros(kmax, float) # Bext source
axion = np.zeros(kmax, float)
Ex= Ey= Bx= By = np.zeros(kmax, float)

xrange = np.linspace(0,kmax, kmax)


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
    dt2 = dt**2 # 6.97e-11 eV^-1
    dt_si = dt*hbar # 5.5e-21
    
    angFreq = 2*math.pi*freq # ~2.8e19 rad/s
    w = angFreq*dt_si # 0.157 rad per grid
    nsteps =12000
    
    Savg = 1400
    Erms = (constants.mu_0 * constants.c * Savg)**0.5 # ~700 (SI)
    elemC = 5.2909e-19
    J2eV = 6.242e18
    E0_natural = Erms * elemC * constants.c**2 *  J2eV*hbar # 103041 eV^2 (convert to natural units)
    
    scale2 = 200000 # in eV^2; scale all quantity by a constant so that E ~ O(1) 
    scale = scale2**0.5
    E0 = E0_natural/scale2 # say 3000 eV^2 = 0.5 grid units
    
    
    
    @numba.jit(nopython=True)
    def get_source(t):
        source = E0*np.sin(w*t)
        return source

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




sourceidx = int(kmax/6)
Bsourceidx = 0
Bext_si = 9 #Tesla
Bext_natural = Bext_si * elemC * constants.c**2 * J2eV * hbar # eV^2
Bext = Bext_natural/scale2

# axion
m = 1/scale # scaled axion mass m = 1 eV 
kappa = 1

#%%

plt.clf()
plt.close()
cycle = 100
lw=2
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xrange,Ex_phy,linewidth=lw)
[im2] = ax.plot(xrange,axion,linewidth=lw)
im.set_color('blue')
im2.set_color('orange')
plt.ylim(-1e-12, 1e-12) # to plot axion only, use this ylim
#plt.ylim(-1, 1) # to plot Ex, use this ylim
plt.legend(['$E_{phy,x}$','axion'])
ax.set_ylabel(r'$\theta$')
init(kmax)

"Main Code"

for t in range(0,nsteps+1):
    source = get_source(t)
    source2 = get_source(t+0.5)
    
    if t == 0:
        axion_past = np.zeros(kmax, float)
    # ABC
    Exleft_0 = Ex[1]
    Eyleft_0 = Ey[1]
    Exright_0 = Ex[-2]
    Eyright_0 = Ey[-2]
        
    # update E
    Ex, Ey = Eupdate1d(Ex, Ey, Bx, By)
    # convert to physical fields
    Ex_phy, Ey_phy, Bx_phy, By_phy = hat2phy(Ex, Ey, Bx, By, axion)
    # inject E source
    Ex_phy[sourceidx] = Ex_phy[sourceidx] + 0.5*source2

    #convert back to hat's fields
    Ex, Ey, Bx, By = phy2hat(Ex_phy, Ey_phy, Bx_phy, By_phy, axion)
    
    # ABC
    if t == 0:
        Exleft_ = None
        Eyleft_ = None
        Exright_ = None
        Eyright_ = None
    if t != 0:
        Ex[0] = Exleft_
        Ey[0] = Eyleft_
        Ex[-1] = Exright_
        Ey[-1] = Eyright_
    
    Exleft_ = Exleft_0
    Eyleft_ = Eyleft_0
    Exright_ = Exright_0
    Eyright_ = Eyright_0
    
    # update B
    Bx, By = Bupdate1d(Ex, Ey, Bx, By)
    
    # update physical fields
    Ex_phy, Ey_phy, Bx_phy, By_phy = hat2phy(Ex, Ey, Bx, By, axion)
    
    # TFSF
    By_phy[sourceidx-1] = By_phy[sourceidx-1] + 0.5*source
    #Bx_phy[sourceidx-1] = Bx_phy[sourceidx-1] + 0.5*source
    
    # update hat fields
    Ex, Ey, Bx, By = phy2hat(Ex_phy, Ey_phy, Bx_phy, By_phy, axion)
    
    if t == 2000:
        
        # convert to physical fields
        Ex_phy, Ey_phy, Bx_phy, By_phy = hat2phy(Ex, Ey, Bx, By, axion)
        # inject B source
        Bx_phy[Bsourceidx:] = Bx_phy[Bsourceidx:] + Bext*np.ones_like(Bx_phy[Bsourceidx:])
        #convert back to hat's fields
        Ex, Ey, Bx, By = phy2hat(Ex_phy, Ey_phy, Bx_phy, By_phy, axion)

        
    # update A
    Ex_phy, Ey_phy, Bx_phy, By_phy = hat2phy(Ex, Ey, Bx, By, axion)
    axion = Aupdate1d(Ex, Ey, Bx, By, axion, axion_past, Ex_phy, Ey_phy, Bx_phy, By_phy)
    
    if t == 0:
        axion_current = np.zeros(kmax, float)
    axion_past = axion_current
    axion_current = axion
    
    # plot
    if t % cycle == 0:
        Ex_phy, Ey_phy, Bx_phy, By_phy = hat2phy(Ex, Ey, Bx, By, axion)
        #im.set_ydata(Ex_phy) # blue
        im2.set_ydata(axion) # orange
        ax.set_title("frame time {}".format(t))
        #plt.savefig('/Users/szechingaudreyfung/Desktop/PHYS 879 HPC/Projects/plots/1dplane_axion/axion{}.png'.format(t))
        plt.show()
        plt.pause(0.05)
print('done')


