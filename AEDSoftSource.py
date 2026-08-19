#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:37:13 2022

@author: szechingaudreyfung
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})


def init(xmax):
    plt.xlim((0, xmax-1))
    plt.ylim(-1,1)
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.show()
    
#%%

# update equation with source injection
def axionUpdate1d(Ez,By,axion, axion_past, source):
    # need global variable m, dt
    for x in range(1,kmax-1): # only update kmax -2 many cells
        Ez[x] = Ez[x] + 0.5 * (By[x] - By[x-1])
        
    # convert Ehat to physical field
    Ey_phy, Ez_phy, By_phy, Bz_phy = hat2phy(Ez, By, axion)
    # inject source
    Ez_phy[sourceidx] = Ez_phy[sourceidx] - source*0.5
    # convert physical field back to Ehat
    Ey, Ez, By, Bz = phy2hat(Ey_phy, Ez_phy, By_phy, Bz_phy, axion)    
    
    for x in range(0, kmax-1):
        By[x] = By[x] + 0.5 * (Ez[x+1] - Ez[x])
    for x in range(1, kmax-1): # only update kmax -2 many cells
        axion[x] = 2* axion[x] - axion_past[x] + 0.5*(axion[x+1] - 2*axion[x]+axion[x-1]) \
        - dt**2 *(axion[x]*By[x]**2/(1+axion[x]**2)**2) + dt**2 *(axion[x]*Ez[x]**2/(1+axion[x]**2)**2)\
        - dt**2 * m**2 * axion[x]
    return Ez, By, axion


def get_source(t):
    t0 = nsteps/5
    spread = nsteps/30
    source = np.exp(-0.5*(t-t0)**2/spread**2)
    return source
nsteps = 2000
plt.clf()
plt.plot(np.arange(nsteps), get_source(np.arange(nsteps)))
plt.xlim(0,nsteps)
plt.ylim(-1,1)
plt.show()

#%%

kmax = 800
nsteps = 2000
Ez = np.zeros(kmax, float)
By = np.zeros(kmax, float)

Ey = np.zeros(kmax, float)
Bz = np.zeros(kmax, float)

axion = np.zeros(kmax, float)
xrange = np.linspace(0, kmax, kmax)
dx = 1
dt = 0.5**0.5 * dx
m = 1 

# source
sourceidx = int(kmax/4)


cycle = 100
lw=2
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xrange,Ez,linewidth=lw)
[im2] = ax.plot(xrange,axion,linewidth=lw)
init(kmax)


def hat2phy(Ez, By, axion):
    Ey_phy = axion * By/(1+axion**2)
    Ez_phy = Ez/(1+axion**2)
    By_phy = By/(1+axion**2)
    Bz_phy = axion*Ez/(1+axion**2)
    
    return Ey_phy, Ez_phy, By_phy, Bz_phy

def phy2hat(Ey_phy, Ez_phy, By_phy, Bz_phy, axion):
    Ez = Ez_phy - axion*Bz_phy
    By = By_phy + axion*Ey_phy
    Ey = np.zeros_like(Ez)
    Bz = np.zeros_like(By)
    
    return Ey, Ez, By, Bz
    

for i in range(0, nsteps+1):
    source = get_source(i)
    if i == 0:
        axion_past = np.zeros(kmax, float)
    Ez, By, axion = axionUpdate1d(Ez, By, axion, axion_past, source)
    
    if i == 0:
        axion_current = np.zeros(kmax, float)
    axion_past = axion_current
    axion_current = axion
    
    if i % cycle == 0:
        Ey_phy, Ez_phy, By_phy, Bz_phy = hat2phy(Ez, By, axion)
        im.set_ydata(Ez_phy)
        im2.set_ydata(axion)
        ax.set_title("frame time {}".format(i))
        plt.show()
        plt.pause(0.01)

print('done')