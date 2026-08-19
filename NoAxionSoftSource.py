#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:29:17 2022

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
# No axion, soft source AED


def update1d(Ez, By, source):
    for x in range(1,kmax-1): # only update kmax -2 many cells
        Ez[x] = Ez[x] + 0.5 * (By[x] - By[x-1])
    # source injectin
    Ez[sourceidx] =  Ez[sourceidx] - source*0.5
    for x in range(0, kmax-1):
        By[x] = By[x] + 0.5 * (Ez[x+1] - Ez[x])
    return Ez, By


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
xrange = np.linspace(0,kmax, kmax)

# source
sourceidx = int(kmax/4)

plt.clf()
cycle = 100
lw=2
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xrange,Ez,linewidth=lw)
[im2] = ax.plot(xrange,By,linewidth=lw)
init(kmax)


for i in range(0,nsteps+1):
    
    source = get_source(i)
    Ez, By = update1d(Ez, By, source)
    
    if i % cycle == 0:
        im.set_ydata(Ez)
        im2.set_ydata(By)
        ax.set_title("frame time {}".format(i))
        plt.show()
        plt.pause(0.01)