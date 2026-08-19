import numpy as np
from matplotlib import pyplot as plt
import numba
#get_ipython().run_line_magic('matplotlib', 'auto')

plt.rcParams.update({'font.size': 17}) # keep those graph fonts readable!
plt.rcParams['figure.dpi'] = 120
#plt.display_figs()

def graph(t):
    plt.clf()
    ax = fig.add_axes([.25, .25, .6, .6])

    img = ax.contourf(Ez)
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$E_z$ (arb.units)')
    ax.set_title('frame time{}'.format(t))
    plt.savefig('/Users/szechingaudreyfung/Desktop/PHYS 879 HPC/Projects/plots/frames/frame%03d.png'%(t))
    # plt.show()
    # plt.pause(0.01)

@numba.jit(nopython=True)
def Eupdate2d(Ez, Bx, By, source):
    for y in range(1,kmax-1):
        for x in range(1,kmax-1):
            Ez[y,x] = Ez[y,x] + 0.5*(By[y,x] - By[y,x-1] - Bx[y,x]+ Bx[y-1,x])
    Ez[jsource, isource] = Ez[jsource, isource] + pulse

    return Ez

@numba.jit(nopython=True)
def Bupdate2d(Ez, Bx, By):
    for y in range(0,kmax-1):
        for x in range(0,kmax-1):
            Bx[y,x] = Bx[y,x] + 0.5*(Ez[y,x] - Ez[y+1,x])
            By[y,x] = By[y,x] + 0.5*(Ez[y,x+1] - Ez[y,x])

    return Bx, By

nsteps = 1000
t = np.arange(0,nsteps+1)
spread = 60
t0 = spread*6

def get_source(t):
    source = -np.exp(-0.5*(t-t0)**2/spread**2)*np.cos(t*np.pi*0.01)
    return source

kmax = 160
Ez = np.zeros([kmax, kmax], float)
Bx = np.zeros([kmax, kmax], float)
By = np.zeros([kmax, kmax], float)

# source
isource = int(kmax/2)
jsource = int(kmax/2)

cycle = 100
fig = plt.figure(figsize=(8,6))

for i in range(0,nsteps+1):
    pulse = get_source(i)

    Ez = Eupdate2d(Ez, Bx, By, pulse)

    Bx, By = Bupdate2d(Ez, Bx, By)

    if i%cycle == 0:
        graph(i)
