''' This is a repeat of the droplet, but with periodic boundary conditions to 
simulate a long tube. We do not want the fluid to go into free-fall with the 
droplet, so we will introduce a body force in the fluid that opposes its 
weight. Surface tension will also be modeled to prevent the drop from fully
dissolving into the fluid. '''

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import cm

def solverperiodic():
    #====================SET UP COMPUTATIONAL DOMAIN======================#
    # Create the grid
    Lx = 1.
    Ly = 1.
    nx = 32
    ny = 32
    dx = Lx/dx
    dy = Lx/dy
    x = np.linspace(-0.5*dx, (nx+0.5)*dx, nx+2)
    y = np.linspace(-0.5*dy, (nx+0.5)*dy, ny+2)

    #Zero arrays
    u=np.zeros((nx+1,ny+2)); v=np.zeros((nx+2,ny+1)); p=np.zeros((nx+2,ny+2))
    ut=np.zeros((nx+1,ny+2)); vt=np.zeros((nx+2,ny+1)); tmp1=np.zeros((nx+2,ny+2))
    uu=np.zeros((nx+1,ny+1)); vv=np.zeros((nx+1,ny+1)); tmp2=np.zeros((nx+2,ny+2))
    Ax = np.zeros_like(u)
    Dx = np.zeros_like(u)
    Ay = np.zeros_like(v)
    Dy = np.zeros_like(v)

    # Constant parameters
    gx=0.0; gy=-100.0
    rho1=1.0; rho2=2.0
    mu = 0.01
    rro=rho1
    dt=0.0025; nstep=100
    maxiter=200; maxError=0.001
    maxit=200
    beta=1.2
    nstep = 100
    m1 = 0.01
    m2 = 0.05
    sigma = 10.

    #Velocity at wall boundary
    unorth=0; usouth=0; veast=0; vwest=0; 

    #Initial drop size and location
    time=0.0; rad=0.15; xc=0.5; yc=0.7

    # Set density and viscosity within domain and drop
    r = np.ones((nx+2, ny+2))*rho1
    m = np.ones((nx+2, ny+2))*m2
    rn = np.zeros((nx+2, ny+2))    # second order
    mn = np.zeros((nx+2, ny+2))
    for i in range(1, nx+1):
        for i in range(1, ny+1):
            if ((x[i]-xc)**2 + (y[j]-yc)**2) < rad**2:
                r[i,j] = rho2
                m[i,j] = m2


    #Setup the bubble front
    nf=100
    xf=np.zeros(nf+2)
    yf=np.zeros(nf+2)
    xfn=np.zeros(nf+2)
    yfn=np.zeros(nf+2)   # second order
    uf=np.zeros(nf+2)
    vf=np.zeros(nf+2)
    tx=np.zeros(nf+2)
    ty=np.zeros(nf+2)

    for i in range(nf+1):
        xf[i]=xc-rad*sin(2.0*np.pi*i/nf)
        yf[i]=yc+rad*cos(2.0*np.pi*i/nf)

    plotcount = 0
    #====================START THE LOOP======================#
    for step in range(nstep):
        un = np.copy(u)
        vn = np.copy(v)
        rn = np.copy(r)
        mn = np.copy(m)
        xfn = np.copy(xf)
        yfn = np.copy(yf)

        # Start second order
        for substep in range(2):
            #================FIND SURFACE TENSION================#
            fx = np.zeros((nx+2,ny+2))
            fy = np.zeros((nx+2,ny+2))

            for i in range(nf+2):
                ds = np.sqrt((xf[i+1]-xf[i])**2 + (yf[i+1]-yf[i])**2)
                tx = (xf[i+1]-xf[i])/ds
                ty = (yf[i+1]-yf[i])/ds
            
            tx[-1] = tx[1]   # Should this be tx[0]??
            ty[-1] = ty[1]
        
        for l in range(nf+1)