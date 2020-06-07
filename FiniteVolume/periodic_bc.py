''' This is a repeat of the droplet, but with periodic boundary conditions to 
simulate a long tube. We do not want the fluid to go into free-fall with the 
droplet, so we will introduce a body force in the fluid that opposes its 
weight. Surface tension will also be modeled to prevent the drop from fully
dissolving into the fluid. '''


def solverperiodic():
    #====================SET UP COMPUTATIONAL DOMAIN======================#
    # Create the grid
    Lx = 1.
    Ly = 1.
    nx = 32
    ny = 32
    dx = Lx/nx
    dy = Lx/ny
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
        for j in range(1, ny+1):
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
        xf[i]=xc-rad*np.sin(2.0*np.pi*i/nf)
        yf[i]=yc+rad*np.cos(2.0*np.pi*i/nf)

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

            for i in range(nf+1):
                ds = np.sqrt((xf[i+1]-xf[i])**2 + (yf[i+1]-yf[i])**2)
                tx[i] = (xf[i+1]-xf[i])/ds
                ty[i] = (yf[i+1]-yf[i])/ds
            
            tx[-1] = tx[1]   # Should this be tx[0]??
            ty[-1] = ty[1]
        
            for l in range(nf+1):
                nfx = sigma*(tx[l]-tx[l-1]) # Force per unit length 
                nfy = sigma*(ty[l]-ty[l-1])
                # Bilinear interpolation of surface tension
                ip = int(np.floor(xf[l]/dx))
                jp = int(np.floor((yf[l] + 0.5*dy)/dy))
                ax = xf[l]/dx - ip
                ay = yf[l]/dy + 0.5 - jp
                fx[ip,jp] += (1-ax)*(1-ay)*nfx/(dx*dy)
                fx[ip+1,jp] += ax*(1-ay)*nfx/(dx*dy)
                fx[ip,jp+1] += ay*(1-ax)*nfx/(dx*dy)
                fx[ip+1,jp+1] += ax*ay*nfx/(dx*dy)
                
                ip = int(np.floor(xf[l]/dx + 0.5))
                jp = int(np.floor(yf[l]/dy))
                ax = (xf[l]/dx + 0.5) - ip
                ay = yf[l]/dy - jp
                fy[ip,jp] += (1-ax)*(1-ay)*nfx/(dx*dy)
                fy[ip+1,jp] += ax*(1-ay)*nfx/(dx*dy)
                fy[ip,jp+1] += ay*(1-ax)*nfx/(dx*dy)
                fy[ip+1,jp+1] += ax*ay*nfx/(dx*dy)
            
            #================VELOCITY AND PRESSURE COMPUTATION===============#
            
            # Tangential velocity at boundaries
            v[0,:] = 2*vwest - v[0,:]
            v[-1,:] = 2*veast - v[-2,:]
            
            # Periodic boundary conditions for u
            u[:,0] = u[:,-2]
            u[:,-1] = u[:,1]
            
            # Compute v-velocity at y = Ly
            for i in range(1,nx+1):
                # Advection matrix
                Ay[i,-2] = (0.25/dx*((u[i,-2]+u[i,-1])*(v[i,-2]+v[i+1,-1]) -
                          (u[i-1,-1]+u[i-1,-2])*(v[i,-2]+v[i-1,-2])) +
                          0.25/dy*((v[i,-1]+v[i,-2])**2 - (v[i,-2]+v[i,-3])**2))
                
                # Diffusion matrix
                Dy[i,-2] = 2/dy*(m[i,-1]*(v[i,-1]-u[i,-2])/dy + m[i,-2] *
                               (u[i,-2]-u[i,-3])/dy) + \
                          1/dx*(0.25*(m[i+1,-2]+m[i+1,-1]+m[i,-2]+m[i,-1]) *
                               ((u[i,-1]-u[i,-2])/dy + 
                               (v[i+1,-2]-v[i,-2])/dx) - 
                                0.25*(m[i-1,-2]+m[i,-2]+m[i-1,-1]+m[i,-1]) *
                               (u[i-1,-1]-u[i-1,-2])/dy + (v[i,-2]-v[i-1,-2])/dx)
                          
                # Temporary velocity
                vt[i,-2] = v[i,-2] + dt*(-Ay[i,-2] + (1-2*rho1/(r[i+1,-2]+r[i,-2]))*gy + 
                                   fy[i,-2]/(r[i,-1]+r[i,-2]) + 
                                   2*Dy[i,-2]/(r[i,-1]+r[i,-2]))
                
            #Periodic boundary condition for v (opposite from example):
            vt[1:-1,0]=vt[1:-1,-2]
            
            # Temporary x-velocity
            for i in range(1,nx):
                for j in range(1,ny+1):
                    # Advection matrix
                    Ax[i,j] = (0.25/dx*((u[i+1,j]+u[i,j])**2 - (u[i,j]+u[i-1,j])**2) + 
                              0.25/dy*((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j]) - 
                                    (u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1])))
                    
                    # Diffusion matrix
                    Dx[i,j] = 2/dx*(m[i+1,j]*(u[i+1,j]-u[i,j])/dx + m[i,j] *
                                   (u[i,j]-u[i-1,j])/dx) + \
                              1/dy*(0.25*(m[i+1,j]+m[i+1,j+1]+m[i,j]+m[i,j+1]) *
                                   ((u[i,j+1]-u[i,j])/dy + 
                                   (v[i+1,j]-v[i,j])/dx) - 
                                    0.25*(m[i,j-1]+m[i-1,j-1]+m[i-1,j]+m[i-1,j]) *
                                   (u[i,j] - u[i,j-1])/dy + (v[i+1,j-1]-v[i,j-1])/dx)
                              
                    # Temporary velocity
                    ut[i,j] = u[i,j] + dt*(-Ax[i,j] + (1-2*rho1/(r[i+1,j]+r[i,j]))*gx + 
                                       fx[i,j]/(r[i+1,j]+r[i,j]) + 
                                       2*Dx[i,j]/(r[i+1,j]+r[i,j]))
            
            # Temporary y-velocity
            for i in range(1,nx+1):
                for j in range(1,ny):
                    # Advection matrix
                    Ay[i,j] = (0.25/dx*((u[i,j]+u[i,j+1])*(v[i,j]+v[i+1,j]) -
                              (u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j])) +
                              0.25/dy*((v[i,j+1]+v[i,j])**2 - (v[i,j]+v[i,j-1])**2))
                    
                    # Diffusion matrix
                    Dy[i,j] = 2/dy*(m[i,j+1]*(v[i,j+1]-u[i,j])/dy + m[i,j] *
                                   (u[i,j]-u[i,j-1])/dy) + \
                              1/dx*(0.25*(m[i+1,j]+m[i+1,j+1]+m[i,j]+m[i,j+1]) *
                                   ((u[i,j+1]-u[i,j])/dy + 
                                   (v[i+1,j]-v[i,j])/dx) - 
                                    0.25*(m[i-1,j]+m[i,j]+m[i-1,j+1]+m[i,j+1]) *
                                   (u[i-1,j+1]-u[i-1,j])/dy + (v[i,j]-v[i-1,j])/dx)
                              
                    # Temporary velocity
                    vt[i,j] = v[i,j] + dt*(-Ay[i,j] + (1-2*rho1/(r[i+1,j]+r[i,j]))*gy + 
                                       fy[i,j]/(r[i,j+1]+r[i,j]) + 
                                       2*Dy[i,j]/(r[i,j+1]+r[i,j]))
                    
            # Source term for pressure equation
            rt = np.copy(r)
            lrg = 1000.
            rt[0,:] = lrg
            rt[-1,:] = lrg
            rt[:,0] = lrg
            rt[:,-1] = lrg
        
            # Assemble and compute pressure
            for i in range(1,nx+1):
                for j in range(1,ny+1):
                    tmp1[i,j] = 1/(2*dt)*((ut[i,j]-ut[i-1,j])/dx + (vt[i,j]-vt[i,j-1])/dy)
                    tmp2[i,j] = 1/((1/dx**2)*(1/(rt[i+1,j]+rt[i,j]) + 1/(rt[i,j]+rt[i-1,j])) + 
                                      (1/dy**2)*(1/(rt[i,j+1]+rt[i,j]) + 1/(rt[i,j]+rt[i,j-1])))
                    
            it = 0
            while True:
                pn = np.copy(p)
                it += 1
                for i in range(1,nx+1):
                    for j in range(1,ny+1):
                        p[i,j] = (1-beta)*p[i,j] + beta*tmp2[i,j]*((1/dx**2*(p[i+1,j]/(rt[i+1,j]+rt[i,j])+p[i-1,j]/(rt[i,j]+rt[i-1,j])) +
                                                              1/dy**2*(p[i,j+1]/(rt[i,j+1]+rt[i,j])+p[i,j-1]/(rt[i,j]+rt[i,j-1])) - tmp1[i,j]))
                #Periodic boundary condition for p:
                p[:,0]=p[:,ny]; p[:,ny+1]=p[:,1]
                if it%100 == 0:
                    print(f'Pressure loop iterations: {it}')
                if np.abs(pn-p).max() < maxError:
                    print(f'Pressure convergence achieved in {it} iterations.')
                    break
                if it > maxit:
                    print(f'Pressure convergence not achieved. Max iterations reached in pressure loop.')
                    break
        
            # Update velocities
            for i in range(1,nx):
                for j in range(1,ny+1):
                    u[i,j] = ut[i,j] - 2*dt/dx*(p[i+1,j]-p[i,j])/(r[i+1,j]+r[i,j])
            for i in range(1,nx+1):
                for j in range(1,ny):
                    v[i,j] = vt[i,j] - 2*dt/dy*(p[i,j+1]-p[i,j])/(r[i,j+1]+r[i,j])
            
            # Calculate v at y=0
            v[1:-1,0] = vt[1:-1,0] - 2*dt/dy*(p[1:-1,1]-p[1:-1,0])/(r[1:-1,1]+r[1:-1,0])
            # Calculate v at y = Ly
            v[1:-1,ny] = vt[1:-1,ny] - 2*dt/dy*(p[1:-1,ny+1]-p[1:-1,ny])/(r[1:-1,ny+1]+r[1:-1,ny])
            
        #===============FRONT TRACKING AND DENSITY DISTRIBUTION==================#
        # Advect front (mine, validated)
        # Bilinear interpolation of velocity in a cell to determine the front
        # velocity
        uf=np.zeros((nf+2)); vf=np.zeros((nf+2))
        for i in range(1, nf+1):
            ip = int(np.floor(xf[i]/dx))
            jp = int(np.floor((yf[i]+0.5*dy)/dy))
            ax = xf[i]/dx-ip
            ay = (yf[i]+0.5*dy)/dy-jp
            uf[i] = (1-ax)*(1-ay)*u[ip,jp] + ax*(1-ay)*u[ip+1,jp] + (1-ax)*ay*u[ip,jp+1] + ax*ay*u[ip+1,jp+1]
    
            ip = int(np.floor((xf[i]+0.5*dx)/dx))
            jp = int(np.floor(yf[i]/dy))
            ax = (xf[i]+0.5*dx)/dx-ip
            ay = (yf[i]/dy)-jp
            vf[i] = (1-ax)*(1-ay)*v[ip,jp] + ax*(1-ay)*v[ip+1,jp] + (1-ax)*ay*v[ip,jp+1] + ax*ay*v[ip+1,jp+1]
            
        for i in range(1, nf+1):
            xf[i] = xf[i] + dt*uf[i]
            yf[i] = yf[i] + dt*vf[i]
            if yf[i] < 0:
                yf[i] = yf[i] + Ly
    
        xf[0] = xf[-2]
        yf[0] = yf[-2]
        xf[-1] = xf[1]
        yf[-1] = yf[1]
    
        
        # Distribute gradient (mine, validated)
        fx = np.zeros((nx+2,ny+2))
        fy = np.zeros((nx+2, ny+2))
        for i in range(1, nf+1):
            nfx = -0.5*(yf[i+1]-yf[i-1])*(rho2-rho1)
            nfy = 0.5*(xf[i+1]-xf[i-1])*(rho2-rho1)
    
            ip=int(np.floor(xf[i]/dx)) 
            jp=int(np.floor((yf[i]+0.5*dy)/dy))
            ax=xf[i]/dx-ip
            ay=(yf[i]+0.5*dy)/dy-jp;
            fx[ip,jp] = fx[ip,jp]+(1-ax)*(1-ay)*nfx/(dx*dy)
            fx[ip+1,jp] = fx[ip+1,jp]+ax*(1-ay)*nfx/(dx*dy)
            fx[ip,jp+1] = fx[ip,jp+1]+ay*(1-ax)*nfx/(dx*dy) 
            fx[ip+1,jp+1]=fx[ip+1,jp+1]+ax*ay*nfx/(dx*dy)
    
            ip=int(np.floor((xf[i]+0.5*dx)/dx))
            jp=int(np.floor(yf[i]/dy))
            ax=(xf[i]+0.5*dx)/dx-ip
            ay=yf[i]/dy-jp
            fy[ip,jp]=fy[ip,jp]+(1-ax)*(1-ay)*nfy/(dx*dy)
            fy[ip+1,jp]=fy[ip+1,jp]+ax*(1-ay)*nfy/(dx*dy)
            fy[ip,jp+1]=fy[ip,jp+1]+(1-ax)*ay*nfy/(dx*dy)
            fy[ip+1,jp+1]=fy[ip+1,jp+1]+ax*ay*nfy/(dx*dy)
    
    
        #Construct the density
        iter=0
        while True:
            rt=r.copy()
            iter=iter+1
            r[1:-1,1:-1]=(1-beta)*r[1:-1,1:-1]+0.25*(r[2:,1:-1]+r[0:-2,1:-1]+r[1:-1,2:]+r[1:-1,0:-2]+
                              dx*(fx[0:-2,1:-1]-fx[1:-1,1:-1])+ dy*(fy[1:-1,0:-2]-fy[1:-1,1:-1]))
            if np.abs(rt-r).max()<maxError:
                break
            if iter>maxiter:
                break
        
        # Update viscosity
        m = np.ones((nx+2,ny+2))*m1
        m[1:-1,1:-1] = m1+(m2-m1)*(r[1:-1,1:-1]-rho1)/(rho2-rho1)
        
        #================END SECOND ORDER=================#
        u = 0.5*(u+un)
        v = 0.5*(v+vn)
        r = 0.5*(r+rn)
        m = 0.5*(m+mn)
        xf = 0.5*(xf+xfn)
        yf = 0.5*(yf+yfn)
        
        #================ADD POINTS TO THE FRONT=================#
        xfold = np.copy(xf)
        yfold = np.copy(yf)
        j = 0
        for i in range(1, nf+1):
            ds = np.sqrt(((xfold[i]-xf[j])/dx)**2 + ((yfold[i] - yf[j])/dy)**2)
            if ds > 0.5:
                if i > nf:
                    oldsize = xf.size
                    xf = np.resize(xf, oldsize+1)
                    yf = np.resize(yf, oldsize+1)
                xf[j] = 0.5*(xfold[i]+xf[j-1])
                yf[j] = 0.5*(yfold[i]+yf[j-1])
                if i > nf:
                    oldsize = xf.size
                    xf = np.resize(xf, oldsize+1)
                    yf = np.resize(yf, oldsize+1)
                xf[j] = xfold[i]
                yf[j] = yfold[i]
            elif ds < 0.25:
                pass
            else:
                j=j+1
                if j > nf:
                    oldsize = xf.size
                    xf = np.resize(xf,oldsize+1)
                    yf = np.resize(yf,oldsize+1)
                xf[j]=xfold[l]
                yf[j]=yfold[l]
                
        xf = np.resize(xf,j+2)
        yf = np.resize(yf,j+2)
        uf = np.resize(uf,j+2)
        vf = np.resize(vf,j+2)
        tx = np.resize(tx,j+2)
        ty = np.resize(ty,j+2)
        nf=j
        xf[0]=xf[nf]
        yf[0]=yf[nf]
        xf[nf+1]=xf[1]
        yf[nf+1]=yf[1]
        
        #==================UPDATE THE FRAME====================#
        yfc = np.mean(yf)
        movecell = int(np.floor(np.abs(0.5-yc)/dy))
        ud=np.copy(u)
        vd=np.copy(v)
        pd=np.copy(p)
        rd=np.copy(r)
        yfd=np.copy(yf)
        yf[:] = yfd[:]-yfc+0.5
        if 0.5 > yfc:
            u[:,0:-movecell-1]=ud[:,movecell:-1]
            u[:,-movecell:-1]=ud[:,0:movecell-1]

            v[:,0:-movecell-1]=vd[:,movecell:-1]
            v[:,-movecell:-1]=vd[:,0:movecell-1]

            p[:,0:-movecell-1]=pd[:,movecell:-1]
            p[:,-movecell:-1]=pd[:,0:movecell-1]

            r[:,0:-movecell-1]=rd[:,movecell:-1]
            r[:,-movecell:-1]=rd[:,0:movecell-1]     

        else:
            u[:,movecell:-1]=ud[:,0:-movecell-1]
            u[:,0:movecell-1]=ud[:,-movecell:-1]

            v[:,movecell:-1]=vd[:,0:-movecell-1]
            v[:,0:movecell-1]=vd[:,-movecell:-1]

            p[:,movecell:-1]=pd[:,0:-movecell-1]
            p[:,0:movecell-1]=pd[:,-movecell:-1]

            r[:,movecell:-1]=rd[:,0:-movecell-1]
            r[:,0:movecell-1]=rd[:,-movecell:-1]
        
        #===============PLOTTING==================#
        uu = 0.5*(u[0:nx,1:ny+1] + u[0:nx,0:ny])
        vv = 0.5*(v[1:nx+1,0:ny] + v[0:nx,0:ny])
        yy, xx = np.mgrid[0:(nx-1)*dx:nx*1j, 0:(ny-1)*dy:ny*1j]
        X, Y = np.meshgrid(x, y)
        plt.clf()
        plt.contourf(x,y,r.T,5, cmap=cm.jet)
        plt.colorbar()
        plt.quiver(xx,yy,uu.T,vv.T)
        plt.plot(xf[0:nf+1], yf[0:nf+1],linewidth=5.0)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.05)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        print(f'Step {step}')
    plt.show()
        
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython import display
    from matplotlib import cm
    #Domain size and constant parameters
    Lx=1.0; Ly=1.0
    gx=0.0; gy=-100.0;
    rho1=1.0; rho2=2.0
    m1=0.01
    m2=0.05
    rro=rho1;
    sigma=10.0
    
    #Velocity at wall boundary
    unorth=0; usouth=0; veast=0; vwest=0; 
    
    #Initial drop size and location
    time=0.0; rad=0.15; xc=0.5; yc=0.5;
    
    #Numerical variables
    nx=32; ny=32;
    dt=0.00125; nstep=100
    maxiter=200; maxError=0.001
    beta=1.2;
    
    solverperiodic()
            
        
                
                
                
                
                
                
                
                
                
                
                
                
                