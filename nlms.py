import numpy as np
import numpy.linalg as lin
from cfg import g, time, kx, ky, kz, plt
from cfg import g4tavg, ehp, esg, g4h_hp, g4h_fnsg
import time as timer
from cfg import elms, g4h_lms, Jmp, energy, itg

def coldot(w, u):
    y=0
    for i in range(w.shape[-1]):
        y+= np.dot(w[:,i],(u[:,i]))
    return y

def eta_lms(u):
    R = np.zeros((u.shape[1], u.shape[0], u.shape[0]),dtype='complex128')
    ls = np.zeros((u.shape[1], u.shape[0]),dtype='complex128')
    lmin, lmax = np.zeros(u.shape[1],dtype='complex128'), np.zeros(u.shape[1],dtype='complex128')
    for i in range(u.shape[1]):
        R[i] = u[:,i:i+1].dot(u.conj()[:,i:i+1].transpose())
        ls[i] = lin.eigvals(R[i])
        lmin[i] = np.min(ls[i])
        lmax[i] = np.max(ls[i])
    return 2/(lmax+lmin)



#eta rules: (1) 0.01/(Px*M) <eta < 0.1/(Px*M)
#           (2) 0 < eta < 2/l_max ; R = <x x^H)> has eigenvalues l
#               - fastest convergence: eta=2/(l_max + l_min)
#               - 0 < eta < 2/tr[R] = 2/sum(l)
#Classic LMS: y = w^H u
#             e = d - (w^H u)
#             w[n+1] = w[n] + (eta x e* x u)
#Normalized LMS:w[n+1] = w[n] + (eta x e* x u)/(u^H u)
#   -with no interference: eta_opt = 1
#   -general case: eta_opt = <(y-d)**2>/<e**2>
#Normalized epsilon-LMS: w[n+1] = w[n] + (eta x e* x u)/(epsilon + u^H u)
#   - 0 < eta < 2 for convergence
#   - takes care of issue with of small input power

#I use a modified normalized LMS algorithm to solve  w_m[n]


def loop(ix, iy, iz, plots=True, printing=True):
    xtr=g[:,ix,iy,iz]#,:-1]
    dtr=g[:,ix,iy,iz,-1]
    xts=g[g.shape[0]//2:,ix,iy,iz]#,:-1]
    dts=g[g.shape[0]//2:,ix,iy,iz,-1]
    M = 5 #number of data points kept
    #eta = .001/(M*Pg[ix,iy,iz,:-1])  #step size
    eta=1.0 #step size
    a = 0.9 #control param
    #if printing:
    #    print('w[n+1] = w[n] + (eta x e[n]* x u[n])/(a + u[n]^T x u[n]) \na = %1.2f, M = %i'%(a,M))

    #Initiate
    w = np.zeros((M,xtr.shape[-1]),dtype='complex128')
    W = np.zeros((xtr.shape[0],w.shape[0],w.shape[1]),dtype='complex128')
    Y = np.zeros((xtr.shape[0]),dtype='complex128')
    E = np.zeros((xtr.shape[0]),dtype='complex128')
    #E[0:M] = d[0:M]

    #Main Loop
    for n in range(M, xtr.shape[0]+1):
        #u = x[n-M:n, :].copy()
        #u[-1,-1] = 0 #d not use current g4
        u = np.column_stack((xtr[n-M:n,:-1],Y[n-M:n])) #Above may be unrealistic, so use y for g4 instead and avoid copying
        #eta = np.mean(np.absolute(d[n-M:n]-Y[n-M:n])**2, axis=0)/np.mean(np.absolute(E[n-M:n])**2, axis=0)
        #eta=eta_lms(u)
        #eta=np.minimum(eta,2)
        y = coldot(w.conj(), u)
        e = dtr[n-1] - y
        #w = w + eta*e.conj()*u/coldot(u.conj(), u)
        w = w + (eta*e.conj()*u)/(a+coldot(u.conj(), u))
        W[n-1], Y[n-1], E[n-1] = w, y, e
    #reconstruct
    w_opt=W[-1]
    #imin = np.argmin((E[500:]*E[500:].conj()).real)
    #w_opt = W[500:][imin]
    yhat = np.zeros(dts.shape,dtype='complex128')
    #start=timer.time()
    Wts = np.zeros((xts.shape[0],w.shape[0],w.shape[1]),dtype='complex128')
    Yts = np.zeros((xts.shape[0]),dtype='complex128')
    Ets = np.zeros((xts.shape[0]),dtype='complex128')
    Wts = np.zeros((xts.shape[0],w.shape[0],w.shape[1]),dtype='complex128')
    for n in range(M, xts.shape[0]+1):
        #u = x[n-M:n, :].copy()
        #u[-1,-1] = 0 #d not use current g4
        u = np.column_stack((xts[n-M:n,:-1],Yts[n-M:n]))
        #yhat[n-1] = coldot(w_opt.conj(), u)
        yh = coldot(w_opt.conj(), u)
        e = dts[n-1]-yhat[n-1]
        Wts[n-1], Yts[n-1], Ets[n-1] = w, yh, e
    #fit_error=np.mean(np.absolute(yhat-d))
    fit_error = np.mean(np.absolute(E))
    #end=timer.time()
    #total=end-start
    #print('time = %f'%total)
    yhat=Yts
    if printing:
        print('e_lms/g4_tavg = %1.3f\n     Errors: LMS/HP = %1.3f, SG/HP = %1.3f'%(fit_error/g4tavg[ix,iy,iz], fit_error/ehp[ix,iy,iz], esg[ix,iy,iz]/ehp[ix,iy,iz]))
        print(np.count_nonzero(np.absolute(E[5:])<np.absolute(E[-1])))
    #Plots
    if plots:
        plot_results(ix, iy, iz, yhat, fit_error, W, Y, E)
    return yhat, fit_error


def plot_results(ix, iy, iz, yhat, fit_error, W, Y, E):
    fig,ax=plt.subplots(2)
    ax[0].plot(np.absolute(E)**2)
    ax[0].set_xlabel(r'$ n $')
    ax[0].set_ylabel(r'$ |e[n]|^2 $')
    t0=g.shape[0]//2
    tf=t0+500

    ax[1].plot(time[t0:tf], g[t0:tf,ix,iy,iz,-1].real, label=r'$Re{g_4},\; ; \langle |g_4| \rangle = %1.2f$'%g4tavg[ix,iy,iz])
    #ax[1].plot(time[10:], Y[10:].real, label=r'$\hat g_4$')
    ax[1].plot(time[t0:tf], yhat[0:500].real, label=r'$Re{\hat g_4},\; ; \langle |\hat g_4 - g_4| \rangle = %1.2f$'%fit_error)
    ax[1].set_xlabel(r'time')
    ax[1].set_ylabel(r'$Re{g_4}$')
    ax[1].legend()
    ax[0].set_title(r'$ k_x %1.2f ,k_y %1.2f,k_z = %1.2f $' %(kx[ix],ky[iy],kz[iz]))
    plt.tight_layout()

def lms_allk():
    nx, ny, nz=len(kx), len(ky), len(kz)
    e_fit=np.zeros((nx, ny, nz))
    y_fit=np.zeros((g.shape[0], nx, ny, nz), dtype='complex128')
    for ix in range(nx):
        print('ix = %i'%ix)
        for iy in range(ny):
            for iz in range(nz):
                yhat, fit_error=loop(ix, iy, iz, plots=False, printing=False)
                e_fit[ix, iy, iz]=fit_error
                y_fit[:, ix, iy, iz]=yhat
    np.save('e_fit', e_fit)
    np.save('y_fit', y_fit)
    return y_fit, e_fit
