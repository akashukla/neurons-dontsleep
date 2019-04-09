# coding: utf-8
import numpy as np
import numpy.linalg as lin
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = [6.4,4.8]
mpl.rc('text',usetex=True)
from cfg import g, kx, ky, kz, time, Pg

def phi0_(v):
    phi = 1/(1+np.exp(-v))
    deriv = np.exp(-v)/((1+np.exp(-v))**2)
    return phi,deriv


def phi1_(v):
    m = 1
    phi = np.tanh(np.absolute(v)/m)*np.exp(1j*np.angle(v))
    deriv = (1-np.tanh(np.absolute(v)/m)**2)*np.exp(1j*np.angle(v))+1j*np.tanh(np.absolute(v)/m)*np.exp(1j*np.angle(v))/(1+v**2)
    return phi,deriv


def phi2_(v):
    c, r = 1, 1
    phi = v/(c+np.absolute(v)/r)
    deriv = ( (c+np.absolute(v)/r) - np.absolute(np.sign(v)/r))/(c+np.absolute(v)/r)**2
    return phi,deriv


def phi3_(v):
    phi = 0.5*(1+np.cos(np.angle(v)))*v
    deriv = -0.25j*np.sin(np.angle(v))*v/v.conj()
    return phi,deriv


def phi_(v,which_func):
    switcher = {
        0: phi0_(v),
        1: phi1_(v),
        2: phi2_(v),
        3: phi3_(v),
    }
    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(which_func, (v,1))


def filter(w, u):
    y=0
    for i in range(w.shape[-1]):
        y+= np.dot(w[:,i].conj(),(u[:,i]))
    return y


def update(w, u, d, eta, gamma):
    v = filter(w, u)
    y, deriv = phi_(v,None)
    e = d - y
    w = w + 2*eta*np.conjugate(e)*u*deriv
    return w, y, e

    #phi=1/(1+e**-v)
    #dphi/dv=(-e**-v)/(1+e**-v)**2
    #complex phi(z) = z/(c+|z|/r) ; c,r > 0 usually c=r=1
    #dphi/dz = ( (c+|z|/r) - z(+-1/r) )/(c+|z|/r)**2 ; +/- for z>0,z<0
    #complex phi(z) = tanh(|z|/m)exp(j*phi(z)) ; phi(z) = phase of z, m>0 usually m=1
    #d/dx (tanh(x)) = 1-tanh(x)**2, d/dx(arctan(x)) = 1/(1+x**2)
    #dphi/dz = (1-tanh(|z|/m)**2)e**(j*np.angle(z))+j*tanh(|z|/m)*np.exp(j*np.angle(z))/(1+z**2)



def leaky_update(w, u, d, eta, gamma):
    y = filter(w, u)
    e = error(d, y)
    w = (1-gamma*eta)*w+2*eta*e*u
    return w, y, e

def loop(ix,iy,iz):
    x=g[:,ix,iy,iz,:-1]
    d=g[:,ix,iy,iz,-1]
    M = 10  #number of data points kept
    eta = .001/(M*Pg[ix,iy,iz,:-1])  #step size
    gamma = 0.1
    #eta rules: (1) 0.01/(Px*M) <eta < 0.1/(Px*M)
    #           (2) 0 < eta < 2/l_max ; R = <x x^H)> has eigenvalues l
    #               - fastest convergence: eta=2/(l_max + l_min)
    #               - 0 < eta < 2/tr[R] = 2/sum(l)
    #Classic LMS: y = w^H u
    #             e= d - w^H u
    #             w[n+1] = w[n] + eta*e.conj()*u
    #Normalized LMS:w[n+1] = w[n] + (eta*e.conj()*u)/(u^H u)
    #   -with no interference: eta_opt = <(y-d)**2>/<e**2>
    #   -general case: eta_opt = <(y-d)**2>/<e**2>

    #Initiate
    w = np.zeros((M,x.shape[-1]),dtype='complex128')
    W=np.zeros((x.shape[0],w.shape[0],w.shape[1]),dtype='complex128')
    Y=np.zeros((x.shape[0]),dtype='complex128')
    E=np.zeros((x.shape[0]),dtype='complex128')

    #Main Loop
    for n in range(M,x.shape[0]):
        u = x[n-M:n,:]
        w,y,e = update(w,u,d[n],eta,gamma)
        W[n-M]=w
        Y[n-M]=y
        E[n-M]=e
    if plots:
        plot_results(W,Y,E)
    return W,Y,E

def plot_results(W,Y,E):
    fig,ax=plt.subplots(2)
    ax[0].plot((E*np.conjugate(E)).real)
    ax[0].set_xlabel(r'$ n $')
    ax[0].set_ylabel(r'$ |e[n]|^2 $')
    ax[1].plot(time[10:],g[10:,0,5,2,-1].real,label=r'$g_4$')
    ax[1].plot(time[10:],Y[:-10].real,label=r'$\hat g_4$')
    ax[1].set_xlabel(r'time')
    ax[1].legend()
    plt.title(r'$ k_x %1.2f ,k_y %1.2f,k_z = %1.2f $' %(kx[ix],ky[iy],kz[iz]))
    plt.tight_layout()

    #Plots
    #plt.figure()
    #plt.stem(w)
    #plt.xlabel(r'$ n $')
    #plt.ylabel(r'$ h[n] $')

    #plt.savefig('q1b')

    #plt.figure()
    #plt.plot(d,label='d')
    #plt.plot(Y,label='y')
    #plt.legend()
