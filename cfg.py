import numpy as np
import matplotlib.pyplot as plt

#params
par = {'nu':.00222, 'omt':7.5, 'omn':1.0, 'Ti0Te':1.0, 'kxmin':0.05, 'kxmax0':1.55, 'kymin':0.05, 'kymax0':1.55, 'kzmin':0.1, 'kzmax0':4.7, 'nkx0':32, 'nky0':64, 'nkz0':96, 'nv0':48, 'nh0':1, 'nspec':1, 'hyp_x':1.5, 'hyp_y':1.5, 'hyp_z':0.0, 'hypx_order':8, 'hypy_order':8, 'hypz_order':8, 'hyp_v':26.8, 'hypv_order':8, 'hyp_conv':2.0, 'num_k_hyp_conv':2, 'hyp_conv_ky':False, 'np_herm':64, 'np_kz':1, 'np_hank':1, 'np_spec':1, 'hyp_nu':0.1, 'nuno_closure':False, 'em_conserve':False, 'etg_factor':0.0}

parcc = {'nu':.00000, 'omt':7.5, 'omn':1.0, 'Ti0Te':1.0, 'kxmin':0.05, 'kxmax0':1.55, 'kymin':0.05, 'kymax0':1.55, 'kzmin':0.1, 'kzmax0':4.7, 'nkx0':32, 'nky0':64, 'nkz0':96, 'nv0':48, 'nh0':1, 'nspec':1, 'hyp_x':0.0, 'hyp_y':0.0, 'hyp_z':0.0, 'hypx_order':0, 'hypy_order':0, 'hypz_order':0, 'hyp_v':00.0, 'hypv_order':0, 'hyp_conv':0.0, 'num_k_hyp_conv':0, 'hyp_conv_ky':False, 'np_herm':64, 'np_kz':1, 'np_hank':1, 'np_spec':1, 'hyp_nu':0.0, 'nuno_closure':False, 'em_conserve':False, 'etg_factor':0.0}

#data
kx, ky, kz = np.load('/home/akash/gs/data/kx.npy'), np.load('/home/akash/gs/data/ky.npy'), np.load('/home/akash/gs/data/kz.npy')
nx,ny,nz = len(kx), len(ky), len(kz)
time = np.load('/home/akash/gs/data/time.npy', mmap_mode='r')[500:]
g = np.load('/home/akash/gs/data/g_allk_g04.npy', mmap_mode='r')[500:]
#gnl = np.load('/home/akash/gs/data/gnl_allk_g04.npy', mmap_mode='r')

#EIGENVALUES, EIGENVECTORS
itg, itgcc = np.load('/home/akash/gs/calculated/itg.npy').real, np.load('/home/akash/gs/calculated/itgcc.npy').real
ev, evcc = np.load('/home/akash/gs/calculated/ev.npy'), np.load('/home/akash/gs/calculated/evcc.npy')

#CLOSURES
g4h_hp = np.load('/home/akash/gs/g4s/g4hhp.npy', mmap_mode='r')[500:]
g4h_fnsg = np.load('/home/akash/gs/g4s/g4h_fnsg.npy', mmap_mode='r')[500:]
g4h_lms = np.load('/home/akash/gs/g4s/y_fit.npy', mmap_mode='r')  # already starts at 500

#NEW ERRORS - just use absolute
ehp=np.load('/home/akash/gs/g4s/raw_ehp.npy', mmap_mode='r')  #raw_ehp=np.mean(np.absolute(g4h_hp-g4), axis=0)
esg=np.load('/home/akash/gs/g4s/esg.npy', mmap_mode='r')  #raw_esg=np.mean(np.absolute(g4h_fnsg-g4), axis=0)
elms=np.load('/home/akash/gs//g4s/e_fit.npy', mmap_mode='r')

g4tavg = np.load('/home/akash/gs/calculated/g4t500.npy', mmap_mode='r')  #g4t500=np.absolute(g[:,:,:,:,4]).mean(axis=0)
#enew_avg=np.mean(np.absolute(g4h_lms-g4tavg),axis=0)/(np.mean((np.absolute(g[:,:,:,:,-1])+np.absolute(g4h_lms)),axis=0))
#^ saved as ehalf
ehalf=np.load('ehalf.npy',mmap_mode='r')

#NEW TAVG G[500:] ONLY

#ENERGY
Jmp=np.load('/home/akash/gs/calculated/Jmp.npy', mmap_mode='r')
Jp=np.load('/home/akash/gs/calculated/Jp.npy', mmap_mode='r')
Jm=np.load('/home/akash/gs/calculated/Jm.npy', mmap_mode='r')
energy=np.load('/home/akash/gs/calculated/energy.npy',mmap_mode='r')

#Pg=np.load('/home/akash/gs/calculated/Pg.npy', mmap_mode='r')
ks=[(0,0,0),(0,3,2),(0,6,2),(0,10,2),(0,15,2),(0,25,2)]

