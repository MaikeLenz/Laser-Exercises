from scipy.integrate import odeint
import numpy as np
n=1.82 
sigma_21=2.8*10**(-23) 
r_1=0.999
r_2=0.90 
A=1*10**(-6) 
L=0.1
d=0.2
alpha=0
tau_21=230*10**(6)
K = c*sigma_21/V_c
Q_thresh=1/(K*tau_c*tau_21)
Q_2=2*Q_thresh

M_2=V_g*N_2 
tau_c=d_star/c*(alpha*L-np.log((r_1*r_2)/2)


#want phi(t) (number of photons in cavity)
dphi = K*M_2*[phi+1]−phi/tau_c 