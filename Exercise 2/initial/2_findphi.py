from distutils.util import Mixin2to3
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

n=1.82 
sigma_21=2.8*10**(-23) 
r_1=0.999
r_2=0.90 
A=1*10**(-6) 
L=0.1
d=0.2
d_star=d+(n-1)*L
alpha=0
tau_21=230*10**(-6)
c=299792458
V_c=A*d_star
K = c*sigma_21/V_c
tau_c=d_star/c*(alpha*L-np.log((r_1*r_2)/2))
Q_thresh=1/(K*tau_c*tau_21)
Q_2=2*Q_thresh
V_g=A*L*n
#M_2=V_g*N_2 

#want phi(t) (number of photons in cavity)
def odes(y,t,K,tau_c,Q_2,tau_21):
    phi=y[0]
    M_2=y[1]
    dphidt= K*M_2*(phi+1)-phi/tau_c
    dM_2dt= Q_2-K*M_2*phi-M_2/tau_21
    return [dphidt,dM_2dt]

#initial values
M_20=0
phi0=0
y0=[phi0,M_20]

t=np.linspace(0,0.001,100000)
y=odeint(odes,y0,t,args=(K,tau_c,Q_2,tau_21))

"""
phi=y[:,0]
M_2=y[:,1]
"""
phi,M_2=y.T
plt.figure()
plt.plot(t,phi,label="phi")
plt.legend()
plt.xlabel("time,t (s)")

plt.figure()
plt.plot(t,M_2, label="M2")
plt.legend()
plt.xlabel("time,t (s)")

plt.show()


#steady state
#for steady state, oscillations in phi fall below 10% of original amplitude for example?
#or to 1/e**2


#%%

#Extension

def func(x,A,B,C,D,E):
    #oscillations under exponential envelope
    x0=0.0008
    return A*np.exp(-E*(x-x0))*np.cos(B*(x-x0)+C) + D

from scipy.optimize import curve_fit
#consider only beyond T=0.0008
cutindex=np.where(t>0.0008)[0][0]
phi_=phi[cutindex:]
t1=t[cutindex:]
popt,pcov=curve_fit(func,t1,phi_,p0=[0.7*10**11,2000000,0,1.5*10**11,10000])
plt.plot(t1,func(t1,*popt),label="curve fit")
plt.plot(t1,phi_,label="phi")
#plt.plot(t1,func(t1,0.7*10**11,2000000,0,1.5*10**11,10000))
print(popt)
plt.legend()
plt.show()
# %%
