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
tau_21=230*10**(6)
c=3*10**8
V_c=A*d_star
K = c*sigma_21/V_c
tau_c=d_star/c*(alpha*L-np.log((r_1*r_2)/2))
Q_thresh=1/(K*tau_c*tau_21)
Q_2=2*Q_thresh
V_g=A*L*n
#M_2=V_g*N_2 

#want phi(t) (number of photons in cavity)
def odes(y,t):
    phi=y[0]
    M_2=y[1]
    dphidt= K*M_2(t)*(phi+1)-phi/tau_c
    dM_2dt= Q_2-K*M_2*phi(t)-M_2/tau_21
    return [dphidt,dM_2dt]

#initial values
M_20=0
phi0=0
y0=[phi0,M_20]

t=np.linspace(0,100,10000)
y=odeint(odes,y0,t)

plt.plot(t,0)