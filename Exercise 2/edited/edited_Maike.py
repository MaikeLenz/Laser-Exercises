from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# time axis
t = np.linspace(0, 2e-3, 10000) # sec
def odes(y,t,K,tau_c,Q2,tau_21):
    phi=y[0]
    M_2=y[1]
    dphidt= K*M_2*(phi+1)-phi/tau_c
    dM_2dt= Q2-K*M_2*phi-M_2/tau_21
    return [dphidt,dM_2dt]

#initial values
M_20=0
phi0=0
y0=[phi0,M_20]

#solve odes
y=odeint(odes,y0,t,args=(K,tau_c,Q2,tau_21))

#unpack results
phi,M_2=y.T

#plot
plt.figure()
plt.plot(t,phi,label="phi")
plt.legend()
plt.xlabel("time,t (s)")

plt.figure()
plt.plot(t,M_2, label="M2")
plt.legend()
plt.xlabel("time,t (s)")

plt.show()


#%%

from scipy.signal import find_peaks
peaks,_ = find_peaks(phi)
phi_peaks = []
t_peaks = []
for i in range(len(peaks)):
    phi_peaks.append(phi[peaks[i]])
    t_peaks.append(t[peaks[i]])

phi_peaks=np.array(phi_peaks)
t_peaks=np.array(t_peaks)


plt.figure()
plt.plot(t,phi,label="phi")
plt.plot(t_peaks,phi_peaks, label="peaks")
plt.legend()
plt.xlabel("time,t (s)")
plt.show()


#%%
from scipy.optimize import curve_fit

def decay(x,A,B,C):
    #exponential decay
    return A*np.exp(-B*x)+C

popt,pcov=curve_fit(decay,t_peaks,phi_peaks,p0=[4*10**12,3000,0])
plt.plot(t_peaks,decay(t_peaks,*popt),label="curve fit")
plt.plot(t_peaks,phi_peaks,label="phi peaks")
plt.show()
print(popt)