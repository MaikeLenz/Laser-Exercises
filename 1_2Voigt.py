from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import random


def width_Voigt(dwL,dwG):
    #returns width ofVoigt profile from Lorentzian with width dwL and Gaussian with width dwG
    return 0.5*dwL**2 + np.sqrt(0.2*dwL**2 + dwG**2)

def Gaussian(w,w0,dw):
    return ((2*np.sqrt(np.log(2)/np.pi))/dw)*np.exp(-4*np.log(2)*(w-w0)**2/dw**2)
    
def Lorentzian(w,w0,dw):   
    return (2/(np.pi*dw))*(dw**2)/(dw**2+4*(w-w0)**2)

dwG=0.5
dwL=0.7
w=np.linspace(0,100,10000) 
gauss=Gaussian(w,50,dwG)
lor=Lorentzian(w,60,dwL)

"""
plt.plot(w,gauss,label="gauss")
plt.plot(w,lor,label="lorentz")
plt.xlabel("w")
plt.ylabel("lineshape")
plt.legend()
plt.show()
"""

#convolution
conv= signal.convolve(lor, gauss, mode='same')

#normalise
norm = np.linalg.norm(conv)
conv = conv/norm

"""
plt.plot(w,gauss,label="gauss")
plt.plot(w,lor,label="lorentz")
plt.plot(w,conv,label="convolution")
plt.xlabel("w")
plt.ylabel("lineshape")
plt.legend()
plt.show()
"""

def FWHM(x,y):
    d = y - (max(y) / 2) 
    indexes = np.where(d > 0)[0] 
    return abs(x[indexes[-1]] - x[indexes[0]])

print(width_Voigt(dwL,dwG))
print(FWHM(w,conv))

#extension
widths=[]
fwhms=[]
for i in range(200):
    dwL=random.random()
    dwG=random.random()
    gauss=Gaussian(w,50,dwG)
    lor=Lorentzian(w,60,dwL)
    conv= signal.convolve(lor, gauss, mode='same')
    widths.append(width_Voigt(dwL,dwG))
    fwhms.append(FWHM(w,conv))


plt.scatter(np.array(fwhms),np.array(widths))
plt.ylabel("Voigt width")
plt.xlabel("FWHM convolution")
plt.show()


