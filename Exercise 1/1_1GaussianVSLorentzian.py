import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

#%%
def gaussian(x, u, w):
    return (2*np.sqrt(np.log(2)/np.pi)/w)*np.exp(-4*np.log(2)*((x-u)**2)/(w**2))

def lorentzian(x, u, w):
    return (2/(np.pi*w))*(w**2)/((w**2)+4*((x-u)**2))
#%%
x = np.arange(0, 100, 1)

plt.figure()
plt.plot(x, gaussian(x, 50, 18), color='tab:blue', label='Gaussian')
plt.plot(x, lorentzian(x, 50, 18), color='tab:red', label='Lorentzian')
plt.legend()

plt.figure()
plt.plot(x, gaussian(x, 50, 18), color='tab:blue', label='Gaussian')
plt.plot(x, lorentzian(x, 50, 18), color='tab:red', label='Lorentzian')
plt.yscale('log')
plt.legend()
#%%
def get_width(x, y, fraction=0.5):
    h = max(y)*fraction
    min_indices = []
    for i in range(len(y)-2):
        if abs(y[i]-h) >= abs(y[i+1]-h) and abs(y[i+1]-h) <= abs(y[i+2]-h):
            min_indices.append(i+1)
    if len(min_indices) != 2:
        print('Failed')
    width = abs(x[min_indices[0]] - x[min_indices[1]])
    return width
#%%
print(get_width(x, gaussian(x, 50, 18)))
print(get_width(x, lorentzian(x, 50, 18)))
#%%
print(quad(gaussian, 0, 100, args=(50, 18)))
print(quad(lorentzian, 0, 100, args=(50, 18)))
