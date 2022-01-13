import numpy as np

def get_width(x,y,frac=2):
    #default is frac=2 which is fwhm width
    d = y - (max(y) / frac) 
    indices = np.where(d > 0)[0] 
    return abs(x[indices[-1]] - x[indices[0]])

x=np.linspace(-50,50,10000)
def f(x):
    return np.exp(-4*np.log(2)*x**2)
y=f(x)
print(get_width(x,y)) #returns 0.9900990099009945