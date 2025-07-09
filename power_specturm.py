import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyshtools as pysh
from decimal import Decimal,getcontext
getcontext().prec = 18

mro120f = np.loadtxt('D:\PythonProject\Data\gmm3_120_sha.tab',dtype=np.str_,delimiter=',',skiprows=1)

l = mro120f[:,0].astype(int)
m = mro120f[:,1].astype(int)
C = mro120f[:,2].astype(float)
S = mro120f[:,3].astype(float)

C_Cm_120 = np.zeros((120, 121))
S_Cm_120 = np.zeros((120, 121))
rows = l - 1
cols = m
C_Cm_120[rows, cols] = C
S_Cm_120[rows, cols] = S

Power_spectrum = np.sum(np.square(C_Cm_120) + np.square(S_Cm_120), axis=1)
degrees = np.arange(1, 121)
denominator = 2 * degrees + 1
Pow_var = np.sqrt(Power_spectrum/denominator)
print(Pow_var)
np.savetxt('Gmm3_Pow_var.txt',Pow_var,fmt='%.20f',delimiter='\n',header='Degree Variance (l=1 to 120)',encoding='ascii')

def Clm(l_val, m_val):
    index = int(m_val + l_val * (l_val + 1) / 2)
    return C[index-1]

def Slm(l_val, m_val):
    index = int(m_val + l_val * (l_val + 1) / 2)
    return S[index-1]

print(Clm(1,1))
# Pow_var = np.sqrt((1/(1+l))*Pow_C)
# print(Pow_var)

