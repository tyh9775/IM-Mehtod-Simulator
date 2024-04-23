import numpy as np
import matplotlib.pyplot as plt
import myconst as mc

def bw_func(x,A,a,b,scl):
  q=np.sqrt((x**2-mc.m_p**2-mc.m_pi**2)**2-4*(mc.m_p*mc.m_pi)**2)/(2*x)
  gam=(a*q**3)/(mc.m_pi**2+b*q**2)
  return scl*(4*gam*mc.m_del0**2)/(A*((x**2-mc.m_del0**2)**2+(mc.m_del0*gam)**2))

def scl_calc(y0,y):
  return max(y)/max(y0)
x=np.arange(mc.md_min,mc.md_max,0.5)
y=bw_func(x,0.95,0.47,0.6,1)
y1=bw_func(x,0.95,0.47,0.6,10)
y2=bw_func(x,0.95,0.47,0.6,100)
plt.plot(x,y,label='1')
plt.plot(x,y1,label='10')
plt.plot(x,y2,label='100')
plt.legend()
plt.show()
print(max(y))
print(max(y1))
print(max(y2))

print(scl_calc(y,y1))