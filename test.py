import matplotlib.pyplot as plt
import numpy as np
import myconst as mc

def bw_func(x,A,a,b):
  q=np.sqrt((x**2-mc.m_n**2-mc.m_pi**2)**2-4*(mc.m_n*mc.m_pi)**2)/(2*x)
  gam=(a*q**3)/(mc.m_pi**2+b*q**2)
  return 100*(4*gam*mc.m_del0**2)/(A*(x**2-mc.m_del0**2)**2+(mc.m_del0*gam)**2)

print(mc.m_n+mc.m_pi)
print(bw_func(1080,0.1,0.47,0.6))
x=np.arange(1000,1400,1)
y=bw_func(x,0.1,0.47,0.6)
print(x[0],y[0])

plt.plot(x,y,".")
plt.show()