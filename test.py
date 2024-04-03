import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#generate a random direction for a given vector and output the x,y,z components
def vec_gen(r):
  rdm1=np.random.uniform(0,1)
  rdm2=np.random.uniform(0,1)
  theta=np.arccos(2*rdm1-1)
  phi=2*np.pi*rdm2
  x=r*np.cos(phi)*np.sin(theta)
  y=r*np.sin(phi)*np.sin(theta)
  z=r*np.cos(theta)
  return x, y, z, theta, phi
def vec_calc(r,theta,phi):
  x=r*np.cos(phi)*np.sin(theta)
  y=r*np.sin(phi)*np.sin(theta)
  z=r*np.cos(theta)
  return x,y,z

r=1
x=[]
y=[]
z=[]
th=[]
ph=[]
for i in range(0,1000000):
  xi,yi,zi,thi,phi=vec_gen(r)
  x.append(xi)
  y.append(yi)
  z.append(zi)
  th.append(thi)
  ph.append(phi)

plt.figure()
plt.hist2d(th,ph,bins=100)
plt.xlabel('theta')
plt.ylabel('phi')
plt.show()
plt.close()


'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
plt.close()'''