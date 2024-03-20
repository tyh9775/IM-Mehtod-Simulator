import myconst as mc
import numpy as np
from matplotlib import pyplot as plt


#to generate n particles according to an (a/T)*e^(-x/T) distribution
def en_dist(a,T,n):
  return a*np.random.exponential(scale=1/T,size=n)


y=[]
for i in range(0,10000):
  y.append(en_dist(10000,100,1)[0])

plt.figure()
hist,b,p=plt.hist(y,bins=np.arange(0,1001,10))
plt.show()
plt.close()

print(np.sum(hist))