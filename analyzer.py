from matplotlib import pyplot as plt
import csv
import myconst as mc
import numpy as np

#for fitting
def poly_func(x,c0,c1,c2,c3,c4):
  return c0+c1*x+c2*x**2+c3*x**3+c4*x**4

def r2_calc(f,x,y,p):
  res=[]
  ss_res=[]
  ss_tot=[]
  m=np.mean(y)
  for i in range(0,len(x)):
    r=y[i]-f(x[i],*p)
    res.append(r)
    ss_res.append(r**2)
    ss_tot.append((y[i]-m)**2)
  ssr=np.sum(ss_res)
  sst=np.sum(ss_tot)
  return 1-(ssr/sst)


IM_list=[]
with open("data_IM.csv", 'r') as file:
  f=csv.reader(file,delimiter=',')
  for row in f:
    for i in range(0,len(row)):
      IM_list.append(float(row[i]))
  file.close()

plt.figure()
hist,bins,packages=plt.hist(IM_list,bins=np.linspace(mc.md_min,mc.md_max,100))


plt.show()
plt.close()