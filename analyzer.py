from matplotlib import pyplot as plt
import csv
import myconst as mc
import numpy as np
from scipy.optimize import curve_fit



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

binsize=1 #in MeV/c^2
plt.figure()
hist,bins,packages=plt.hist(IM_list,bins=np.arange(int(mc.md_min)-1,int(mc.md_max)+1,binsize))
stp=int(mc.m_cut/binsize)
x_omit=int(np.where(bins==mc.m_del0)[0]) #omit the inv mass of delta

#data to be consider for the fitting of the "noise"
x_new=bins[0:x_omit-stp].tolist()+bins[x_omit+stp+1:].tolist()
y_new=hist[0:x_omit-stp].tolist()+hist[x_omit+stp+1:].tolist()
x_start=np.where(hist>0.05*max(hist))[0][0]
x_end=np.where(hist[x_start:]<0.05*max(hist))[0][0]

#data to be considered for counting the number of deltas
x_skipped=bins[x_omit-stp:x_omit+stp+1]
y_skipped=hist[x_omit-stp:x_omit+stp+1]

#fitting
xplt1=np.arange(bin[x_start],bin[x_start+x_end],0.5)
ini_g1=[0,0,0,0,0]
popt1,pcov1=curve_fit(poly_func, x_new,y_new,ini_g1)
yplt1=poly_func(xplt1,*popt1)
r2_poly=r2_calc(poly_func,x_data_new,y_data_new,popt1)
r=str(round(r2_poly,5))
plt.plot(xplt1,yplt1,label='poly fit \n R^2=%s'%(r))



plt.show()
plt.close()