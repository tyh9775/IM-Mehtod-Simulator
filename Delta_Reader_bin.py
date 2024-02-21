import numpy as np
import myconst as mc
import csv
from array import array
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def bin_readheader(file,hsize):
  f=array('i')
  f.frombytes(file.read(hsize))
  g=list(f)
  return g

def bin_readline(file,bin_size):
  f=array('f')
  f.frombytes(file.read(bin_size))
  g=list(f)
  return g


#distance formula: sqrt(x1^2+x2^2+...+xn^2)
def dist_form(vec):
  vsum2=0
  for i in range(0,len(vec)):
    vsum2+=vec[i]**2
  return np.sqrt(vsum2)  

#sum of two vectors
def v_sum(v1,v2):
  vtot=[]
  for i in range (0,len(v1)):
    vtot.append(v1[i]+v2[i]) 
  return vtot

#given total energy and rest mass, calculate the Lorentz factor and relative velocity
def gam_calc(En,m0):
  gam=En/m0
  return gam

#LT in 3D space in matrix form
def gam_mat(gam,v,vx,vy,vz,p4):
    A=np.array([[gam,-gam*vx,-gam*vy,-gam*vz],
                [-gam*vx,1+(gam-1)*vx**2/v**2,(gam-1)*vx*vy/v**2,(gam-1)*vx*vz/v**2],
                [-gam*vy,(gam-1)*vx*vy/v**2,1+(gam-1)*vy**2/v**2,(gam-1)*vy*vz/v**2],
                [-gam*vz,(gam-1)*vx*vz/v**2,(gam-1)*vy*vz/v**2,1+(gam-1)*vz**2/v**2]])
    return np.dot(A,p4)

#find the invariant mass given the total energy and the momentum
def inv_m(en,p):
  m2=en**2-p**2
  return np.sqrt(m2)

def IM_method(plist,pilist,pcut):
  m_list=[]
  for i in range(0,len(pilist)):
    for j in range(0,len(plist)):
      Ep=float(plist[j][0]) #total E of p
      Epi=float(pilist[i][0]) #total E of pi
      pp=[]
      ppi=[]
      for k in range(0,3):
        pp.append(float(plist[j][k+1])) #3D momentum of p
        ppi.append(float(pilist[i][k+1])) #3D momentum of pi
      ptot=v_sum(pp,ppi) #total momentum of the two particles
      pmag=dist_form(ptot) #magnitude of the total momentum
      Etot=Ep+Epi #total energy of the two particles
      
      #move to the "delta" frame assuming the pair can create one
      gam=gam_calc(Etot,mc.m_del0)
      if gam<1:
        continue
      v=np.sqrt(1-1/gam**2)

      vx=ptot[0]/(gam*mc.m_del0)
      vy=ptot[1]/(gam*mc.m_del0)
      vz=ptot[2]/(gam*mc.m_del0)

      p4p=[Ep]+pp #4 momentum of p in lab frame
      p4pi=[Epi]+ppi #4 momentum of pi in lab frame
      ptest=gam_mat(gam,v,vx,vy,vz,p4p) #4 momentum of p in delta frame
      pitest=gam_mat(gam,v,vx,vy,vz,p4pi) #4 momentum of pi in delta frame
      pt_tot=v_sum(ptest,pitest)
      pt_mag=dist_form(pt_tot[1:])
      

      #momentum cut
      if pt_mag < pcut:
        m_list.append(inv_m(Etot,pmag))
  return m_list

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

hsize=8
buffer_size=16
with open("data_bin.bin",'rb') as file:
  for row in file:
    #momenta of protons and pions
    p_list=[]
    pi_list=[]
    header=bin_readheader(file,hsize)
    print(header)
    eNum=int(header[0])
    pNum=int(header[1])
    for i in range(0,pNum):
      f=bin_readline(file,buffer_size)
      PID=f[0]
      if PID==2212:
        p_list.append()
      elif PID==211:
        pi_list.append(f[1:])

  file.close()




#graphing and fitting
#mass cut done with the fitting
binsize=1 #in MeV/c^2
plt.figure()
hist,bins,packages=plt.hist(IM_list,bins=np.arange(int(mc.md_min)-1,int(mc.md_max)+1,binsize))
stp=int(mc.m_cut/binsize) #determines the width of the cut
x_omit=int(np.where(bins==mc.m_del0)[0][0]) #omit the inv mass of delta
#data to be consider for the fitting of the "noise"
x_start=np.where(hist>0.05*max(hist))[0][0]
x_end=np.where(hist[x_start:]<0.05*max(hist))[0][0]
x_new=bins[x_start:x_omit-stp].tolist()+bins[x_omit+stp+1:x_start+x_end].tolist()
y_new=hist[x_start:x_omit-stp].tolist()+hist[x_omit+stp+1:x_start+x_end].tolist()
#data to be considered for counting the number of deltas
x_skipped=bins[x_omit-stp:x_omit+stp]
y_skipped=hist[x_omit-stp:x_omit+stp]
print(x_skipped)
#fitting
xplt=np.arange(bins[x_start],bins[x_start+x_end],0.5)
ini_g=[0,0,0,0,0]
popt,pcov=curve_fit(poly_func, x_new,y_new,ini_g)
yplt=poly_func(xplt,*popt)
r2_poly=r2_calc(poly_func,x_new,y_new,popt)
r=str(round(r2_poly,5))
plt.plot(xplt,yplt,label='poly fit \n R^2=%s'%(r))
plt.plot(x_skipped,y_skipped,'.')
#plt.plot(x_skipped,poly_func(x_skipped,*popt),'.')
#guessing count
y_est=[]
for i in range(0,len(x_skipped)):
  xi=np.where(xplt==x_skipped[i])[0][0]
  y_est.append(y_skipped[i]-yplt[xi])

print("estimated number of deltas:",sum(y_est))

plt.title("Invariant Mass of Proton and Pion Pairs in Lab Frame")
plt.ylabel("Count")
plt.xlabel("Mass (MeV/c^2)")
plt.legend(loc='upper right')
plt.ylim(0,max(hist)*1.1)
plt.figtext(0.75,0.65,"m_err=%d \n p_min=%d"%(mc.m_cut,mc.p_cut),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
plt.savefig("IM_pairs.png")
plt.show()
plt.close()
