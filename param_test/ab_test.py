import numpy as np
import random
from scipy.integrate import simpson
import csv
import myconst as mc
from matplotlib import pyplot as plt
import os


#Breit_Wigner distribution for the  mass distribution of delta resonances:
#class bw_dist(st.rv_continuous):
def bw_pdf(md,md0,mn,mpi,A=None,a=None,b=None):
  if A is None:
    A=0.95
  if a is None:
    a=0.47
  if b is None:
    b=0.6 
  if md==mn+mpi:
    q=0
  elif md<mn+mpi:
    print("invalide mass of delta resonance produced; check mass of nucleon and pion")
    quit()
  else:
    q=np.sqrt((md**2-mn**2-mpi**2)**2-4*(mn*mpi)**2)/(2*md)
  gmd=(a*q**3)/(mpi**2+b*q**2)
  
  return (4*md0**2*gmd)/(A*((md**2-md0**2)**2+md0**2*gmd**2))


#distance formula: sqrt(x1^2+x2^2+...+xn^2)
def dist_form(vec):
  vsum2=0
  for i in range(0,len(vec)):
    vsum2+=float(vec[i])**2
  return np.sqrt(vsum2)  

#sum of two vectors
def v_sum(v1,v2):
  vtot=[]
  for i in range (0,len(v1)):
    vtot.append(v1[i]+v2[i]) 
  return vtot

#find the invariant mass given the total energy and the momentum
def inv_m(en,p):
  return np.sqrt(en**2-p**2)

#for varying rs (center of collision energy)
#0<sig<20
def sqrt_s(sig):
  return 1000*(2.015+np.sqrt((0.015*sig)/(20-sig)))

#given the momentum and the rest mass, solve for the total energy
def E_solv(p,m):
  return np.sqrt(p**2+m**2)

#given total energy and rest mass, calculate the Lorentz factor and relative velocity
def gam_calc(En,m0):
  gam=En/m0
  v=np.sqrt(1-1/gam**2)
  return gam, v

#given KE and rest mass, calculate Lorentz factor, rel v, total E, and rel p
def kgam_calc(KE,m0):
  gam=1+KE/m0
  v=np.sqrt(1-1/gam**2)
  Et=KE+m0
  prel=gam*m0*v
  return gam,v,Et,prel

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

#given a->b+c decay, solves for p of b and c
def dec_mom_sol(m0,m1,m2):
  return np.sqrt(m0**4-2*(m0*m1)**2-2*(m0*m2)**2+m1**4-2*(m1*m2)**2+m2**4)/(2*m0)

#LT in 3D space in matrix form
def gam_mat(gam,v,vx,vy,vz,p4):
    A=np.array([[gam,-gam*vx,-gam*vy,-gam*vz],
                [-gam*vx,1+(gam-1)*vx**2/v**2,(gam-1)*vx*vy/v**2,(gam-1)*vx*vz/v**2],
                [-gam*vy,(gam-1)*vx*vy/v**2,1+(gam-1)*vy**2/v**2,(gam-1)*vy*vz/v**2],
                [-gam*vz,(gam-1)*vx*vz/v**2,(gam-1)*vy*vz/v**2,1+(gam-1)*vz**2/v**2]])
    return np.dot(A,p4)

#to generate n particles according to an exponential distribution
def exp_dist(scl,n):
  return np.random.exponential(scale=scl,size=n)


def simp_gen(numD,numF,DT,A=None,a=None,b=None):
  p_list=[]
  pi_list=[]
  #build mass distribution 
  x_bw=np.linspace(mc.md_min,mc.md_max,100)
  y_bw=[]
  for i in range(0,len(x_bw)):
    y_bw.append(bw_pdf(x_bw[i],mc.m_del0,mc.m_p,mc.m_pi,A=A,a=a,b=b))
  norm_const=simpson(y=y_bw,x=x_bw)
  y_norm=y_bw/norm_const

  if numD==0 and numF==0:
    print("numD,numF:",0,0)
    print("No particles detected")
    print()
    return
  
  N_events=mc.nevts

  for i in range(0,N_events):
    N_delta=numD
    N_free=numF
        
    for j in range(0,N_delta):

      ######################################
      #starting in center of collision frame
      ######################################

      #randomly choose the mass of the delta resonance according to bw dist
      #using monte carlo method
      mdel=random.uniform(mc.md_min,mc.md_max)
      ytest=random.uniform(0,max(y_norm))
      while ytest > bw_pdf(mdel,mc.m_del0,mc.m_p,mc.m_pi,A=A,a=a,b=b)/norm_const:
        mdel=random.uniform(mc.md_min,mc.md_max)
        ytest=random.uniform(0,max(y_norm))
      #PID of delta:
      dpid=2224 #delta++
      
      #give delta a random momentum
      ke_del=exp_dist(DT,1)[0]
      dgam,dv,Edel,pdel=kgam_calc(ke_del,mdel)

      #md_IM=np.sqrt(Edel**2-pdel**2)

      #give the velocity some direction
      vdx,vdy,vdz,dth,dph=vec_gen(dv)

      pdx,pdy,pdz=vec_calc(pdel,dth,dph)
      datadel=[dpid,Edel,pdx,pdy,pdz,j+1]

      ############################################
      #LT to the rest frame of the delta resonance
      ############################################

      #momentum of the particles in CoM frame (decay equation solved with algebraic solver)
      pcm=dec_mom_sol(mdel,mc.m_p,mc.m_pi)

      #momentum of the pion in the delta frame according to bw_dist
      #pcm=q_solv(mdel,mc.m_p,mc.m_pi)
      #the same as before after checking

      #total energy of each particle in CoM frame
      Ep=E_solv(pcm,mc.m_p)
      Epi=E_solv(pcm,mc.m_pi)

      #give the proton and pion momenta direction in the delta frame
      ppx,ppy,ppz,pth,pph=vec_gen(pcm)

      #write the 4 momenta of p and pi in delta frame
      p4pD=[Ep,ppx,ppy,ppz]
      p4piD=[Epi,-ppx,-ppy,-ppz]

      #####################
      #LT back to lab frame
      #####################

      #4 momenta of p and pi in lab frame and use write to output file
      p4pL=gam_mat(dgam,dv,-vdx,-vdy,-vdz,p4pD)
      p4piL=gam_mat(dgam,dv,-vdx,-vdy,-vdz,p4piD)

      datap=[2212]
      datapi=[211]
      for k in range(0, len(p4pL)):
        datap.append(p4pL[k])
        datapi.append(p4piL[k])
      #give "parent" particle data
      datap.append(j+1)
      datapi.append(j+1)
      
      pi_list.append(datapi)
      p_list.append(datap)
    ########################
    #Free particle generator
    ########################
          
    #In lab frame

    #generate the momenta of the particles in lab frame according to exp dist
    ke_N=exp_dist(150,N_free)
    ke_Pi=exp_dist(200,N_free)  

    pN=[]
    pPi=[]
    for k in range(0,len(ke_N)):
      pN.append(kgam_calc(ke_N[k],mc.m_p)[3])
      pPi.append(kgam_calc(ke_Pi[k],mc.m_pi)[3])
    for k in range(0,len(pN)):
      #give the particles a direction and write the 4 momenta
      pxp,pyp,pzp,th_p,ph_p=vec_gen(pN[k])
      pxpi,pypi,pzpi,th_pi,ph_pi=vec_gen(pPi[k]) 
      Etp=E_solv(pN[k],mc.m_p)
      Etpi=E_solv(pPi[k],mc.m_pi)
      p4pf=[Etp,pxp,pyp,pzp]
      p4pif=[Etpi,pxpi,pypi,pzpi]

      datap=[2212]
      datapi=[211]
      for kk in range(0, len(p4pf)):
        datap.append(p4pf[kk])
        datapi.append(p4pif[kk])
      
      datap.append(0)
      datapi.append(0)
      pi_list.append(datapi)
      p_list.append(datap)
  return p_list, pi_list

def IM_method_simp(plist,pilist):
  mdel=0
  mntdel=0
  Ep=float(plist[0]) #total E of p
  Epi=float(pilist[0]) #total E of pi
  pp=[]
  ppi=[]
  for k in range(0,3):
    pp.append(float(plist[k+1])) #3D momentum of p
    ppi.append(float(pilist[k+1])) #3D momentum of pi
  ptot=v_sum(pp,ppi) #total momentum of the two particles
  pmag=dist_form(ptot) #magnitude of the total momentum
  Etot=Ep+Epi #total energy of the two particles
  p4p=[Ep]+pp #4 momentum of p 
  p4pi=[Epi]+ppi #4 momentum of pi    
  mdel_rec=inv_m(Etot,pmag)
  gam=Etot/mdel_rec
  if gam<1:
    print("negative energy or mass detected")
    quit()
  v=np.sqrt(1-1/gam**2)

  vx=ptot[0]/(gam*mdel_rec)
  vy=ptot[1]/(gam*mdel_rec)
  vz=ptot[2]/(gam*mdel_rec)
  #move to the "delta" frame assuming the pair can create one
  ptest=gam_mat(gam,v,vx,vy,vz,p4p) #4 momentum of p in delta frame
  pitest=gam_mat(gam,v,vx,vy,vz,p4pi) #4 momentum of pi in delta frame
  pt_tot=v_sum(ptest,pitest) #total 4 momentum of p and pi in delta frame
  pt_mag=dist_form(pt_tot[1:]) #magnitude of the 3D momentum in delta frame
  #momentum cut
  if pt_mag < mc.p_cut:        
    mdel=mdel_rec
    mntdel=pmag
  return mdel, mntdel

def row_read(data):
  PID=data[0]
  p4p=[]
  pi4p=[]
  if PID==2212:
    p4p=data[1:5]
    return p4p
  if PID==211:
    pi4p=data[1:5]
    return pi4p
  else:
    return
    
def simp_reader(p_all,pi_all):
  m_list=[]
  mnt_list=[]
  for i in range(0,len(pi_all)):
    p4p=row_read(pi_all[i])
    pi4p=row_read(p_all[i])
    mdel,mntdel=IM_method_simp(p4p,pi4p)
    m_list.append(mdel)
    mnt_list.append(mntdel)
  
  return m_list, mnt_list


#numbers of particles/pairs generated
Delta_num=[1,2,3]
Free_num=[0,1,2,3]

#set constants/parameters
delta_temp=300
#fix A to 0.95 (leave A=None)
a=np.random.uniform(0.1,2)
b=np.random.uniform(0.1,2)

plist1,pilist1=simp_gen(1,0,300,a=a,b=b)
mlist1,mntlist1=simp_reader(plist1,pilist1)


plt.figure()
plt.hist(mlist1,bins=100)
plt.show()
plt.close()

'''
N_samples=10000
a_list=[]
b_list=[]
m_val=[]
for n in range(0,N_samples):
  a=np.random.uniform(0.1,2)
  b=np.random.uniform(0.1,2)
  a_list.append(a)
  b_list.append(b)
  x_bw=np.linspace(mc.md_min,mc.md_max,100)
  y_bw=[]
  for i in range(0,len(x_bw)):
    y_bw.append(bw_pdf(x_bw[i],mc.m_del0,mc.m_p,mc.m_pi,a=a,b=b))
  norm_const=simpson(y=y_bw,x=x_bw)
  y_norm=y_bw/norm_const
  mdel=random.uniform(mc.md_min,mc.md_max)
  ytest=random.uniform(0,max(y_norm))
  while ytest > bw_pdf(mdel,mc.m_del0,mc.m_p,mc.m_pi,a=a,b=b)/norm_const:
    mdel=random.uniform(mc.md_min,mc.md_max)
    ytest=random.uniform(0,max(y_norm))
  m_val.append(mdel)


cor=np.corrcoef(a_list,b_list)[0,1]

print("a,b correlation:",cor)

plt.figure()
plt.hist(m_val,bins=100)
plt.show()
plt.close()

a_flat=np.repeat(a_list,len(b_list))
b_flat=np.tile(b_list,len(a_list))

plt.figure()
plt.scatter(a_flat, b_flat, c=f_values)
plt.colorbar(label='Mean f value')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Correlation Graph between a and b')
plt.grid(True)
plt.show()

'''
