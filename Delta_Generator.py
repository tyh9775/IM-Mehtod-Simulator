import numpy as np
import random
from scipy.integrate import simpson
import csv
from array import array
import struct

#load constants from the file in the repository
import myconst as mc

#create output files 
with open("data.csv", 'w',newline='') as file:
  file.close()

with open("data_bin.bin",'wb') as fb:
  fb.close()


#Switch for deltas to be generated
Delta=True

#Switch for free particles to be generated with the delta resonances
Free=True 

#should output the PID and 4-momentum of every particle generated 
#header should include the number of events and the total number of particles 


if Delta is False and Free is False:
    print("No particles generated!")
    quit()


#Breit_Wigner distribution for the  mass distribution of delta resonances:
#class bw_dist(st.rv_continuous):
def bw_pdf(md,md0,mn,mpi):
  A=0.95 
  if md==mn+mpi:
    q=0
  elif md<mn+mpi:
    print("invalide mass of delta resonance produced; check mass of nucleon and pion")
    quit()
  else:
    q=np.sqrt((md**2-mn**2-mpi**2)**2-4*(mn*mpi)**2)/(2*md)
  gmd=(0.47*q**3)/(mpi**2+0.6*q**2)
  return (4*md0**2*gmd)/((A)*((md**2-md0**2)**2+md0**2*gmd**2))

#calculate the momentum of delta given the center of collision energy, mdel, and mn
def bw_mom(rs,m1,m2):
  return np.sqrt((rs**2+m1**2-m2**2)**2/(4*rs**2)-m1**2)

#given the momentum and the rest mass, solve for the total energy
def E_solv(p,m):
  return np.sqrt(p**2+m**2)

#given total energy and rest mass, calculate the Lorentz factor and relative velocity
def gam_calc(En,m0):
  gam=En/m0
  v=np.sqrt(1-1/gam**2)
  return gam, v

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

#to generate n particles according to an (a/T)*e^(-x/T) distribution
def en_dist(a,T,n):
  return a*np.random.exponential(scale=1/T,size=n)

#given KE and rest mass, calculate Lorentz factor, rel v, total E, and rel p
def kgam_calc(KE,m0):
  gam=1+KE/m0
  v=np.sqrt(1-1/gam**2)
  Et=KE+m0
  prel=gam*m0*v
  return gam,v,Et,prel



#build mass distribution 
x_bw=np.linspace(mc.md_min,mc.md_max,100)
y_bw=[]
for i in range (0,len(x_bw)):
  y_bw.append(bw_pdf(x_bw[i],mc.m_del0,mc.m_p,mc.m_pi))
norm_const=simpson(y=y_bw,x=x_bw)
y_norm=y_bw/norm_const

#constants for free particle generation 
#KE of pions should be higher than the KE of protons in general since mpi<mp
T1=100
a1=T1*250
T2=100
a2=T2*300

#number of events
N_events=mc.nevts

#event counter
counter=0

#number of created delta resonances
ND_total=0

#number of all particles
NP_total=0


for i in range(0,N_events):
  counter = counter+1
  #number of particles per event
  particles=0

  ################
  #Delta Generator
  ################

  if Delta is True:
    N_delta=2 #number of resonances created per event
    #N_detla should be randomized according to some distribution eventually
    #consider making it scale with the energy of delta
  else:
    N_delta=0
    
    
  if Free is True:
    N_free=10 #number of free particle pairs per event
    #should also be randomized (Boltzmann dist?)
  else:
    N_free=0 
        
  particles=particles+N_delta*2+N_free*2
  NP_total=NP_total+particles

  with open("data_bin.bin",'ab') as fb:
    bheader=array('i',[counter,particles,N_delta])
    bheader.tofile(fb)
    fb.close()

  with open("data.csv", 'a', newline='') as file:
    fw=csv.writer(file,delimiter=',')
    fw.writerow([counter,particles,N_delta])
    file.close()
  for j in range(0,N_delta):
    ND_total=ND_total+1
    ######################################
    #starting in center of collision frame
    ######################################

    #randomly choose the mass of the delta resonance according to bw dist
    #using monte carlo method
    mdel=random.uniform(mc.md_min,mc.md_max)
    ytest=random.uniform(0,max(y_norm))
    while ytest > bw_pdf(mdel,mc.m_del0,mc.m_p,mc.m_pi)/norm_const:
      mdel=random.uniform(mc.md_min,mc.md_max)
      ytest=random.uniform(0,max(y_norm))

    #PID of delta:
    dpid=2224 #delta++

    #calculate the momentum, total energy, and the relative velocity
    pdel=bw_mom(mc.rt_s,mdel,mc.m_p)
    Edel=E_solv(pdel,mdel)
    dgam,dv=gam_calc(Edel,mdel)

    #calculate the IM of generated delta
    md_IM=np.sqrt(Edel**2-pdel**2)

    #give the velocity some direction
    vdx,vdy,vdz,dth,dph=vec_gen(dv)

    pdx,pdy,pdz=vec_calc(pdel,dth,dph)
    datadel=[dpid,Edel,pdx,pdy,pdz,j+1]
    

    ############################################
    #LT to the rest frame of the delta resonance
    ############################################

    #decay the delta into a proton and a pion
    #a->b+c decay
    #can use energy conservation to solve for momentum of b and c in CoM frame
    #E=m_a=sqrt(p^2+mb^2)+sqrt(p^2+mc^2)
    #solved for p using n online algebraic tool

    #momentum of the particles in CoM frame
    pcm=dec_mom_sol(mdel,mc.m_p,mc.m_pi)

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
    
    with open('data.csv','a',newline='') as file:
      g=csv.writer(file, delimiter=',')
      g.writerow(datadel)
      g.writerow(datap)
      g.writerow(datapi)
      file.close()
    #in binary
    with open("data_bin.bin",'ab') as fb:
      bdata=array('f',datap+datapi)
      bdata.tofile(fb)
      fb.close()    



  ########################
  #Free particle generator
  ########################
        
  #In lab frame

  #generate the kinetic energy of the particles in lab frame
  p_k=en_dist(a1,T1,N_free)
  pi_k=en_dist(a2,T2,N_free)  
  
  for k in range(0,len(p_k)):
    #calculate L factor, rel v, total E, and rel momenta of p an pi
    gam1,v1,pEt,ppr=kgam_calc(p_k[k],mc.m_p)
    gam2,v2,piEt,pipr=kgam_calc(pi_k[k],mc.m_pi)


    #give the particles a direction and write the 4 momenta
    pxp,pyp,pzp,th_p,ph_p=vec_gen(ppr)
    p4pf=[pEt,pxp,pyp,pzp]
    pxpi,pypi,pzpi,th_pi,ph_pi=vec_gen(pipr) 
    p4pif=[piEt,pxpi,pypi,pzpi]

    datap=[2212]
    datapi=[211]
    for k in range(0, len(p4pf)):
      datap.append(p4pf[k])
      datapi.append(p4pif[k])
    
    datap.append(0)
    datapi.append(0)
          
    with open('data.csv','a',newline='') as file:
      g=csv.writer(file, delimiter=',')
      g.writerow(datap)
      g.writerow(datapi)
      file.close()

    #in binary
    with open("data_bin.bin",'ab') as fb:
      bdata=array('f',datap+datapi)
      bdata.tofile(fb)
      fb.close()    
    


check=True

if check:
  print("Number of Delta resonances created:",ND_total)
  print("Number of all particles detected:", NP_total)
