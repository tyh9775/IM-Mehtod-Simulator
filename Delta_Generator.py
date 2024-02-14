import numpy as np
import random
from scipy.integrate import simpson
import csv
from array import array

#load constants from the file in the repository
import myconst as mc

#create an output file 
'''
with open("data.csv", 'w') as file:
  file.close()
'''
with open("data1.csv",'w') as f:
  f.close()

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
def bw_mom(s,m1,m2):
  return np.sqrt((s**2+m1**2-m2**2)**2/(4*s**2)-m1**2)

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


#bounds for the Breit-Wigner distribution of the delta mass in lab frame
md_min=mc.m_p+mc.m_pi
md_max=mc.rt_s-mc.m_p 

#build mass distribution 
x_bw=np.linspace(md_min,md_max,100)
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
N_events=1

#number of created delta resonances
N_total=0



for i in range(0,N_events):
  ################
  #Delta Generator
  ################

  if Delta is True:
    N_delta=2 #number of resonances created per event
    #N_detla should be randomized according to some distribution eventually
    #consider making it scale with the energy of delta
    
    N_total=N_total+N_delta

    for j in range(0,N_delta):
      N_total=N_total+1
      ######################################
      #starting in center of collision frame
      ######################################

      #randomly choose the mass of the delta resonance according to bw dist
      #using monte carlo method
      mdel=random.uniform(md_min,md_max)
      ytest=random.uniform(0,max(y_norm))
      while ytest > bw_pdf(mdel,mc.m_del0,mc.m_p,mc.m_pi)/norm_const:
        mdel=random.uniform(md_min,md_max)
        ytest=random.uniform(0,max(y_norm))

      #calculate the momentum, total energy, and the relative velocity
      pdel=bw_mom(mc.rt_s,mdel,mc.m_p)
      Edel=E_solv(pdel,mc.m_del0)
      dgam,dv=gam_calc(Edel,mc.m_del0)
      
      #give the velocity some direction
      vdx,vdy,vdz,dth,dph=vec_gen(dv)

      ############################################
      #LT to the rest frame of the delta resonance
      ############################################

      #decay the delta into a proton and a pion
      #a->b+c decay
      #can use energy conservation to solve for momentum of b and c in CoM frame
      #E=m_a=sqrt(p^2+mb^2)+sqrt(p^2+mc^2)
      #solved for p using n online algebraic tool

      #momentum of the particles in CoM frame
      pcm=dec_mom_sol(mc.m_del0,mc.m_p,mc.m_pi)
      #(use mdel instead?)

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
      '''
      with open('data.csv','a',newline='') as file:
        g=csv.writer(file, delimiter=',')
        g.writerow(p4pL)
        g.writerow(p4piL)
        file.close()
      '''
      #Invariant mass as PID
      pdata1=[mc.m_p,p4pL[1],p4pL[2],p4pL[3]]
      pidata1=[mc.m_pi,p4piL[1],p4piL[2],p4piL[3]]

      with open("data1.csv",'a',newline='') as f:
        fw=csv.writer(f,delimiter=',')
        fw.writerow(pdata1)
        fw.writerow(pidata1)
        f.close()

      #in binary
      with open("data_bin.bin",'ab') as fb:
        bdata=array('f',pdata1+pidata1)
        bdata.tofile(fb)
        fb.close()


  ########################
  #Free particle generator
  ########################
        
  #In lab frame

  if Free is True:
    N_free=2 #number of free particle pairs per event
    #should also be randomized (Boltzmann dist?)

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

      '''
      with open('data.csv','a',newline='') as file:
        g=csv.writer(file, delimiter=',')
        g.writerow(p4pf)
        g.writerow(p4pif)
        file.close()
      '''
      #Invariant mass as PID
      pfdata1=[mc.m_p,p4pf[1],p4pf[2],p4pf[3]]
      pifdata1=[mc.m_pi,p4pif[1],p4pif[2],p4pif[3]]      

      with open("data1.csv",'a',newline='') as f:
        fw=csv.writer(f,delimiter=',')
        fw.writerow(pfdata1)
        fw.writerow(pifdata1)
        f.close()

      #in binary
      with open("data_bin.bin",'ab') as fb:
        bdata=array('f',pfdata1+pifdata1)
        bdata.tofile(fb)
        fb.close()

check=True

if check:
  print("Number of Delta resonances created:",N_total)


with open('data1.csv','r') as f:
  lines=f.readlines()
  f.close()

with open('data1s.csv','w') as g:
  random.shuffle(lines)
  g.writelines(lines)
  g.close()