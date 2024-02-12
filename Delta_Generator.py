import numpy as np
import random
from scipy.integrate import simpson


#Switch for deltas to be generated
Delta=True

#Switch for free particles to be generated with the delta resonances
Free=False 

#should output the PID and 4-momentum of every particle generated 
#header should include the number of events and the total number of particles 


if Delta is False and Free is False:
    print("No particles generated!")
    quit()


#Breit_Wigner distribution for the  mass distribution of delta resonances:
#class bw_dist(st.rv_continuous):
def bw_pdf(md,md0,mn,mpi):
  A=0.95 
  q=np.sqrt((md**2-mn**2-mpi**2)**2-4*(mn*mpi)**2)/(2*md)
  gmd=(0.47*q**3)/(mpi**2+0.6*q**2)
  return (4*md0**2*gmd)/((A)*((md**2-md0**2)**2+md0**2*gmd**2))

#calculate the momentum of delta given the center of collision energy, mdel, and mn
def bw_mom(s,m1,m2):
  return np.sqrt((s**2+m1**2-m2**2)**2/(4*s**2)-m1**2)

#given the momentum and the rest mass, solve for the kinetic energy and the total energy
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

#distance formula: sqrt(x1^2+x2^2+...+xn^2)
def dist_form(vec):
  vsum2=0
  for i in range(0,len(vec)):
    vsum2+=vec[i]**2
  return np.sqrt(vsum2)  

#given two 4-vectors, calculate the sum of the spatial components
def p_sum(p4a,p4b):
  ptot=[]
  for i in range (0,3):
    ptot.append(p4a[i+1]+p4b[i+1]) 
  return ptot

#to generate n particles according to an (a/T)*e^(-x/T) distribution
def en_dist(a,T,n):
  return a*np.random.exponential(scale=1/T,size=n)

#constant values
m_del0=1232 #MeV/c^2 - rest mass of delta resonance
m_p=938 #MeV/c^2 - rest mass of proton
m_pi=139.570 #MeV/c^2 - rest mass of charged pion
Eb=270 #Beam energy per nucleon (AMeV)
rt_s=2015+Eb #energy at the center of collision (sqrt of s)
#2015 MeV is the minimum C.M. energy of the collision for the energy-dependent isospin-averaged isotropic cross section to be non-zero
#may have to come up with a more dynamic way to adjust rt_s


#bounds for the Breit-Wigner distribution of the delta mass in lab frame
md_min=m_p+m_pi
md_max=rt_s-m_p 

#build mass distribution 
x_bw=np.linspace(md_min,md_max,1000)
y_bw=[]
for i in range (0,len(x_bw)):
  y_bw.append(bw_pdf(x_bw[i],m_del0,m_p,m_pi))
norm_const=simpson(y=y_bw,x=x_bw)
y_norm=y_bw/norm_const

#constants for free particle generation 
#KE of pions should be higher than the KE of protons in general since mpi<mp
T1=100
a1=T1*250
T2=100
a2=T2*300

#number of events
N_events=10000

#number of created delta resonances
N_total=0

p_list=[]
pi_list=[]

for i in range(0,N_events):
  ################
  #Delta Generator
  ################

  if Delta is True:
    N_delta=2 #number of resonances created per event
    #N_detla should be randomized according to some distribution eventually
    #consider making it scale with the energy of delta
    
    for i in range(0,N_delta):
      ######################################
      #starting in center of collision frame
      ######################################

      #randomly choose the mass of the delta resonance according to bw dist
      #using monte carlo method
      mdel=random.uniform(md_min,md_max)
      ytest=random.uniform(0,max(y_norm))
      while ytest > bw_pdf(mdel,m_del0,m_p,m_pi)/norm_const:
        mdel=random.uniform(md_min,md_max)
        ytest=random.uniform(0,max(y_norm))

      #calculate the momentum, total energy, and the relative velocity
      pdel=bw_mom(rt_s,mdel,m_p)
      Edel=E_solv(pdel,m_del0)
      dgam,dv=gam_calc(Edel,m_del0)
      
      #give the velocity some direction
      vdx,vdy,vdz,dth,dph=vec_gen(dv)

      ############################################
      #LT to the rest frame of the delta resonance
      ############################################

      #decay the delta into a proton and a pion
      #a->b+c decay
      #can use energy conservation to solve for momentum of b and c in CoM frame
      #E=m_a=sqrt(p^2+mb^2)+sqrt(p^2+mc^2)
      #solved for p using an online algebraic tool

      #momentum of the particles in CoM frame
      pcm=dec_mom_sol(m_del0,m_p,m_pi)
      #(use mdel instead?)

      #total energy of each particle in CoM frame
      Ep=E_solv(pcm,m_p)
      Epi=E_solv(pcm,m_pi)

      #give the proton and pion momenta direction in the delta frame
      ppx,ppy,ppz,pth,pph=vec_gen(pcm)

      #write the 4 momenta of p and pi in delta frame
      p4pD=[Ep,ppx,ppy,ppz]
      p4piD=[Epi,-ppx,-ppy,-ppz]

      #####################
      #LT back to lab frame
      #####################

      p4pL=gam_mat(dgam,dv,-vdx,-vdy,-vdz,p4pD)
      p4piL=gam_mat(dgam,dv,-vdx,-vdy,-vdz,p4piD)



  ########################
  #Free particle generator
  ########################

  if Free is True:
    N_free =2 #number of free particle pairs per event
    #should also be randomized (Boltzmann dist?)

    #generate the kinetic energy of the particles in lab frame
    p_k=en_dist(a1,T1,N_free)
    pi_k=en_dist(a2,T2,N_free)  
    
    for k in range(0,len(p_k)):
      gam1,v1,p_mom,p_et=gam_calc(p_k[k],m_p)
      pxp,pyp,pzp,th_p,ph_p=vec_gen(p_mom)
      p4p=[p_et,pxp,pyp,pzp]
      gam2,v2,pi_mom,pi_et=gam_calc(pi_k[k],m_pi)
      pxpi,pypi,pzpi,th_pi,ph_pi=vec_gen(pi_mom) 
      p4pi=[pi_et,pxpi,pypi,pzpi]
      
      p_list.append(p4p)
      pi_list.append(p4pi)
  