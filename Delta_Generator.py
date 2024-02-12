import numpy as np
import random
from scipy.integrate import simpson


#Switch for deltas to be generated
Delta=False

#Switch for free particles to be generated with the delta resonances
Free=False 

#should output the PID and 4-momentum of every particle generated as the output
#header should include the number of events, the total number of particles, and the 

if Delta is False and Free is False:
    print("No particles generated!")
    quit()


#to generate n particles according to an (a/T)*e^(-x/T) distribution
def en_dist(a,T,n):
  return a*np.random.exponential(scale=1/T,size=n)

#find the invariant mass given the total energy and the momentum
def inv_m(en,p):
  m2=en**2-p**2
  return np.sqrt(m2)

#given KE and rest mass, calculates the Lorentz factor, relative velocity, the relativistic momentum, and total energy
def gam_calc(KE,m):
  gam=1+KE/m
  v=np.sqrt(1-1/gam**2)
  p=gam*m*v
  et=gam*m
  return gam, v, p, et

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

#given the momentum and the rest mass, solve for the kinetic energy and the total energy
def E_solv(p,m):
  En=np.sqrt(p**2+m**2)
  KE=En-m
  return KE, En

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

#Breit_Wigner distribution for the  mass distribution of delta resonances:
#class bw_dist(st.rv_continuous):
def bw_pdf(md,md0,mn,mpi):
  A=0.95 
  q=np.sqrt((md**2-mn**2-mpi**2)**2-4*(mn*mpi)**2)/(2*md)
  gmd=(0.47*q**3)/(mpi**2+0.6*q**2)
  return (4*md0**2*gmd)/((A)*((md**2-md0**2)**2+md0**2*gmd**2))



#constant values
mdel=1232 #MeV/c^2 - rest mass of delta resonance
m_p=938 #MeV/c^2 - rest mass of proton
m_pi=139.570 #MeV/c^2 - rest mass of charged pion
Eb=270 #Beam energy per nucleon (AMeV)

#bounds for the Breit-Wigner distribution of the delta mass in lab frame
md_min=m_p+m_pi
md_max=2015+Eb-m_p #2015 MeV is the minimum C.M. energy of the delta for the energy-dependent isospin-averaged isotropic cross section to be non-zero

#build mass distribution 
x_bw=np.linspace(md_min,md_max,1000)
y_bw=[]
for i in range (0,len(x_bw)):
  y_bw.append(bw_pdf(x_bw[i],mdel,m_p,m_pi))
norm_const=simpson(y=y_bw,x=x_bw)
y_norm=y_bw/norm_const

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
    




  ########################
  #Free particle generator
  ########################

  if Free is True:
    ()
