import numpy as np
import random

#Switch for deltas to be generated
Delta=False

#Switch for free particles to be generated with the delta resonances
Free=False 

#should output the PID and 4-momentum of every particle generated as the output
#header should include the number of events generated, the number of particles generated

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




#number of events
N_events=100


for i in range(0,N_events):
  ################
  #Delta Generator
  ################

  if Delta is True:
    ()





  ########################
  #Free particle generator
  ########################

  if Free is True:
    ()
