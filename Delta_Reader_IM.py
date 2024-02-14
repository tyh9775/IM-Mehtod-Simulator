import numpy as np
import csv

#load constants
import myconst as mc

def PID(p):
  prt=0
  if float(p) < mc.m_pi+0.01 and float(p) > mc.m_pi-0.01:
    prt=0
  else:
    prt=1
  return prt

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

#given the momentum and the rest mass, solve for the total energy
def E_solv(p,m):
  pmg=dist_form(p)
  return np.sqrt(pmg**2+m**2)

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


#momenta of protons and pions
p_list=[]
pi_list=[]

with open("data1s.csv",'r') as file:
  f=csv.reader(file, delimiter=',')
  for row in f:
    p4=[float(i) for i in row]
    if PID(p4[0])==0:
      pi_list.append(p4[1:])
    else:
      p_list.append(p4[1:])
  file.close()



for ii in range(0,len(mc.p_cut)):
  for jj in range(0,len(mc.m_cut)):
    #invariant mass
    m_list=[]
    for i in range(0,len(pi_list)):
      for j in range(0,len(p_list)):
        pp=p_list[j] #3D momentum of p
        ppi=pi_list[i] #3D momentum of pi
        Ep=E_solv(pp,mc.m_p) #total E of p
        Epi=E_solv(ppi,mc.m_pi) #total E of pi
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
        if pt_mag < mc.p_cut[ii]:
          m_list.append(inv_m(Etot,pmag))

    print(m_list)
