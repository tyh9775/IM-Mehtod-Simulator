import numpy as np
import csv

#load constants
import myconst as mc

#for identifiying event number and the number of particles in the event
def h_read(header):
  e_num=int(header[0])
  numPart=int(header[1])
  return e_num, numPart


#for identifying particles
def PID(p):
  prt=0
  if int(p[0])==2212:
    prt=0 #proton
  elif int(p[0])==211:
    prt=1 #postive pion
  elif int(p[0])==-211:
    prt=2 #negative pion
  elif int(p[0])==2112:
    prt=3 #neutron
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

'''
#given the momentum and the rest mass, solve for the total energy
def E_solv(p,m):
  pmg=dist_form(p)
  return np.sqrt(pmg**2+m**2)
'''
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


#csv file to save invariant mass data of all events
with open('data_IM.csv','w') as fm:
  fm.close()


IM_list=[] #the invariant mass of the pairs from all events


with open("data.csv",'r') as file:
  f=csv.reader(file, delimiter=',')
  for row in f:
    #momenta of protons and pions
    p_list=[]
    pi_list=[]
    row1=next(f)
    print(row1)
    eventNum,partNum=h_read(row1)
    for i in range(0,partNum):
      rowdata=next(f)
      identifier=PID(rowdata)
      if identifier==0:
        p_list.append(rowdata[1:])
      elif identifier==1:
        pi_list.append(rowdata[1:])


    #invariant mass of the p and pi in the event
    m_list=IM_method(p_list,pi_list,mc.p_cut)

    for kk in range(0,len(m_list)):
      IM_list.append(m_list[kk])
    
    with open('data_IM.csv','a',newline='') as fm:
      g=csv.writer(fm,delimiter=',')
      g.writerow(m_list)
      fm.close()

  file.close()


print(IM_list)


