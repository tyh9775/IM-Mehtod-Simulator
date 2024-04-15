import numpy as np
import random
from scipy.integrate import simpson
import csv
import myconst as mc

#Breit_Wigner distribution for the  mass distribution of delta resonances:
#class bw_dist(st.rv_continuous):
def bw_pdf(md,md0,mn,mpi,bw_width=None):
  A=0.95 
  if md==mn+mpi:
    q=0
  elif md<mn+mpi:
    print("invalide mass of delta resonance produced; check mass of nucleon and pion")
    quit()
  else:
    q=np.sqrt((md**2-mn**2-mpi**2)**2-4*(mn*mpi)**2)/(2*md)
  if bw_width is not None:
    gmd=bw_width
  else:
    gmd=(0.1*q**3)/(mpi**2+0.6*q**2)
  
  return (4*md0**2*gmd)/((A)*((md**2-md0**2)**2+md0**2*gmd**2))

def q_solv(md,mn,mpi):
  return np.sqrt((md**2-mn**2-mpi**2)**2-4*(mn*mpi)**2)/(2*md)

#calculate the momentum of delta given the center of collision energy, mdel, and mn
def bw_mnt(rs,m1,m2):
  return np.sqrt((rs**2+m1**2-m2**2)**2/(4*rs**2)-m1**2)

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

#build mass distribution 
x_bw=np.linspace(mc.md_min,mc.md_max,100)
y_bw=[]
for i in range (0,len(x_bw)):
  y_bw.append(bw_pdf(x_bw[i],mc.m_del0,mc.m_p,mc.m_pi))
norm_const=simpson(y=y_bw,x=x_bw)
y_norm=y_bw/norm_const

#in order: # deltas, # free pairs, name of output file, temp of delta when 
#generated with exponential KE, norm m dist width, bw m dist width
def generator(numD,numF,filename,norm_w=None,bw_w=None,DT=None):

  if numD==0 and numF==0:
    print("numD,numF:",0,0)
    print("No particles detected")
    print()
    return
  
  with open(filename, 'w', newline='') as file:
    file.close()

  N_events=mc.nevts
  counter=0
  ND_total=0
  NP_total=0

  for i in range(0,N_events):
    counter = counter+1
    particles=0
    N_delta=numD
    N_free=numF
    
    particles=particles+N_delta*2+N_free*2
    NP_total=NP_total+particles

    with open(filename, 'a', newline='') as file:
      writer = csv.writer(file)
      if bw_w is not None:
        writer.writerow([counter,particles,N_delta,bw_w])
      else:
        writer.writerow([counter,particles,N_delta])
      file.close()

    for j in range(0,N_delta):
      ND_total=ND_total+1

      ######################################
      #starting in center of collision frame
      ######################################

      if norm_w is not None and bw_w is None:
        #randomly choose a mass based on a normal distribution
        mdel=np.random.normal(1232,norm_w,1)[0]
      elif norm_w is not None and bw_w is not None:
        print('choose either normal or bw dist, not both')
        return
      elif norm_w is None and bw_w is None:
        #randomly choose the mass of the delta resonance according to bw dist
        #using monte carlo method
        mdel=random.uniform(mc.md_min,mc.md_max)
        ytest=random.uniform(0,max(y_norm))
        while ytest > bw_pdf(mdel,mc.m_del0,mc.m_p,mc.m_pi)/norm_const:
          mdel=random.uniform(mc.md_min,mc.md_max)
          ytest=random.uniform(0,max(y_norm))
      elif norm_w is None and bw_w is not None:
        mdel=random.uniform(mc.md_min,mc.md_max)
        ytest=random.uniform(0,max(y_norm))
        while ytest > bw_pdf(mdel,mc.m_del0,mc.m_p,mc.m_pi,bw_width=bw_w)/norm_const:
          mdel=random.uniform(mc.md_min,mc.md_max)
          ytest=random.uniform(0,max(y_norm))
      #PID of delta:
      dpid=2224 #delta++
      
      #give delta a random momentum
      if DT is not None:
        ke_del=exp_dist(DT,1)[0]
        dgam,dv,Edel,pdel=kgam_calc(ke_del,mdel)
      else:
        pdel=bw_mnt(mc.rt_s,mdel,mc.m_p)
        Edel=E_solv(pdel,mdel)
        dgam,dv=gam_calc(Edel,mdel)

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
      
      with open(filename,'a',newline='') as file:
        g=csv.writer(file, delimiter=',')
        g.writerow(datadel)
        g.writerow(datap)
        g.writerow(datapi)
        file.close()

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
            
      with open(filename,'a',newline='') as file:
        g=csv.writer(file, delimiter=',')
        g.writerow(datap)
        g.writerow(datapi)
        file.close()
  print("numD,numF:",numD,numF)
  print("Number of Delta resonances created:",ND_total)
  print("Number of all particles detected:", NP_total)
  print()
  return 

#numbers of particles/pairs generated
Delta_num=[1,2,3]
Free_num=[0,1,2,3]

#set constants/parameters
dnorm_w=[5,10,20]
dbw_w=[5,10,20]
delta_temp=300


'''filename=f"test_ndist.csv"
generator(Delta_num,Free_num,filename,DT=delta_temp,norm_w=dnorm_w)
'''

for gamma in dbw_w:
  filename=f"test_bw_{gamma}.csv"
  generator(Delta_num[0],Free_num[0],filename,bw_w=gamma,DT=delta_temp)