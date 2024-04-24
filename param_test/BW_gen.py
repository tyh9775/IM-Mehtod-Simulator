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

def fwhm_calc(data,bins):
  max_value=np.max(data)
  max_ind=np.argmax(data)
  half_max_val=max_value/2
  #find where the data crosses the half piont on the left and right of the peak
  left_ind=np.argmin(np.abs(data[0:max_ind]-half_max_val))
  right_ind=np.argmin(np.abs(data[max_ind:]-half_max_val))+max_ind
  fwhm=bins[right_ind]-bins[left_ind]

  return fwhm,half_max_val,left_ind,right_ind,max_ind

show=False
show_all=False

def generator(numD,numF,filename,norm_w=None,DT=None,A=None,a=None,b=None,output_folder=None):

  #build mass distribution 
  x_bw=np.linspace(mc.md_min,mc.md_max,100)
  y_bw=[]
  for i in range(0,len(x_bw)):
    y_bw.append(bw_pdf(x_bw[i],mc.m_del0,mc.m_p,mc.m_pi,A=A,a=a,b=b))
  norm_const=simpson(y=y_bw,x=x_bw)
  y_norm=y_bw/norm_const

  fwhm,hlf_val,lft,rgt,mxi=fwhm_calc(y_bw,x_bw)

  plt.figure()
  plt.plot(x_bw,y_bw,'.')
  plt.hlines(y=hlf_val,xmin=x_bw[lft],xmax=x_bw[rgt],label=f'fwhm={round(fwhm,3)}')
  plt.xlabel("Mass (Mev/c^2)")
  #plt.ylabel("Probability")
  plt.title("Breit-Wigner Distribution")
  if output_folder is not None:
    if A is None and a is None:
      plot_path=os.path.join(output_folder,f'BW_{b}.png')
      plt.figtext(0.75,0.75,f"A=0.95 \n a=0.47 \n b={b}",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    elif A is None and b is None:
      plot_path=os.path.join(output_folder,f'BW_{a}.png')
      plt.figtext(0.75,0.75,f"A=0.95 \n a={a} \n b=0.6",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    elif a is None and b is None:
      plot_path=os.path.join(output_folder,f'BW_{A}.png')
      plt.figtext(0.75,0.75,f"A={A} \n a=0.47 \n b=0.6",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    else:
      plot_path=os.path.join(output_folder,f'BW_0.95_0.47_0.6.png')
      plt.figtext(0.75,0.75,f"A=0.95 \n a=0.47 \n b=0.6",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
  else:
    plot_path=f'BW_{A}_{a}_{b}.png'
    if A is None and a is None:
      plot_path=output_folder,f'BW_{b}.png'
      plt.figtext(0.75,0.75,f"A=0.95 \n a=0.47 \n b={b}",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    elif A is None and b is None:
      plot_path=output_folder,f'BW_{a}.png'
      plt.figtext(0.75,0.75,f"A=0.95 \n a={a} \n b=0.6",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    elif a is None and b is None:
      plot_path=output_folder,f'BW_{A}.png'
      plt.figtext(0.75,0.75,f"A={A} \n a=0.47 \n b=0.6",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    else:
      plot_path=output_folder,f'BW_0.95_0.47_0.6.png'
      plt.figtext(0.75,0.75,f"A=0.95 \n a=0.47 \n b=0.6",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
  plt.legend(loc='upper left')
  plt.savefig(plot_path)
  if show is True:
    plt.show()
  plt.close()

  if numD==0 and numF==0:
    print("numD,numF:",0,0)
    print("No particles detected")
    print()
    return
  
  if output_folder is not None:
    abs_path=os.path.dirname(__file__)
    with open(os.path.join(abs_path,output_folder,filename), 'w', newline='') as file:
      file.close()
  else:
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

    if output_folder is None:
      with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([counter,particles,N_delta])
        file.close()
    
    else:
      with open(os.path.join(output_folder,filename), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([counter,particles,N_delta])
        file.close()

    for j in range(0,N_delta):
      ND_total=ND_total+1

      ######################################
      #starting in center of collision frame
      ######################################

      if norm_w is not None:
        #randomly choose a mass based on a normal distribution
        mdel=np.random.normal(1232,norm_w,1)[0]
      else:
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
      
      if output_folder is None:
        with open(filename,'a',newline='') as file:
          g=csv.writer(file, delimiter=',')
          g.writerow(datadel)
          g.writerow(datap)
          g.writerow(datapi)
          file.close()
      else:
        with open(os.path.join(output_folder,filename),'a',newline='') as file:
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
      
      if output_folder is None:
        with open(filename,'a',newline='') as file:
          g=csv.writer(file, delimiter=',')
          g.writerow(datap)
          g.writerow(datapi)
          file.close()
      else:
        with open(os.path.join(output_folder,filename),'a',newline='') as file:
          g=csv.writer(file, delimiter=',')
          g.writerow(datap)
          g.writerow(datapi)
          file.close()    
  print("numD,numF:",numD,numF)
  print("Number of Delta resonances created:",ND_total)
  print("Number of all particles detected:", NP_total)
  print()
  return x_bw,y_bw,fwhm,mxi


#numbers of particles/pairs generated
Delta_num=[1,2,3]
Free_num=[0,1,2,3]

#set constants/parameters
dnorm_w=[5,10,20]
delta_temp=300
A=[0.1,0.5,1,2]
a=[0.1,0.5,1,2]
b=[0.1,0.5,1,2]

abs_path=os.path.dirname(__file__)
output_folderA=os.path.join(abs_path,"bw_A")
os.makedirs(output_folderA,exist_ok=True)
xlistA=[]
ylistA=[]
fwhm_listA=[]
max_listA=[]
ver_listA=[]
for Ai in A:
  filename=f"BW_A_{Ai}.csv"
  x,y,fwhm,mxi=generator(Delta_num[0],Free_num[0],filename,DT=delta_temp,A=Ai,output_folder=output_folderA)
  xlistA.append(x)
  ylistA.append(y)
  fwhm_listA.append(fwhm)
  max_listA.append(x[mxi])
  ver_listA.append(y[mxi])

plt.figure()
for i in range(0,len(xlistA)):
  plt.plot(xlistA[i],ylistA[i],'.',label=f'A={A[i]}')
plt.legend()
plt.title("BW Dist. with Varying 'A' Values")
plt.savefig(os.path.join(output_folderA,"BW_all.png"))
if show_all is True:
  plt.show()
plt.close()


xlista=[]
ylista=[]
fwhm_lista=[] 
max_lista=[]
ver_lista=[]
output_foldera=os.path.join(abs_path,"bw_qa")
os.makedirs(output_foldera,exist_ok=True)
for ai in a:
  filename=f"BW_a_{ai}.csv"
  x,y,fwhm,mxi=generator(Delta_num[0],Free_num[0],filename,DT=delta_temp,a=ai,output_folder=output_foldera)
  xlista.append(x)
  ylista.append(y)
  fwhm_lista.append(fwhm)
  max_lista.append(x[mxi])
  ver_lista.append(y[mxi])

plt.figure()
for i in range(0,len(xlista)):
  plt.plot(xlista[i],ylista[i],'.',label=f'a={a[i]}')
plt.legend()
plt.title("BW Dist. with Varying 'a' Values")
plt.savefig(os.path.join(output_foldera,"BW_all.png"))
if show_all is True:
  plt.show()
plt.close()

xlistb=[]
ylistb=[] 
fwhm_listb=[]
max_listb=[]
ver_listb=[]
output_folderb=os.path.join(abs_path,"bw_qb")
os.makedirs(output_folderb,exist_ok=True)
for bi in b:
  filename=f"BW_b_{bi}.csv"
  x,y,fwhm,mxi=generator(Delta_num[0],Free_num[0],filename,DT=delta_temp,b=bi,output_folder=output_folderb)
  xlistb.append(x)
  ylistb.append(y)
  fwhm_listb.append(fwhm)
  max_listb.append(x[mxi])
  ver_listb.append(y[mxi])

plt.figure()
for i in range(0,len(xlistb)):
  plt.plot(xlistb[i],ylistb[i],'.',label=f'b={b[i]}')
plt.legend()
plt.title("BW Dist. with Varying 'b' Values")
plt.savefig(os.path.join(output_folderb,"BW_all.png"))
if show_all is True:
  plt.show()
plt.close()


plt.figure()
plt.plot(A,fwhm_listA,label="A")
plt.plot(a,fwhm_lista,label="a")
plt.plot(b,fwhm_listb,label="b")
plt.legend()
plt.xlabel("Numerical Value")
plt.ylabel("FWHM")
plt.title('FWHM vs Various Parameters')
plt.savefig("FWHM_comp.png")
if show_all is True:
  plt.show()
plt.close()

plt.figure()
plt.plot(A,max_listA,label="A")
plt.plot(a,max_lista,label="a")
plt.plot(b,max_listb,label="b")
plt.legend()
plt.xlabel("Numerical Value")
plt.ylabel("X Value of Peak")
plt.title('X of Peak vs Various Parameters')
plt.savefig("xpeak_comp.png")
if show_all is True:
  plt.show()
plt.close()

plt.figure()
plt.plot(A,ver_listA,label="A")
plt.plot(a,ver_lista,label="a")
plt.plot(b,ver_listb,label="b")
plt.legend()
plt.xlabel("Numerical Value")
plt.ylabel("Y Value of Peak")
plt.title('Y of Peak vs Various Parameters')
plt.savefig("ypeak_comp.png")
if show_all is True:
  plt.show()
plt.close()
