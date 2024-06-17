import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import myconst as mc
import os
import glob
import re

mass_method=True
eventcheck=False
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

#given a->b+c decay, solves for p of b and c
def dec_mnt_sol(m0,m1,m2):
  return np.sqrt(m0**4-2*(m0*m1)**2-2*(m0*m2)**2+m1**4-2*(m1*m2)**2+m2**4)/(2*m0)

#LT in 3D space in matrix form
def gam_mat(gam,v,vx,vy,vz,p4):
    A=np.array([[gam,-gam*vx,-gam*vy,-gam*vz],
                [-gam*vx,1+(gam-1)*vx**2/v**2,(gam-1)*vx*vy/v**2,(gam-1)*vx*vz/v**2],
                [-gam*vy,(gam-1)*vx*vy/v**2,1+(gam-1)*vy**2/v**2,(gam-1)*vy*vz/v**2],
                [-gam*vz,(gam-1)*vx*vz/v**2,(gam-1)*vy*vz/v**2,1+(gam-1)*vz**2/v**2]])
    return np.dot(A,p4)

#find the invariant mass given the total energy and the momentum
def inv_m(en,p):
  return np.sqrt(en**2-p**2)

def IM_method(plist,pilist):
  m_list=[]
  mnt_list=[]
  for i in range(0,len(pilist)):
    for j in range(0,len(plist)):
      #starting in lab frame
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
      dec_mnt=dec_mnt_sol(mdel_rec,mc.m_p,mc.m_pi) #theoretical mag of the mnt of p or pi in delta frame
      ptest_mag=dist_form(ptest[1:]) #mag of the mnt of p in delta frame
      pitest_mag=dist_form(pitest[1:]) #mag of the mnt of pi in delta frame
      
      if eventcheck is True:
        print()
        d4=[Etot]+ptot
        dtest=gam_mat(gam,v,vx,vy,vz,d4)
        print("proton 4-mnt:",p4p)
        print("pion 4-mnt:",p4pi)
        print("vector sum of the mnt:",ptot)
        print("mag of 3D mnt:",pmag)
        print("calculated inv m:",mdel_rec)
        print()
        print("mnt of recreated delta:",d4)
        print("mnt of delta in it's own rest frame:",dtest)
        print()
        print("mnt of proton after LT:",ptest)
        print("mnt of pion after LT:",pitest)
        print("total mnt after LT:",pt_tot)
        print("mag of tot mnt after LT:",pt_mag)
        print("theoretical mag of mnt of p or pi:",dec_mnt)
        print("mag of mnt of p after LT:",ptest_mag)
        print()

      #momentum cut and upper limit on the invariant mass
      if mass_method is True:
        if pt_mag < mc.p_cut:
          m_list.append(mdel_rec)
          mnt_list.append(pmag)
      else:
        m_list.append(mdel_rec)
        mnt_list.append(pmag)        
  if eventcheck is True:
    quit()
  return m_list, mnt_list

#for fitting
def poly_func(x,c0,c1,c2,c3,c4):
  return c0+c1*x+c2*x**2+c3*x**3+c4*x**4

def gaus_func(x,A,x0,sig):
  return A*np.exp(-((x-x0)**2)/(2*sig**2))/(np.sqrt(2*np.pi)*sig)

def exp_func(x,a,b,c):
  return a*np.exp(-b*x)+c

def bw_func(x,A,a,b):
  q=np.sqrt((x**2-mc.m_p**2-mc.m_pi**2)**2-4*(mc.m_p*mc.m_pi)**2)/(2*x)
  gam=(a*q**3)/(mc.m_pi**2+b*q**2)
  return (4*gam*mc.m_del0**2)/(A*((x**2-mc.m_del0**2)**2+(mc.m_del0*gam)**2))

def bw_func_alt(x,A,a,b,scl):
  q=np.sqrt((x**2-mc.m_p**2-mc.m_pi**2)**2-4*(mc.m_p*mc.m_pi)**2)/(2*x)
  gam=(a*q**3)/(mc.m_pi**2+b*q**2)
  return scl*(4*gam*mc.m_del0**2)/(A*((x**2-mc.m_del0**2)**2+(mc.m_del0*gam)**2))

def cmb_func_p(x,A,a,b,c0,c1,c2,c3,c4,scl1,scl2):
  return scl1*bw_func(x,A,a,b)+scl2*poly_func(x,c0,c1,c2,c3,c4)

def cmb_func_e(x,A,a,b,a1,b1,c1,scl1,scl2):
  return scl1*bw_func(x,A,a,b)+scl2*exp_func(x,a1,b1,c1)

def cmp_func(x,A,b,c):
  return A*(x**2)*np.exp(-b*x)+c

def poly_fit_optimize(x,y,max_deg):
  errors=[]
  models=[]
  for i in range(0,max_deg):
    coef=np.polyfit(x,y,deg=i)
    models.append(coef)
    p=np.poly1d(coef)
    y_pred=p(x)
    mse = np.mean((y-y_pred)**2)
    errors.append(mse)
  best_deg=np.argmin(errors)+1
  best_coef=models[best_deg-1]
  return best_coef, best_deg

def fwhm_calc(data,bins):
  max_value=np.max(data)
  max_ind=np.argmax(data)
  half_max_val=max_value/2
  #find where the data crosses the half piont on the left and right of the peak
  left_ind=np.argmin(np.abs(data[0:max_ind]-half_max_val))
  right_ind=np.argmin(np.abs(data[max_ind:]-half_max_val))+max_ind
  fwhm=bins[right_ind]-bins[left_ind]

  return fwhm,half_max_val,left_ind,right_ind,max_ind

def param_reader(filename):
  numbers=re.findall(r'\d+\.\d+|\d+', filename)
  A=re.findall(r'_A_', filename)
  a=re.findall(r'_a_', filename)
  b=re.findall(r'_b_', filename)
  numbers=[float(num) for num in numbers]
  if len(A) != 0:
    A=numbers[-1]
  else:
    A=0.95
  if len(a) != 0:
    a=numbers[-1]
  else:
    a=0.47
  if len(b) != 0:
    b=numbers[-1]
  else:
    b=0.6
  param=[A,a,b]
  return param

def r2_calc(f,x,y,p):
  res=[]
  ss_res=[]
  ss_tot=[]
  m=np.mean(y)
  for i in range(0,len(x)):
    r=y[i]-f(x[i],*p)
    res.append(r)
    ss_res.append(r**2)
    ss_tot.append((y[i]-m)**2)
  ssr=np.sum(ss_res)
  sst=np.sum(ss_tot)
  return 1-(ssr/sst)

def chi2(x,y,yerr,A,a,b):
  sum=0
  for ind in range(0,len(x)):
    if yerr[ind]==0:
      yerr[ind]=1e-10
    sum+=(y[ind]-bw_func(x[ind],A,a,b))**2/yerr[ind]**2
  return sum

def chi2gen(f,x,y,yerr,p):
  sum=0
  for ind in range(0,len(x)):
    if yerr[ind]==0:
      yerr[ind]=1e-10
    sum+=(y[ind]-f(x[ind],*p))**2/yerr[ind]**2
  return sum

def reader(directory,file_pattern,output_folder):
#for comparing different widths
  IM_all=[]
  act_all=[]
  cr_all=[]
  mnt_all=[]

  for filename in glob.glob(os.path.join(directory,file_pattern)):
    param=param_reader(filename)
    print(param)
    new_file_path= os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_IM.csv")
    with open(new_file_path,"w",newline='') as new_file:
      new_file.close()

    IM_list=[] #the invariant mass of the recreated deltas from all events
    act_list=[] #momenta of "real" deltas
    act_IM=[] #inv mass of "real" deltas
    free_mnt=[] #mnt of "fake" deltas
    free_IM=[] #inv mass of "fake" deltas
    momentum_list=[] #the momenta of recreated deltas
    p_mnt=[] #momenta of detected protons
    pi_mnt=[] #momenta of detected pions
    p_en=[] #energy of protons
    pi_en=[] #energy of pions

    cr_IM=[] #IM of related pairs
    cr_mnt=[] #mnt of related pairs

    #open data file, do a momentum cut, and calculate the invariant mass of the particle pairs
    with open(filename,'r') as file:
      f=csv.reader(file, delimiter=',')
      for row in f:
        #4 momenta of protons and pions
        p_list=[]
        pi_list=[]

        #free particles
        pflist=[]
        piflist=[]

        del_list=[] #4 mnt of 'real' delta
        
        #eventNum,partNum,Ndelta
        header=row
        eventNum=int(header[0])
        partNum=int(header[1])
        Ndelta=int(header[2])
        plist={}
        pilist={}

        for j in range(0,Ndelta):
          pname=f'p{j+1}_list'
          piname=f'pi{j+1}_list'
          plist[pname]=[]
          pilist[piname]=[]

        for i in range(0,partNum+Ndelta):
          rowdata=next(f)
          PID=int(rowdata[0]) #identify the particle with PDG codes
          Par_ID=int(rowdata[5])
          if PID==2224:
            del_list.append(rowdata[1:5])
            dpmag=dist_form(rowdata[2:5])
            dm=inv_m(float(rowdata[1]),dpmag)
            act_IM.append(dm)
          elif PID==2212: #proton
            p_list.append(rowdata[1:5])
            p_en.append(float(rowdata[1]))
            p_mnt.append(dist_form(rowdata[2:4]))
            if Par_ID==0:
              pflist.append(rowdata[1:5])
            else:
              plist[f'p{Par_ID}_list'].append(rowdata[1:5])
          elif PID==211: #pion+
            pi_list.append(rowdata[1:5])
            pi_en.append(float(rowdata[1]))
            pi_mnt.append(dist_form(rowdata[2:4]))
            if Par_ID==0:
              piflist.append(rowdata[1:5])
            else:
              pilist[f'pi{Par_ID}_list'].append(rowdata[1:5])       
        #invariant mass of the p and pi in the event with momentum cut applied
        #random.shuffle(p_list)
        #random.shuffle(pi_list)
        
        #inv mass of all pairs
        m_list,mnt_list=IM_method(p_list,pi_list)

        #inv mass of free particle pairs
        mf_list,mntf_list=IM_method(pflist,piflist)


        for ff in range(0,len(mf_list)):
          free_IM.append(mf_list[ff])
        for ff in range(0,len(mntf_list)):
          free_mnt.append(mf_list[ff])

        m_cr_list=[]
        mnt_cr_list=[]

        for i in range(0,Ndelta):
          m_cr,mnt_cr=IM_method(plist[f'p{i+1}_list'],pilist[f'pi{i+1}_list'])
          m_cr_list.append(m_cr)
          mnt_cr_list.append(mnt_cr)
        
        for jj in range(0,len(del_list)):
          act_list.append(dist_form(del_list[jj][1:]))

        for kk in range(0,len(m_list)):
          IM_list.append(m_list[kk])
        for ll in range(0,len(mnt_list)):
          momentum_list.append(mnt_list[ll])

        for ii in range(0,len(m_cr_list)):
          for kk in range(0,len(m_cr_list[ii])):
            cr_IM.append(m_cr_list[ii][kk])
          for ll in range(0,len(mnt_cr_list[ii])):
            cr_mnt.append(mnt_cr_list[ii][ll])
        
        with open(new_file_path,'a',newline='') as new_file:
          nfwriter=csv.writer(new_file,delimiter=',')
          nfwriter.writerow(m_list)
          new_file.close()

      file.close()

    IM_all.append(IM_list)
    cr_all.append(cr_IM)
    mnt_all.append(mnt_list)


    #graphing and fitting
    binsize=5 #in MeV/c^2
    if len(IM_list)!=0:
      plt.figure()
      hist,bins,packages=plt.hist(IM_list,bins=np.arange(int(min(IM_list)),int(max(IM_list)),binsize),alpha=0)
      bins_cntr=0.5*(bins[:-1]+bins[1:])    
      hist_err=np.sqrt(hist)
      plt.errorbar(bins_cntr,hist,xerr=binsize/2,yerr=hist_err,fmt='.')
      #limit the range the fit is done to exclude bins with ~0 counts after the peak
      peak_pt=np.where(hist==max(hist))[0][0]
      endpoints=np.where(hist[peak_pt:]<=0.01*max(hist))[0]
      if len(endpoints)==0:
        endpoint=len(bins_cntr)-1
      else:
        endpoint=peak_pt+endpoints[0]
      xfitbw=np.arange(min(bins_cntr),bins_cntr[endpoint],0.5)
      ydef=bw_func(xfitbw,*param)
      sclr=max(hist)/max(ydef)
      yfit_par=bw_func(xfitbw,*param)
      lwrbnd=[0,0,0]
      uprbnd=[np.inf,np.inf,np.inf]
      sigpts=hist_err[:endpoint]
      sigpts=[1 if sigs==0 else sigs for sigs in sigpts]
      popt,pcov=curve_fit(bw_func,bins_cntr[:endpoint],hist[:endpoint],p0=[*param],bounds=(lwrbnd,uprbnd),sigma=sigpts,absolute_sigma=True)
      A_est=np.abs(popt[0]*sclr)
      a_est=np.abs(popt[1])
      b_est=np.abs(popt[2])
      eA=np.sqrt(pcov[0,0])*sclr
      ea=np.sqrt(pcov[1,1])
      eb=np.sqrt(pcov[2,2])
      chi_sq=chi2(bins_cntr[:endpoint],hist[:endpoint],sigpts,*popt)
      yfitbw=bw_func(xfitbw,*popt)
      yfit_par=np.multiply(yfit_par,sclr)
      fwhm,hlf_val,lft,rgt,mxi=fwhm_calc(yfitbw,xfitbw)
      plt.plot(xfitbw,yfitbw,label='BW fit')
      plt.plot(xfitbw,yfit_par,'--',label='Function w/ given param')
      plt.hlines(y=hlf_val,xmin=xfitbw[lft],xmax=xfitbw[rgt],label=f'fwhm={round(fwhm,3)}',colors='red')
      plt.title("Invariant Mass of Recreated Delta in Lab Frame")
      plt.ylabel("Count")
      plt.xlabel("Mass (MeV/c^2)")
      plt.legend(loc='upper left')
      plt.ylim(0,max(hist)*1.1)
      plt.xlim(min(bins_cntr)-2*binsize,mc.m_max)
      plt.figtext(0.75,0.75,"par=%s \n A=%s+/-%s\n a=%s+/-%s\n b=%s+/-%s \n chi_sq=%s"%(param,round(A_est,3),round(eA,3),round(a_est,3), round(ea,3),round(b_est,3),round(eb,3),round(chi_sq,3)),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_IM_plot.png")
      plt.savefig(plot_file_path)
      #plt.show()
      plt.close()
      print("total number of counted particles after momentum cut:", np.sum(hist))
    
    #"actual" deltas
    if len(act_IM)!=0:
      plt.figure()
      hist_act,bins_act,pack_act=plt.hist(act_IM,bins=np.arange(min(act_IM),max(act_IM),binsize),alpha=0)
      bins_cntr_act=0.5*(bins_act[:-1]+bins_act[1:])
      xfitbw_act=np.arange(min(bins_cntr_act),max(bins_cntr_act),0.5)
      ydef_act=bw_func(xfitbw_act,*param)
      sclr_act=max(hist_act)/max(ydef_act)
      yfit_par_act=bw_func(xfitbw_act,*param)
      yfit_par_act=np.multiply(yfit_par_act,sclr_act)
      act_err=np.sqrt(hist_act)
      peak_act=np.where(hist_act==max(hist_act))[0][0]
      endpoints=np.where(hist_act[peak_act:]<=0.01*max(hist_act))[0]
      if len(endpoints)==0:
        endpoint=len(bins_cntr_act)-1
      else:
        endpoint=peak_act+endpoints[0]
      sigpts_act=act_err[:endpoint]
      sigpts_act=[1 if sigs==0 else sigs for sigs in sigpts_act]
      popt_act,pcov_act=curve_fit(bw_func,bins_cntr_act[:endpoint],hist_act[:endpoint],p0=[0.95,0.47,0.6],bounds=(lwrbnd,uprbnd ),sigma=sigpts_act,absolute_sigma=True)
      Aa_est=np.abs(popt_act[0]*sclr_act)
      aa_est=np.abs(popt_act[1])
      ba_est=np.abs(popt_act[2])
      eAa=np.sqrt(pcov_act[0,0])*sclr_act
      eaa=np.sqrt(pcov_act[1,1])
      eba=np.sqrt(pcov_act[2,2])
      act_chi_sq=chi2(bins_cntr_act[:endpoint],hist_act[:endpoint],sigpts_act,*popt_act)
      yfitbw_act=bw_func(xfitbw_act,*popt_act)
      fwhm_act,hlf_val,lft,rgt,mxi=fwhm_calc(yfitbw_act,xfitbw_act)
      plt.plot(xfitbw_act,yfitbw_act,label='BW fit')
      plt.plot(xfitbw_act,yfit_par_act,'--',label='Function w/ given param')
      plt.errorbar(bins_cntr_act[:endpoint],hist_act[:endpoint],xerr=binsize/2,yerr=sigpts_act,fmt='.')
      plt.hlines(y=hlf_val,xmin=xfitbw_act[lft],xmax=xfitbw_act[rgt],label=f'fwhm={round(fwhm_act,3)}',colors='red')
      plt.figtext(0.75,0.75,"par=%s \n A=%s+/-%s \n a=%s+/-%s \n b=%s+/-%s \n chi_sq=%s"%(param,round(Aa_est,3),round(eAa,3),round(aa_est,3),round(eaa,3),round(ba_est,3),round(eba,3),round(act_chi_sq,5)),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
      plt.title("IM of Real Deltas")
      plt.xlabel("Mass (MeV/c^2)")
      plt.ylabel("Count")
      plt.legend(loc='upper left')
      #plt.xlim(min(bins_cntr_act)-2*binsize,mc.m_max)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_act_IM_plot.png")
      plt.savefig(plot_file_path)
      #plt.show()
      plt.close()


    #related pairs
    if len(cr_IM)!=0:
      
      plt.figure()
      hist_cr,bins_cr,pack_cr=plt.hist(cr_IM,bins=np.arange(int(min(cr_IM)),int(max(cr_IM)),binsize),alpha=0)
      bins_cntr_cr=0.5*(bins_cr[:-1]+bins_cr[1:])
      xfitbw_cr=np.arange(min(bins_cntr_cr),max(bins_cntr_cr),0.5)    
      ydef_cr=bw_func(xfitbw_cr,*param)
      sclr_cr=max(hist_cr)/max(ydef_cr)
      yfit_par_cr=bw_func(xfitbw_cr,*param)
      yfit_par_cr=np.multiply(yfit_par_cr,sclr_cr)
      cr_err=np.sqrt(hist_cr)
      peak_cr=np.where(hist_cr==max(hist_cr))[0][0]
      endpoints=np.where(hist_cr[peak_cr:]<=0.01*max(hist_cr))[0]
      if len(endpoints)==0:
        endpoint=len(bins_cntr_cr)-1
      else:
        endpoint=peak_cr+endpoints[0]
      sigpts_cr=cr_err[:endpoint]
      sigpts_cr=[1 if sigs==0 else sigs for sigs in sigpts_cr]
      popt_cr,pcov_cr=curve_fit(bw_func,bins_cntr_cr[:endpoint],hist_cr[:endpoint],p0=[0.95,0.47,0.6],bounds=(lwrbnd,uprbnd),sigma=sigpts_cr,absolute_sigma=True)
      Ac_est=np.abs(popt_cr[0]*sclr_cr)
      ac_est=np.abs(popt_cr[1])
      bc_est=np.abs(popt_cr[2])    
      eAc=np.sqrt(pcov_cr[0,0])*sclr_cr
      eac=np.sqrt(pcov_cr[1,1])
      ebc=np.sqrt(pcov_cr[2,2])    
      cr_chi_sq=chi2(bins_cntr_cr[:endpoint],hist_cr[:endpoint],sigpts_cr,*popt_cr)
      yfitbw_cr=bw_func(xfitbw_cr,*popt_cr)
      fwhm_cr,hlf_val,lft,rgt,mxi=fwhm_calc(yfitbw_cr,xfitbw_cr)
      plt.plot(xfitbw_cr,yfitbw_cr,label='BW fit')
      plt.plot(xfitbw_cr,yfit_par_cr,'--',label='Function w/ given param')    
      plt.errorbar(bins_cntr_cr,hist_cr,xerr=binsize/2,yerr=cr_err,fmt='.')
      plt.hlines(y=hlf_val,xmin=xfitbw_cr[lft],xmax=xfitbw_cr[rgt],label=f'fwhm={round(fwhm_cr,3)}',colors='red')
      plt.figtext(0.75,0.75,"par=%s \n A=%s+/-%s \n a=%s+/-%s \n b=%s+/-%s \n chi_sq=%s"%(param,round(Ac_est,3),round(eAc,3),round(ac_est,3),round(eac,3),round(bc_est,3),round(ebc,3),round(cr_chi_sq,5)),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
      plt.title("IM of Related Pairs")
      plt.xlabel("Mass (MeV/c^2)")
      plt.ylabel("Count")
      plt.legend(loc='upper left')
      #plt.xlim(min(bins_cntr_cr)-2*binsize,mc.m_max)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_IM_plot.png")
      plt.savefig(plot_file_path)
      #plt.show()
      plt.close()

    #free pairs ("fake" delta)
    if len(free_IM)!=0:
      plt.figure()
      hist_f,bins_f,pack_f=plt.hist(free_IM,bins=np.arange(int(min(free_IM)),int(max(free_IM)),binsize),alpha=0)
      bins_cntr_f=0.5*(bins_f[:-1]+bins_f[1:])
      xfit_f=np.arange(min(bins_cntr_f),max(bins_cntr_f),0.5)    
      f_err=np.sqrt(hist_f)
      sigpts_f=[1 if sigs==0 else sigs for sigs in f_err]
      popt_f_e,pcov_f_e=curve_fit(exp_func,bins_cntr_f,hist_f,p0=[max(hist_f),1/mc.TDel,1],bounds=([0,0,0],[np.inf,1,np.inf]))
      #print(popt_f_e)
      f_e_chi_sq=chi2gen(exp_func,bins_cntr_f,hist_f,sigpts_f,popt_f_e)
      yfit_f_e=exp_func(xfit_f,*popt_f_e)
      plt.plot(xfit_f,yfit_f_e,label='exp fit, chi2=%s'%round(f_e_chi_sq,3))
      popt_f_p,pcov_f_p=curve_fit(poly_func,bins_cntr_f,hist_f,p0=[max(hist_f),0,0,0,0],sigma=sigpts_f,absolute_sigma=True)
      f_p_chi_sq=chi2gen(poly_func,bins_cntr_f,hist_f,sigpts_f,popt_f_p)
      yfit_f_p=poly_func(xfit_f,*popt_f_p)
      #print(popt_f_p)
      plt.plot(xfit_f,yfit_f_p,label='poly fit, chi2=%s'%round(f_p_chi_sq,3))
      
      c_opt_f,max_deg_f=poly_fit_optimize(bins_cntr_f,hist_f,max_deg=10)
      f_opt_f=np.poly1d(c_opt_f)
      
      yfit_f_opt=f_opt_f(xfit_f)
      plt.plot(xfit_f,yfit_f_opt,label="pwr %d poly"%(max_deg_f))
      '''bins_f_rscl=[i-min(bins_cntr_f) for i in bins_cntr_f]
      xfit_f_rscl=np.arange(min(bins_f_rscl),max(bins_f_rscl),0.5)
      popt_f_p_rscl,pcov_f_p_rscl=curve_fit(poly_func,bins_f_rscl,hist_f,p0=[max(hist_f),0,0,0,0],sigma=sigpts_f,absolute_sigma=True)
      print(popt_f_p_rscl)
      yfit_f_p_rscl=poly_func(np.arange(min(bins_f_rscl),max(bins_f_rscl),0.5),*popt_f_p_rscl)
      plt.plot(xfit_f-1,yfit_f_p_rscl,label='poly fit, rscl')'''

      popt_f_cmp,pcov_f_cmp=curve_fit(cmp_func,bins_cntr_f,hist_f,p0=[1/mc.TDel,1/mc.TDel,1],sigma=sigpts_f,absolute_sigma=True)
      f_cmp_chi_sq=chi2gen(cmp_func,bins_cntr_f,hist_f,sigpts_f,popt_f_cmp)
      yfit_f_cmp=cmp_func(xfit_f,*popt_f_cmp)
      plt.plot(xfit_f,yfit_f_cmp,label='cmp fit, chi2=%s'%round(f_cmp_chi_sq,3))
      
      plt.errorbar(bins_cntr_f,hist_f,xerr=binsize/2,yerr=sigpts_f,fmt='.')
      plt.title("IM of Free Pairs")
      plt.xlabel("Mass (MeV/c^2)")
      plt.ylabel("Count")
      plt.legend(loc='upper right')
      #plt.xlim(min(bins_cntr_f)-2*binsize,mc.m_max)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_free_IM_plot.png")
      plt.savefig(plot_file_path)
      #plt.show()
      plt.close()
      
      

    #################
    #addition method#
    #################
    #adding free and actual
    if len(free_IM)!=0 and len(act_IM)!=0:      
      plt.figure()
      plt.errorbar(bins_cntr,hist,xerr=binsize/2,yerr=hist_err,fmt='.',label='Recreated Deltas')
      plt.errorbar(bins_cntr_f,hist_f,xerr=binsize/2,yerr=f_err,fmt='.',label='Free Particles')
      plt.errorbar(bins_cntr_act,hist_act,xerr=binsize/2,yerr=act_err,fmt='.',label='Real Deltas')
      cmb_min=min(min(bins_cntr_f),min(bins_cntr_act))
      cmb_max=max(max(bins_cntr_f),max(bins_cntr_act))     
      cmn_hista,cmb_bins=np.histogram(free_IM,bins=np.arange(cmb_min,cmb_max,binsize))
      cmn_histb,cmb_bins=np.histogram(act_IM,bins=cmb_bins)
      cmb_hist=[a+b for a,b in zip(cmn_hista,cmn_histb)]            
      binsize_cmb=(cmb_bins[1]-cmb_bins[0])
      cmb_err=np.sqrt(cmb_hist)
      cmn_x=np.arange(cmb_min,cmb_max,0.5)
      cmb_y=[]
      cmb_y1=[]
      #scld_y=[]
      #scld_y2=[]
      cmb_y2=[]
      cmb_ye=[]
      cmb_y3=[]
      #opt_add=[]
      fit_sclr_list=[]
      for i in range(len(bins_cntr_act),len(bins_cntr_f)):
        if hist_f[i]>0:
          fit_sclr_list.append(hist[i]/hist_f[i])
      fit_sclr1=np.average(fit_sclr_list)
      mxpt=np.where(yfitbw_act==max(yfitbw_act))[0]
      fpval_at_max=poly_func(cmn_x[mxpt],*popt_f_p)[0]
      feval_at_max=exp_func(cmn_x[mxpt],*popt_f_e)[0]
      #fcmpval_at_max=exp_func(cmn_x[mxpt],*popt_f_cmp)[0]
      foptval_at_max=f_opt_f(cmn_x[mxpt][0])
      fit_sclr2=1+Ndelta*(fpval_at_max/max(hist))
      fit_sclr3=1+Ndelta*(feval_at_max/max(hist))
      #fit_sclr4=1+Ndelta*(fcmpval_at_max/max(hist))
      fit_sclr5=1+Ndelta*(foptval_at_max/max(hist))
      for i in range(0,len(xfitbw_act)):
        cmb_y.append(yfitbw_act[i]+yfit_f_p[i])
        cmb_ye.append(yfitbw_act[i]+yfit_f_e[i])
        cmb_y1.append(fit_sclr2*yfitbw_act[i]+fit_sclr1*yfit_f_p[i])
        cmb_y2.append(fit_sclr3*yfitbw_act[i]+fit_sclr1*yfit_f_e[i])
        cmb_y3.append(fit_sclr5*yfitbw_act[i]+fit_sclr1*yfit_f_opt[i])
        #cmb_y3.append(fit_sclr4*yfitbw_act[i]+fit_sclr1*yfit_f_cmp[i])
        #scld_y.append(fit_sclr1*yfit_f_p[i])
        #scld_y2.append(fit_sclr1*yfit_f_e[i])
        #opt_add.append(yfitbw_act[i]+fit_sclr1*yfit_f_opt[i])
      for i in range(len(xfitbw_act),len(xfit_f)):
        cmb_y.append(yfit_f_p[i])
        cmb_ye.append(yfit_f_e[i])
        cmb_y1.append(yfit_f_p[i]*fit_sclr1)
        cmb_y2.append(yfit_f_e[i]*fit_sclr1)
        cmb_y3.append(yfit_f_opt[i]*fit_sclr1)
        #cmb_y3.append(yfit_f_cmp[i]*fit_sclr1)
        #scld_y.append(fit_sclr1*yfit_f_p[i])
        #scld_y2.append(fit_sclr1*yfit_f_e[i])
        #opt_add.append(fit_sclr1*yfit_f_opt[i])
      xfit_diff=min(xfitbw_act)-min(xfit_f)
      cmn_x=[i+xfit_diff for i in cmn_x]

      plt.plot(xfitbw_act,yfitbw_act,label='BW Fit (Real)')
      plt.plot(xfit_f,yfit_f_p,label='poly fit (Free)')
      plt.plot(xfit_f,yfit_f_e,label='exp fit (Free)')
      plt.plot(xfit_f,yfit_f_opt,label='opt fit (Free)')
      plt.plot(cmn_x,cmb_y,label='cmb (poly)')
      plt.plot(cmn_x,cmb_ye,label='cmb (exp)')
      plt.plot(cmn_x,cmb_y1,label='cmb w/ scl (poly)')
      plt.plot(cmn_x,cmb_y2,label='cmb fit w/ scl (exp)')
      plt.plot(cmn_x,cmb_y3,label='cmb fit w/ scl (deg: %d)'%max_deg_f) 
      #plt.plot(cmn_x,opt_add,label='cmb fit (bg scl only)')
      #plt.plot(cmn_x,cmb_y3,label='combined fit w/ scaling (cmp)')      
      plt.errorbar(cmb_bins[:-1],cmb_hist,xerr=binsize_cmb/2,yerr=cmb_err,fmt='.',label="Combined")
      plt.title("Real + Free")
      plt.xlabel("Mass (MeV/c^2)")
      plt.ylabel("Count")
      plt.legend(loc='upper right')
      plt.xlim(min(bins_cntr)-2*binsize,mc.m_max)
      plt.grid
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_add_IM_plot.png")
      plt.savefig(plot_file_path)
      #plt.show()
      plt.close()
      
      '''#fitting with combined function using the existing histograms and parameters
      plt.figure()
      lwrbnd_cmb=[0,0,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,fit_sclr2-2,fit_sclr1-2]
      uprbnd_cmb=[5,5,5,np.inf,np.inf,np.inf,np.inf,np.inf,fit_sclr2+2,fit_sclr1+2]
      lwrbnd_cmb1=[0,0,0,0,0,-np.inf,fit_sclr2-2,fit_sclr1-2]
      uprbnd_cmb1=[5,5,5,np.inf,np.inf,np.inf,fit_sclr2+2,fit_sclr1+2]
      startpt=np.where(hist==max(hist))[0][0]
      #endpt=np.where(np.abs(bins_cntr-mc.m_max+200)<=1)[0][0]
      hist_diff=np.diff(hist)
      limiter=6
      hist_lim=limiter*np.std(hist_diff)
      ind_jump=np.where(np.abs(hist_diff[startpt:])>hist_lim)[0]
      while len(ind_jump)==0:
        if limiter==0:
          continue
        else:
          limiter=limiter-1
          hist_lim=limiter*np.std(hist_diff)
          ind_jump=np.where(np.abs(hist_diff[startpt:])>hist_lim)[0]
      bg_start=startpt+ind_jump[-1]+1
      cmb_fit_bins=bins_cntr[:bg_start]
      cmb_fit_hist=hist[:bg_start]
      cmb_fit_err=hist_err[:bg_start]
      cmb_fit_err=[1 if i==0 else i for i in cmb_fit_err]
      
      popt_guess,pcov_guess1=curve_fit(cmb_func_p,cmb_fit_bins,cmb_fit_hist,p0=[*param,*popt_f_p,fit_sclr2,fit_sclr1])
      popt_guess1,pcov_guess1=curve_fit(cmb_func_e,cmb_fit_bins,cmb_fit_hist,p0=[A_est,*param[1:],*popt_f_e,fit_sclr2,fit_sclr1],bounds=[lwrbnd_cmb1,uprbnd_cmb1])
      xfit_cmb=np.arange(min(bins_cntr),max(bins_cntr),0.5)
      #xjumppt=np.where(np.abs(xfit_cmb-bins_cntr[startpt+ind_jump[-1]]<1))[0]
      yfit_cmb=cmb_func_p(xfit_cmb,*popt_guess)
      yfit_cmb2=cmb_func_e(xfit_cmb,*popt_guess1)
      #bg_start=startpt+ind_jump[-1]+1
      #bg_bins=bins_cntr[bg_start:]
      #print(popt_guess)
      #print(popt_guess1)
      plt.plot(xfit_cmb,yfit_cmb,label='cmb fit w/ p')
      plt.plot(xfit_cmb,yfit_cmb2,label='cmb fit w/ e')
      plt.errorbar(bins_cntr,hist,xerr=binsize/2,yerr=hist_err,fmt='.')
      #plt.scatter(bins_cntr[startpt+ind_jump],hist[startpt+ind_jump],color='r',label='jumps')
      plt.xlim(min(bins_cntr)-2*binsize,mc.m_max)
      plt.legend(loc='upper right')
      plt.show()
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cmb_IM_plot.png")
      plt.close()'''
      
      #####################
      #subtraction methods#
      #####################
      
      #goal: "take out" the affect of the background in the IM histogram
      #step 1: find, if any, the distinguishing point(s) btwn bg and real
      #step 2: curve_fit on section with just the bg
      #step 3: extrapolate to the rest of the histogram
      #step 4: subtract fit values from the histogram
      plt.figure()
      plt.errorbar(bins_cntr,hist,xerr=binsize/2,yerr=hist_err,fmt='.',label='Recreated Deltas')
      plt.errorbar(bins_cntr_act,hist_act,xerr=binsize/2,yerr=act_err,fmt='.',label='Actual Deltas')
      #find jumps and kinks
      startpt=np.where(hist==max(hist))[0][0]
      hist_diff=np.diff(hist)
      limiter=6
      hist_lim=limiter*np.std(hist_diff)
      ind_jump=np.where(np.abs(hist_diff[startpt:])>hist_lim)[0]
      while len(ind_jump)==0:
        if limiter==0:
          continue
        else:
          limiter=limiter-1
          hist_lim=limiter*np.std(hist_diff)
          ind_jump=np.where(np.abs(hist_diff[startpt:])>hist_lim)[0]
      bg_start=startpt+ind_jump[-1]+1
      #using theoretical max of delta IM if it is greater than the jump point
      max_bin_pt=np.where(np.abs(bins_cntr-mc.md_max)<=binsize)[0][0]
      if bg_start<max_bin_pt:
        bg_start=max_bin_pt
      bg_bins=bins_cntr[bg_start:]
      bg_bins_alt=bins_cntr[:5].tolist()+bg_bins.tolist()
      #bg_bins_rscl=[ii-mc.md_min for ii in bg_bins]
      #bg_bins_alt=[ii-mc.md_min for ii in bg_bins_alt]
      bg_hist=hist[bg_start:]
      bg_hist_alt=hist[:5].tolist()+bg_hist.tolist()
      bg_err=hist_err[bg_start:]
      bg_err=[1 if i==0 else i for i in bg_err]
      bg_err_alt=np.sqrt(bg_hist_alt)
      bg_err_alt=[1 if i==0 else i for i in bg_err_alt]
      popt_bg_p,pcov_bg_p=curve_fit(poly_func,bg_bins_alt,bg_hist_alt,p0=[bg_hist[0],0,0,0,0],sigma=bg_err_alt,absolute_sigma=True)
      popt_bg_e,pcov_bg_e=curve_fit(exp_func,bg_bins,bg_hist,p0=[max(hist),2/(mc.Tpfree+mc.Tpifree),0],sigma=bg_err,absolute_sigma=True)
      #popt_bg_cmp,pcov_bg_cmp=curve_fit(cmp_func,bg_bins,bg_hist,p0=[max(hist),1/mc.TDel,0],sigma=bg_err,absolute_sigma=True)
      #popt_bg_cmp_r,pcov_bg_cmp_r=curve_fit(cmp_func,bg_bins_rscl,bg_hist,p0=[max(hist),1/mc.TDel,0],sigma=bg_err,absolute_sigma=True)
      #print(popt_bg_p)
      #popt_bg_p2,pcov_bg_p2=curve_fit(poly_func,bg_bins_alt,bg_hist_alt,p0=[bg_hist[0],0,0,0,0],sigma=bg_err_alt,absolute_sigma=True)
      opt_coef,opt_deg=poly_fit_optimize(bg_bins_alt,bg_hist_alt,10)
      opt_p=np.poly1d(opt_coef)
      bg_x=np.arange(min(bins_cntr),max(bins_cntr),0.5)
      #bg_x_r=bg_x-mc.md_min
      bg_y_p=poly_func(bg_x,*popt_bg_p)
      bg_y_e=exp_func(bg_x,*popt_bg_e)
      #bg_y_cmp=cmp_func(bg_x,*popt_bg_cmp)
      #bg_y_cmp_r=cmp_func(bg_x_r,*popt_bg_cmp_r)
      #bg_y_p2=poly_func(bg_x,*popt_bg_p2)
      bg_y_opt=opt_p(bg_x)
      bg_endpt=np.where(np.abs(bg_x-mc.m_max)<=1)[0][0]

      y_sub_p=[]
      y_sub_e=[]
      y_sub_opt=[]
      
      for i in range(0,len(hist)-1):
        xbinpt=np.where(bg_x==bins_cntr[i])[0][0]
        y_sub_p.append(hist[i]-bg_y_p[xbinpt])
        y_sub_e.append(hist[i]-bg_y_e[xbinpt])
        y_sub_opt.append(hist[i]-bg_y_opt[xbinpt])

      plt.plot(bg_x,bg_y_e,label='bg w/ exp fit')
      plt.plot(bg_x,bg_y_p,label='bg w/ poly fit')
      #plt.plot(bg_x,bg_y_cmp,label='bg w/ cmp fit')
      #plt.plot(bg_x[:bg_endpt],bg_y_p2[:bg_endpt],label='bg w/ p2 fit')
      plt.plot(bg_x,bg_y_opt,label='bg w/ opt fit deg:%d'%opt_deg)
      plt.plot(bins_cntr[:-1],y_sub_e,'.',label='- exp')
      plt.plot(bins_cntr[:-1],y_sub_p,'.',label='- poly')
      plt.plot(bins_cntr[:-1],y_sub_opt,'.',label='- opt')
      #plt.plot(bg_x,bg_y_cmp_r,label='bg w/ cmp fit (rscl)')
      plt.scatter(bins_cntr[startpt+ind_jump],hist[startpt+ind_jump],color='r',label='jumps')
      plt.scatter(bins_cntr[bg_start],hist[bg_start],label='bg start')
      plt.legend(loc='upper right')
      plt.xlim(min(bins_cntr)-2*binsize,mc.m_max)
      plt.ylim(-20,max(hist)*1.2)
      #plt.show()
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_sub_IM_plot.png")
      plt.savefig(plot_file_path)
      plt.close()

    contour=False
    if len(IM_list)!=0:
      plt.figure()
      plt.title("Correlation Between 'a' and 'b' (Recreated Delta)")
      plt.xlabel('a')
      plt.ylabel('b')
      aa=np.linspace(a_est-ea,a_est+ea,101)
      bb=np.linspace(b_est-eb,b_est+eb,101)
      z=np.zeros((len(aa),len(bb)))
      for i in range (len(aa)):
        for j in range(len(bb)):
          z[j,i]=chi2(bins_cntr,hist,hist_err,popt[0],aa[i],bb[j])-chi2(bins_cntr,hist,hist_err,*popt)
      aa,bb=np.meshgrid(aa,bb)
      cplot=plt.contour(aa,bb,z,levels=[1,2])
      plt.grid()
      plt.axhline(b_est)
      plt.axvline(a_est)
      plt.clabel(cplot,inline=1,)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cntr_plot_ab.png")
      plt.savefig(plot_file_path)    
      if contour is True:
        plt.show()
      plt.close()

      plt.figure()
      plt.title("Correlation Between 'A' and 'a' (Recreated Delta)")
      plt.xlabel('A')
      plt.ylabel('a')
      AA=np.linspace(A_est-eA,A_est+eA,101)
      aa=np.linspace(a_est-ea,a_est+ea,101)
      z=np.zeros((len(AA),len(aa)))
      for i in range (len(AA)):
        for j in range(len(aa)):
          z[j,i]=chi2(bins_cntr,hist,hist_err,AA[i]/sclr,aa[j],popt[2])-chi2(bins_cntr,hist,hist_err,*popt)
      AA,aa=np.meshgrid(AA,aa)
      cplot=plt.contour(AA,aa,z,levels=[1,2])
      plt.grid()
      plt.axhline(a_est)
      plt.axvline(A_est)
      plt.clabel(cplot,inline=1,)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cntr_plot_Aa.png")
      plt.savefig(plot_file_path)    
      if contour is True:
        plt.show()
      plt.close()

      plt.figure()
      plt.title("Correlation Between 'b' and 'A' (Recreated Delta)")
      plt.xlabel('b')
      plt.ylabel('A')
      bb=np.linspace(b_est-eb,b_est+eb,101)
      AA=np.linspace(A_est-eA,A_est+eA,101)
      z=np.zeros((len(bb),len(AA)))
      for i in range (len(bb)):
        for j in range(len(AA)):
          z[j,i]=chi2(bins_cntr,hist,hist_err,AA[j]/sclr,popt[1],bb[i])-chi2(bins_cntr,hist,hist_err,*popt)
      bb,AA=np.meshgrid(bb,AA)
      cplot=plt.contour(bb,AA,z,levels=[1,2])
      plt.grid()
      plt.axhline(A_est)
      plt.axvline(b_est)
      plt.clabel(cplot,inline=1,)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cntr_plot_bA.png")
      plt.savefig(plot_file_path)    
      if contour is True:
        plt.show()
      plt.close()
              
    if len(act_IM)!=0:
      plt.figure()
      plt.title("Correlation Between 'a' and 'b' (Actual Delta)")
      plt.xlabel('a')
      plt.ylabel('b')
      aa=np.linspace(aa_est-eaa,aa_est+eaa,101)
      bb=np.linspace(ba_est-eba,ba_est+eba,101)
      z=np.zeros((len(aa),len(bb)))
      for i in range (len(aa)):
        for j in range(len(bb)):
          z[j,i]=chi2(bins_cntr_act,hist_act,act_err,popt_act[0],aa[i],bb[j])-chi2(bins_cntr_act,hist_act,act_err,*popt_act)
      aa,bb=np.meshgrid(aa,bb)
      cplot=plt.contour(aa,bb,z,levels=[1,2])
      plt.grid()
      plt.axhline(ba_est)
      plt.axvline(aa_est)
      plt.clabel(cplot,inline=1)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_act_cntr_plot_ab.png")
      plt.savefig(plot_file_path)    
      if contour is True:
        plt.show()
      plt.close()

      plt.figure()
      plt.title("Correlation Between 'A' and 'a' (Actual Delta)")
      plt.xlabel('A')
      plt.ylabel('a')
      AA=np.linspace(Aa_est-eAa,Aa_est+eAa,101)
      aa=np.linspace(aa_est-eaa,aa_est+eaa,101)
      z=np.zeros((len(aa),len(bb)))
      for i in range (len(AA)):
        for j in range(len(aa)):
          z[j,i]=chi2(bins_cntr_act,hist_act,act_err,AA[i]/sclr_act,aa[j],popt_act[2])-chi2(bins_cntr_act,hist_act,act_err,*popt_act)
      AA,aa=np.meshgrid(AA,aa)
      cplot=plt.contour(AA,aa,z,levels=[1,2])
      plt.grid()
      plt.axhline(aa_est)
      plt.axvline(Aa_est)
      plt.clabel(cplot,inline=1)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_act_cntr_plot_Aa.png")
      plt.savefig(plot_file_path)    
      if contour is True:
        plt.show()
      plt.close()

      plt.figure()
      plt.title("Correlation Between 'b' and 'A' (Actual Delta)")
      plt.xlabel('b')
      plt.ylabel('A')
      bb=np.linspace(ba_est-eba,b_est+eba,101)
      AA=np.linspace(Aa_est-eAa,A_est+eAa,101)
      z=np.zeros((len(bb),len(AA)))
      for i in range (len(bb)):
        for j in range(len(AA)):
          z[j,i]=chi2(bins_cntr_act,hist_act,act_err,AA[j]/sclr_act,popt_act[1],bb[i])-chi2(bins_cntr_act,hist_act,act_err,*popt_act)
      bb,AA=np.meshgrid(bb,AA)
      cplot=plt.contour(bb,AA,z,levels=[1,2])
      plt.grid()
      plt.axhline(Aa_est)
      plt.axvline(ba_est)
      plt.clabel(cplot,inline=1)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_act_cntr_plot_bA.png")
      plt.savefig(plot_file_path)    
      if contour is True:
        plt.show()
      plt.close()
    

    if len(cr_IM)!=0:
      plt.figure()
      plt.title("Correlation Between 'a' and 'b' (Related Delta)")
      plt.xlabel('a')
      plt.ylabel('b')
      aa=np.linspace(ac_est-eac,ac_est+eac,101)  
      bb=np.linspace(bc_est-ebc,bc_est+ebc,101)
      z=np.zeros((len(aa),len(bb)))
      for i in range (len(aa)):
        for j in range(len(bb)):
          z[j,i]=chi2(bins_cntr_cr,hist_cr,cr_err,popt_cr[0],aa[i],bb[j])-chi2(bins_cntr_cr,hist_cr,cr_err,*popt_cr)
      aa,bb=np.meshgrid(aa,bb)
      cplot=plt.contour(aa,bb,z,levels=[1,2])
      plt.grid()
      plt.axhline(bc_est)
      plt.axvline(ac_est)
      plt.clabel(cplot,inline=1)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_cntr_plot_ab.png")
      plt.savefig(plot_file_path)
      if contour is True:
        plt.show()
      plt.close()

      plt.figure()
      plt.title("Correlation Between 'A' and 'a' (Related Delta)")
      plt.xlabel('A')
      plt.ylabel('a')
      AA=np.linspace(Ac_est-eAc,Ac_est+eAc,101)
      aa=np.linspace(ac_est-eac,ac_est+eac,101)
      z=np.zeros((len(AA),len(aa)))
      for i in range (len(AA)):
        for j in range(len(aa)):
          z[j,i]=chi2(bins_cntr_cr,hist_cr,cr_err,AA[i]/sclr_cr,aa[j],popt_cr[2])-chi2(bins_cntr_cr,hist_cr,cr_err,*popt_cr)
      AA,aa=np.meshgrid(AA,aa)
      cplot=plt.contour(AA,aa,z,levels=[1,2])
      plt.grid()
      plt.axhline(ac_est)
      plt.axvline(Ac_est)
      plt.clabel(cplot,inline=1)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_cntr_plot_Aa.png")
      plt.savefig(plot_file_path)
      if contour is True:
        plt.show()
      plt.close()


      plt.figure()
      plt.title("Correlation Between 'b' and 'A' (Related Delta)")
      plt.xlabel('b')
      plt.ylabel('A')
      bb=np.linspace(bc_est-ebc,bc_est+ebc,101)
      AA=np.linspace(Ac_est-eAc,Ac_est+eAc,101)
      z=np.zeros((len(bb),len(AA)))
      for i in range (len(bb)):
        for j in range(len(AA)):
          z[j,i]=chi2(bins_cntr_cr,hist_cr,cr_err,AA[j]/sclr_cr,popt_cr[1],bb[i])-chi2(bins_cntr_cr,hist_cr,cr_err,*popt_cr)
      bb,AA=np.meshgrid(bb,AA)
      cplot=plt.contour(bb,AA,z,levels=[1,2])
      plt.grid()
      plt.axhline(Ac_est)
      plt.axvline(bc_est)
      plt.clabel(cplot,inline=1)
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_cntr_plot_bA.png")
      plt.savefig(plot_file_path)
      if contour is True:
        plt.show()
      plt.close()

    rmnt=False
    if len(cr_mnt)!=0:
      plt.figure()
      hist_cr_mnt,bins_cr_mnt,pack_cr_mnt=plt.hist(cr_mnt,bins=np.arange(0,int(max(cr_mnt))+1,binsize))
      plt.title("Momenta of Related Pairs")
      plt.xlabel("Momentum (MeV/c)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_mnt_plot.png")
      plt.savefig(plot_file_path)
      if rmnt is True:
        plt.show()
      plt.close()

    indv_mnt=False
    #momenta of protons and pions
    if len(p_mnt)!=0:
      binsize_indv=5
      plt.figure()
      hist_p,bins_p,pack_p=plt.hist(p_mnt,bins=np.arange(0,int(max(p_mnt))+1,binsize_indv))
      plt.title("Momenta of Protons")
      plt.xlabel("Momentum (MeV/c)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_p_mnt_plot.png")
      plt.savefig(plot_file_path)
      if indv_mnt is True:
        plt.show()
      plt.close()

    if len(pi_mnt)!=0:
      plt.figure()
      hist_pi,bins_pi,pack_pi=plt.hist(pi_mnt,bins=np.arange(0,int(max(pi_mnt))+1,binsize_indv))
      plt.title("Momenta of Pions")
      plt.xlabel("Momentum (MeV/c)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_pi_mnt_plot.png")
      plt.savefig(plot_file_path)
      if indv_mnt is True:
        plt.show()
      plt.close()

    dmnt=False
    if len(momentum_list)!=0:
      #mnt of deltas
      plt.figure()
      hist_rec,bins_rec,pack_rec=plt.hist(momentum_list,bins=np.arange(0,int(max(momentum_list))+1,binsize_indv))
      plt.title("Momenta of Recreated Deltas")
      plt.xlabel("Momentum (MeV/c)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_rec_plot.png")
      plt.savefig(plot_file_path)
      if dmnt is True:
        plt.show()
      plt.close()

    if len(act_list)!=0:
      plt.figure()
      hist_act,bins_act,packages_act=plt.hist(act_list,bins=bins_rec)
      plt.title("Momenta of Actual Deltas")
      plt.xlabel("Momentum (MeV/c)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_act_plot.png")
      plt.savefig(plot_file_path)
      if dmnt is True:
        plt.show()
      plt.close()

    #energy of protons and pions
    if len(p_en)!=0:
      binsize_E=2
      indv_E=False
      plt.figure()
      hist_pE,bins_pE,pack_pE=plt.hist(p_en,bins=np.arange(0,int(max(p_en))+1,binsize_E))
      plt.title("Energy of Protons")
      plt.xlabel("Energy (MeV/c^2)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_p_en_plot.png")
      plt.savefig(plot_file_path)
      if indv_E is True:
        plt.show()
      plt.close()

    if len(pi_en)!=0:
      plt.figure()
      hist_piE,bins_piE,pack_piE=plt.hist(pi_en,bins=np.arange(0,int(max(pi_en))+1,binsize_E))
      plt.title("Energy of Pions")
      plt.xlabel("Energy (MeV/c^2)")
      plt.ylabel("Count")
      plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_pi_en_plot.png")
      plt.savefig(plot_file_path)
      if indv_E is True:
        plt.show()
      plt.close()   
    
    #efficiency over mnt
    eff_list=[]
    eff_err=[]
    
    if len(act_IM)!=0:
      for i in range(0,len(bins_act)-1):
        if hist_rec[i] == 0 or hist_act[i]==0:
          eff_list.append(0)
          eff_err.append(0)
        else:
          eff_list.append(hist_act[i]/hist_rec[i])
          rec_err=np.sqrt(hist_rec[i]*(1-hist_rec[i]/len(hist_rec)))
          act_err=np.sqrt(hist_act[i]*(1-hist_act[i]/len(hist_act)))
          eff_err.append((hist_act[i]/hist_rec[i])*np.sqrt((act_err/hist_act[i])**2+(rec_err/hist_rec[i])**2))
  print()
  return IM_all,act_all,cr_all,mnt_all

#read files
abs_path=os.path.dirname(__file__)

for dn in range(1,len(mc.Dlist)):
  for fn in range(1,len(mc.Flist)):
    if dn==0 and fn==0:
      continue
    print("Ndelta:",mc.Dlist[dn],",","Nfree:",mc.Flist[fn])
    ptcl_dir='D_%d_F_%d'%(mc.Dlist[dn],mc.Flist[fn])
    directoryA=os.path.join(abs_path,ptcl_dir,'bw_A')
    directorya=os.path.join(abs_path,ptcl_dir,'bw_qa')
    directoryb=os.path.join(abs_path,ptcl_dir,'bw_qb')
    file_patternA="BW_A_*.csv"
    file_patterna="BW_a_*.csv"
    file_patternb="BW_b_*.csv"
    #output folders
    graph_folderA=os.path.join(directoryA,"results")
    graph_foldera=os.path.join(directorya,"results")
    graph_folderb=os.path.join(directoryb,"results")
    os.makedirs(graph_folderA,exist_ok=True)
    os.makedirs(graph_foldera,exist_ok=True)
    os.makedirs(graph_folderb,exist_ok=True)

    reader(directoryA,file_patternA,graph_folderA)
    print()
    reader(directorya,file_patterna,graph_foldera)
    print()
    reader(directoryb,file_patternb,graph_folderb)
  
