import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import myconst as mc
import os
import glob


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
      #momentum cut
      if pt_mag < mc.p_cut:        
        m_list.append(mdel_rec)
        mnt_list.append(pmag)
  return m_list, mnt_list

#for fitting
def poly_func(x,c0,c1,c2,c3,c4):
  return c0+c1*x+c2*x**2+c3*x**3+c4*x**4

def gaus_func(x,A,x0,sig):
  return A*np.exp(-((x-x0)**2)/(2*sig**2))/(np.sqrt(2*np.pi)*sig)

def bw_func(x,A,a,b):
  q=np.sqrt((x**2-mc.m_n**2-mc.m_pi**2)**2-4*(mc.m_n*mc.m_pi)**2)/(2*x)
  gam=(a*q**3)/(mc.m_pi**2+b*q**2)
  return (4*gam*mc.m_del0**2)/(A*(x**2-mc.m_del0**2)**2+(mc.m_del0*gam)**2)

def fwhm_calc(data,bins):
  max_value=np.max(data)
  max_ind=np.argmax(data)
  half_max_val=max_value/2
  #find where the data crosses the half piont on the left and right of the peak
  left_ind=np.argmin(np.abs(data[0:max_ind]-half_max_val))
  right_ind=np.argmin(np.abs(data[max_ind:]-half_max_val))+max_ind
  fwhm=bins[right_ind]-bins[left_ind]

  return fwhm,half_max_val,left_ind,right_ind,max_ind

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

fitting=False

def reader(directory,file_pattern,output_folder):
#for comparing different widths
  IM_all=[]
  act_all=[]
  cr_all=[]
  mnt_all=[]

  for filename in glob.glob(os.path.join(directory,file_pattern)):
    new_file_path= os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_IM.csv")
    with open(new_file_path,"w",newline='') as new_file:
      new_file.close()

    IM_list=[] #the invariant mass of the recreated deltas from all events
    act_list=[] #momenta of "real" deltas
    act_IM=[] #inv mass of "real" deltas
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
            dpmag=dist_form(rowdata[2:4])
            dm=inv_m(float(rowdata[1]),dpmag)
            act_IM.append(dm)
          elif PID==2212: #proton
            p_list.append(rowdata[1:5])
            p_en.append(float(rowdata[1]))
            p_mnt.append(dist_form(rowdata[2:4]))
            if Par_ID==0:
              continue
            else:
              plist[f'p{Par_ID}_list'].append(rowdata[1:5])
          elif PID==211: #pion+
            pi_list.append(rowdata[1:5])
            pi_en.append(float(rowdata[1]))
            pi_mnt.append(dist_form(rowdata[2:4]))
            if Par_ID==0:
              continue
            else:
              pilist[f'pi{Par_ID}_list'].append(rowdata[1:5])       
        #invariant mass of the p and pi in the event with momentum cut applied
        #random.shuffle(p_list)
        #random.shuffle(pi_list)
        m_list,mnt_list=IM_method(p_list,pi_list)

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
    #mass cut done with the fitting
    binsize=1 #in MeV/c^2
    plt.figure()
    hist,bins,packages=plt.hist(IM_list,bins=np.arange(int(mc.md_min)-1,int(mc.md_max)+1,binsize))
    if fitting is True:
      stp=int(mc.m_cut/binsize) #determines the width of the cut
      x_omit=int(np.where(bins==mc.m_del0)[0][0]) #omit the inv mass of delta
      #data to be consider for the fitting of the "noise"
      x_start=np.where(hist>0.05*max(hist))[0][0]
      x_end=np.where(hist[x_start:]<0.05*max(hist))[0][0]
      x_new=bins[x_start:x_omit-stp].tolist()+bins[x_omit+stp+1:x_start+x_end].tolist()
      y_new=hist[x_start:x_omit-stp].tolist()+hist[x_omit+stp+1:x_start+x_end].tolist()
      #data to be considered for counting the number of deltas
      x_skipped=bins[x_omit-stp:x_omit+stp]
      y_skipped=hist[x_omit-stp:x_omit+stp]
      print(x_skipped)
      #fitting
      xplt=np.arange(bins[x_start],bins[x_start+x_end],0.5)
      ini_g=[0,0,0,0,0]
      popt,pcov=curve_fit(poly_func, x_new,y_new,ini_g)
      yplt=poly_func(xplt,*popt)
      r2_poly=r2_calc(poly_func,x_new,y_new,popt)
      r=str(round(r2_poly,5))
      plt.plot(xplt,yplt,label='poly fit \n R^2=%s'%(r))
      plt.plot(x_skipped,y_skipped,'.')
      #plt.plot(x_skipped,poly_func(x_skipped,*popt),'.')
      #guessing count
      y_est=[]
      for i in range(0,len(x_skipped)):
        xi=np.where(xplt==x_skipped[i])[0][0]
        y_est.append(y_skipped[i]-yplt[xi])

      print("estimated number of deltas:",sum(y_est))

    param_norm,cov_norm=curve_fit(gaus_func,bins[:-1],hist,p0=[max(hist),mc.m_del0,1])
    xfitnorm=np.arange(min(bins),max(bins),0.5)
    yfitnorm=gaus_func(xfitnorm,*param_norm)
    fwhm=param_norm[2]*2*(2*np.log(2))**0.5
    plt.plot(xfitnorm,yfitnorm,label='normal fit')
    plt.title("Invariant Mass of Proton and Pion Pairs in Lab Frame")
    plt.ylabel("Count")
    plt.xlabel("Mass (MeV/c^2)")
    plt.legend(loc='upper right')
    plt.ylim(0,max(hist)*1.1)
    plt.figtext(0.75,0.65,"m_err=%d \n p_min=%d \n mu=%s \n std=%s \n fwhm=%s"%(mc.m_cut,mc.p_cut,round(param_norm[1],3),round(param_norm[2],3),round(fwhm,3)),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_IM_plot.png")
    plt.savefig(plot_file_path)
    plt.show()
    plt.close()

    print("total number of counted particles after momentum cut:", np.sum(hist))

    rlt=False
    #related pairs
    binsize_new=2
    plt.figure()
    hist_cr,bins_cr,pack_cr=plt.hist(cr_IM,bins=np.arange(int(min(cr_IM))-1,int(max(cr_IM))+1,binsize_new))
    param_norm_cr,cov_norm_cr=curve_fit(gaus_func,bins_cr[:-1],hist_cr,p0=[max(hist_cr),mc.m_del0,1])
    fwhm_cr=param_norm_cr[2]*2*(2*np.log(2))**0.5
    xfitnorm_cr=np.arange(min(bins_cr),max(bins_cr),0.5)
    yfitnorm_cr=gaus_func(xfitnorm_cr,*param_norm_cr)
    plt.plot(xfitnorm_cr,yfitnorm_cr,label='normal fit')
    plt.figtext(0.75,0.65,str("mu=%s \n std=%s"%(round(param_norm_cr[1],3),round(param_norm_cr[2],3))).rstrip('0').rstrip('.'),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='none',edgecolor='black'))
    plt.title("IM of Related Pairs")
    plt.xlabel("Mass (MeV/c^2)")
    plt.ylabel("Count")
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_IM_plot.png")
    plt.savefig(plot_file_path)
    if rlt is True:
      plt.show()
    plt.close()


    plt.figure()
    hist_cr_mnt,bins_cr_mnt,pack_cr_mnt=plt.hist(cr_mnt,bins=np.arange(0,int(max(cr_mnt))+1,binsize_new))
    plt.title("Momenta of Related Pairs")
    plt.xlabel("Momentum (MeV/c)")
    plt.ylabel("Count")
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_cr_mnt_plot.png")
    plt.savefig(plot_file_path)
    if rlt is True:
      plt.show()
    plt.close()

    indv_mnt=False
    #momenta of protons and pions
    binsize_new=5
    plt.figure()
    hist_p,bins_p,pack_p=plt.hist(p_mnt,bins=np.arange(0,int(max(p_mnt))+1,binsize_new))
    plt.title("Momenta of Protons")
    plt.xlabel("Momentum (MeV/c)")
    plt.ylabel("Count")
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_p_mnt_plot.png")
    plt.savefig(plot_file_path)
    if indv_mnt is True:
      plt.show()
    plt.close()

    binsize_new=5
    plt.figure()
    hist_pi,bins_pi,pack_pi=plt.hist(pi_mnt,bins=np.arange(0,int(max(pi_mnt))+1,binsize_new))
    plt.title("Momenta of Pions")
    plt.xlabel("Momentum (MeV/c)")
    plt.ylabel("Count")
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_pi_mnt_plot.png")
    plt.savefig(plot_file_path)
    if indv_mnt is True:
      plt.show()
    plt.close()

    dmnt=False
    #mnt of deltas
    binsize_new=5
    plt.figure()
    hist_rec,bins_rec,pack_rec=plt.hist(momentum_list,bins=np.arange(0,int(max(momentum_list))+1,binsize_new))
    plt.title("Momenta of Recreated Deltas")
    plt.xlabel("Momentum (MeV/c)")
    plt.ylabel("Count")
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_rec_plot.png")
    plt.savefig(plot_file_path)
    if dmnt is True:
      plt.show()
    plt.close()

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
    indv_E=False
    plt.figure()
    hist_pE,bins_pE,pack_pE=plt.hist(p_en,bins=np.arange(0,int(max(p_en))+1,binsize_new))
    plt.title("Energy of Protons")
    plt.xlabel("Energy (MeV/c^2)")
    plt.ylabel("Count")
    plot_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_p_en_plot.png")
    plt.savefig(plot_file_path)
    if indv_E is True:
      plt.show()
    plt.close()

    plt.figure()
    hist_piE,bins_piE,pack_piE=plt.hist(pi_en,bins=np.arange(0,int(max(pi_en))+1,binsize_new))
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

    for i in range(0,len(bins_act)-1):
      if hist_rec[i] == 0 or hist_act[i]==0:
        eff_list.append(0)
        eff_err.append(0)
      else:
        eff_list.append(hist_act[i]/hist_rec[i])
        rec_err=np.sqrt(hist_rec[i]*(1-hist_rec[i]/len(hist_rec)))
        act_err=np.sqrt(hist_act[i]*(1-hist_act[i]/len(hist_act)))
        eff_err.append((hist_act[i]/hist_rec[i])*np.sqrt((act_err/hist_act[i])**2+(rec_err/hist_rec[i])**2))

  return IM_all,act_all,cr_all,mnt_all

#read files
directoryA='param_test/bw_A'
directorya='param_test/bw_qa'
directoryb='param_test/bw_qb'
file_patternA="BW_A_*.csv"
file_patterna="BW_a_*.csv"
file_patternb="BW_b_*.csv"
#output folder
graph_folderA=os.path.join("param_test","bw_A","results")
graph_foldera=os.path.join("param_test","bw_qa","results")
graph_folderb=os.path.join("param_test","bw_qb","results")
os.makedirs(graph_folderA,exist_ok=True)
os.makedirs(graph_foldera,exist_ok=True)
os.makedirs(graph_folderb,exist_ok=True)
reader(directoryA,file_patternA,graph_folderA)