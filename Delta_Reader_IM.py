import numpy as np
import random
import myconst as mc
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
    vtot.append(v1[i]+v2[i+1]) 
  return vtot


#momenta of protons and pions
p_list=[]
pi_list=[]

with open("data1s.csv",'r') as file:
  f=csv.reader(file, delimiter=',')
  for row in f:
    if PID(row[0])==0:
      pi_list.append(row[1:])
    else:
      p_list.append(row[1:])
  file.close()

