import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, peak_widths
#import scipy.stats as st
from scipy.integrate import simpson
#from scipy.interpolate import interp1d
import random

#load constants
import myconst as mc



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