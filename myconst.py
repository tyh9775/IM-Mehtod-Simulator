#constant values to be used across every script

m_del0=1232 #MeV/c^2 - rest mass of delta resonance
m_p=938.272 #MeV/c^2 - rest mass of proton
m_n=939.565 #MeV/c^2 - rest mass of neutron
m_pi=139.570 #MeV/c^2 - rest mass of charged pion
Eb=270 #Beam energy per nucleon (AMeV)
rt_s=2015+Eb #energy at the center of collision (sqrt of s)
#2015 MeV is the minimum C.M. energy of the collision for the energy-dependent isospin-averaged isotropic cross section to be non-zero
#may have to come up with a more dynamic way to adjust rt_s

#bounds for the Breit-Wigner distribution of the delta mass in lab frame
md_min=m_p+m_pi
md_max=rt_s-m_p 

#maximum momentum from the maximum mass
pd_max=(((rt_s**2+md_min**2-m_p**2)/(2*rt_s))**2-md_min**2)**0.5

#number of events
nevts=1000

#numbers of particles/pairs generated
Dlist=[0,1,2,5]
Flist=[0,1,2,5]

p_cut=10 #momentum restriction on the particles in delta frame
m_cut=2 #mass error bar for delta 
