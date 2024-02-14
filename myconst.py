#constant values to be used across every script

m_del0=1232 #MeV/c^2 - rest mass of delta resonance
m_p=938.272 #MeV/c^2 - rest mass of proton
m_n=939.565 #MeV/c^2 - rest mass of neutron
m_pi=139.570 #MeV/c^2 - rest mass of charged pion
Eb=270 #Beam energy per nucleon (AMeV)
rt_s=2015+Eb #energy at the center of collision (sqrt of s)
#2015 MeV is the minimum C.M. energy of the collision for the energy-dependent isospin-averaged isotropic cross section to be non-zero
#may have to come up with a more dynamic way to adjust rt_s

p_cut=[10,100]
m_cut=[2]