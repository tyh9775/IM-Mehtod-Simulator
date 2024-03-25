import numpy as np
import matplotlib.pyplot as plt

# Constants
k_J = 1.380649e-23  # Boltzmann constant in J/K
k_MeV = k_J * 6.242e12  # Boltzmann constant in MeV/K
T = 300  # Temperature in Kelvin

# Energy range
E_min_J = 0  # Minimum energy in Joules
E_max_J = 10  # Maximum energy in Joules

# Convert energy range to MeV
E_min_MeV = E_min_J * 6.242e12
E_max_MeV = E_max_J * 6.242e12

# Number of samples
num_samples = 10000

# Calculate the scale parameter for the exponential distribution
scale_parameter = 1 / (k_MeV * T)

# Generate random numbers following the Boltzmann distribution
energies = np.random.exponential(scale=scale_parameter, size=num_samples)

# Filter out energies outside the specified range
energies = energies[(energies >= E_min_MeV) & (energies <= E_max_MeV)]

# Check if energies array is empty
if len(energies) > 0:
    # Plot histogram
    x=plt.hist(energies, bins=50)
    plt.title('Boltzmann Distribution')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Probability Density')
    plt.show()
else:
    print("No energies within the specified range.")

print(np.sum(x[0]))