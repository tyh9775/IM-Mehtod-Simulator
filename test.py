import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate some sample data
mean = 0
std_dev = 1
x = np.linspace(-5, 5, 1000)
pdf = norm.pdf(x, mean, std_dev)

# Find the maximum value and its index
max_value = np.max(pdf)
max_index = np.argmax(pdf)

# Find the points where the PDF reaches half of the maximum value
half_max_value = max_value / 2
# Points to the left and right of the peak where PDF crosses half maximum
left_index = np.argmin(np.abs(pdf[:max_index] - half_max_value))
right_index = np.argmin(np.abs(pdf[max_index:] - half_max_value)) + max_index
# FWHM is the distance between these points
fwhm = x[right_index] - x[left_index]

# Plot the PDF along with the FWHM
plt.plot(x, pdf, label='PDF')
plt.axhline(half_max_value, color='red', linestyle='--', xmin=(x[left_index] - x[0]) / (x[-1] - x[0]), xmax=(x[right_index] - x[0]) / (x[-1] - x[0]))
plt.axvline(x[left_index], color='green', linestyle='--')
plt.axvline(x[right_index], color='green', linestyle='--', label='FWHM')
plt.legend()
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('PDF with Full Width Half Maximum (FWHM)')
plt.grid(True)
plt.show()

print("Full Width Half Maximum (FWHM):", fwhm)
