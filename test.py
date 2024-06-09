import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Generate synthetic data
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data = 3 * x_data**2 + 2 * x_data + 1 + 10 * np.random.normal(size=len(x_data))

# Perform a polynomial fit (degree 2 in this case)
degree = 2
coeffs = np.polyfit(x_data, y_data, degree)

# Create a Polynomial object for easy manipulation
poly = Polynomial(coeffs[::-1])  # np.polyfit returns coefficients in reverse order

# Define the shift
x_shift = 2.0

# Shift the polynomial
def shifted_polynomial(x, poly, shift):
    return poly(x - shift)

# Plot the original data and the polynomial fit
x_fit = np.linspace(0, 10, 200)
y_fit = poly(x_fit)
y_fit_shifted = shifted_polynomial(x_fit, poly, x_shift)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, label='Polynomial Fit', color='orange')
plt.plot(x_fit, y_fit_shifted, label=f'Shifted Polynomial (x - {x_shift})', color='green')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fit and Shifted Polynomial')
plt.show()

print("Original polynomial coefficients:", coeffs)
