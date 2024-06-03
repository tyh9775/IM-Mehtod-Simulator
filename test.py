import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data (replace with your actual data)
x_data = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
y_data = np.array([960.1, 480.7, 241.1, 120.3, 60.5, 31.2, 16.3, 8.5, 4.1, 2.3])

# Define the exponential decay function
def exponential_decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# Initial guess for the parameters
initial_guess = [1, 0.001, 1]

# Perform the curve fit
params, covariance = curve_fit(exponential_decay_model, x_data, y_data, p0=initial_guess)

# Get the fitted curve
fitted_y_data = exponential_decay_model(x_data, *params)

# Print the parameters
print(f"Fitted parameters: a = {params[0]}, b = {params[1]}, c = {params[2]}")

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, fitted_y_data, label='Fitted curve', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Decay Fit')
plt.show()
