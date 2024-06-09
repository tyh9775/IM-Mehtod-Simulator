import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the individual functions
def func1(x, a, b):
    return a * x + b

def func2(x, c, d):
    return c * np.sin(d * x)

# Define the composite function that linearly combines and scales func1 and func2
def composite_func(x, a, b, c, d, scale1, scale2):
    return scale1 * func1(x, a, b) + scale2 * func2(x, c, d)

# Example data
x_data = np.linspace(0, 10, 100)
# Generate some synthetic data for demonstration purposes
# true parameters: a=2, b=1, c=1.5, d=0.5, scale1=1.0, scale2=0.5
y_data = composite_func(x_data, 2, 1, 1.5, 0.5, 1.0, 0.5) + 0.5 * np.random.normal(size=len(x_data))

# Perform the fit
initial_guess = [1, 1, 1, 1, 1, 1]  # Initial guess for the parameters
params, covariance = curve_fit(composite_func, x_data, y_data, p0=initial_guess)

# Extract the fitted parameters
a_fit, b_fit, c_fit, d_fit, scale1_fit, scale2_fit = params

# Print the fitted parameters
print(f"Fitted parameters: a={a_fit}, b={b_fit}, c={c_fit}, d={d_fit}, scale1={scale1_fit}, scale2={scale2_fit}")

# Plot the data and the fitted function
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, composite_func(x_data, *params), label='Fitted function', color='red')
plt.legend()
plt.show()
