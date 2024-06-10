import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the functions
def f(x):
    return x**2 + 2*x + 1

def g(x):
    return 2*x + 3

# Define the function to find roots for (f(x) - g(x) = 0)
def h(x):
    return f(x) - g(x)

# Use fsolve to find the roots of h(x) = 0 within a specific range
# We need to iterate over a range to ensure we capture all intersection points
x_values = np.linspace(-10, 10, 400)
initial_guesses = []

# Adding initial guesses based on a grid search
for x in x_values:
    initial_guesses.append(x)

# Finding unique roots
roots = []
for guess in initial_guesses:
    root = fsolve(h, guess)[0]
    # Checking if the root is a duplicate or not by comparing with existing roots
    if np.isclose(root, roots, atol=1e-5).any():
        continue
    else:
        roots.append(root)

# Plot the functions and their intersection points
x = np.linspace(-10, 10, 400)
y_f = f(x)
y_g = g(x)

plt.plot(x, y_f, label='$f(x) = x^2 + 2x + 1$')
plt.plot(x, y_g, label='$g(x) = 2x + 3$')

# Plot intersection points
for root in roots:
    plt.plot(root, f(root), 'ro')  # Intersection points
    plt.annotate(f'({root:.2f}, {f(root):.2f})', (root, f(root)), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Intersection of two graphs')
plt.show()

# Print the intersection points
print("Intersection points:")
for root in roots:
    print(f"x = {root:.2f}, y = {f(root):.2f}")
