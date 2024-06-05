import numpy as np
import matplotlib.pyplot as plt

# Example histograms data
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(1, 2, 1500)

# Identify the larger dataset
if len(data1) > len(data2):
    larger_data = data1
    smaller_data = data2
else:
    larger_data = data2
    smaller_data = data1

# Compute histogram for the larger dataset
larger_hist, larger_bin_edges = np.histogram(larger_data, bins='auto')

# Rebin the smaller dataset using the bin edges from the larger dataset
smaller_hist, _ = np.histogram(smaller_data, bins=larger_bin_edges)

# Add the histograms
combined_hist = larger_hist + smaller_hist

# Plot the histograms
plt.hist(larger_data, bins=larger_bin_edges, alpha=0.5, label='Larger Data', color='blue')
plt.hist(smaller_data, bins=larger_bin_edges, alpha=0.5, label='Smaller Data', color='red')
plt.hist(np.concatenate([data1, data2]), bins=larger_bin_edges, alpha=0.5, label='Combined', color='green')
plt.legend()
plt.show()
