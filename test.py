import numpy as np
import matplotlib.pyplot as plt


# Generate a synthetic histogram data
np.random.seed(42)
hist_data = np.random.randn(1000)

# Compute histogram
hist, bin_edges = np.histogram(hist_data, bins=100)

# Calculate differences in histogram counts
hist_diffs = np.diff(hist)

# Define a threshold for significant changes in the histogram
hist_threshold = 3 * np.std(hist_diffs)

# Identify bin indices where the difference exceeds the threshold
hist_kinks_jumps_indices = np.where(np.abs(hist_diffs) > hist_threshold)[0]

# Plot the histogram and highlight kinks and jumps
plt.figure(figsize=(14, 7))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge", alpha=0.7)
for idx in hist_kinks_jumps_indices:
    plt.bar(bin_edges[idx:idx+2], hist[idx:idx+2], width=np.diff(bin_edges[idx:idx+2]), color='red', align='edge')
plt.title('Histogram with Identified Kinks and Jumps')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

