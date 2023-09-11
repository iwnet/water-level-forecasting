import numpy as np
import data_loading
import matplotlib.pyplot as plt

# Assuming 'time_series' is your time series data
# Replace this with your actual time series data

station = "Pfelling -130"

df = data_loading.readDataset()
series = df[station]

# Step 1: Calculate differences
differences = np.abs(np.diff(series))

# Step 2: Count percentage of items with differences less than a threshold
threshold = 25
percentage_less_than_threshold = np.sum(differences < threshold) / len(differences) * 100

# Print the percentage for verification
print(f"Percentage of differences less than {threshold}: {percentage_less_than_threshold:.2f}%")

# Step 3: Plot the results
plt.hist(differences, bins=100, edgecolor='black')
plt.title('Distribution of Differences Between Consecutive Points')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold}')
plt.legend()
plt.savefig("water_level_variation.png")
