import matplotlib.pyplot as plt
import numpy as np

# Given data
peak_performance = 4.24  # GINTOP/s
peak_vector_performance = 18.96  # GINTOP/s
peak_bandwidth = 51  # GB/s
peak_vector_bandwidth = 151 # GB/s
operational_intensity = 1.77  # Operations/Byte
performance = 0.20  # GINTOP/s

# Generate operational intensities for the roofline plot
extended_oi = np.linspace(0.01, 100, 1000)

# Roofline limits with extended operational intensities
extended_compute_roof = np.minimum(peak_performance, peak_bandwidth * extended_oi)

# Plotting the extended roofline model
plt.figure(figsize=(10, 7))
plt.axhline(y=peak_performance, color='black', linestyle='--', label='Peak Performance')
plt.axhline(y=peak_vector_performance, color='black', linestyle='--', label='Peak Vector Performance')
plt.axvline(x=peak_performance / peak_bandwidth, color='black', linestyle='--', label='Ridge Point')
plt.axvline(x=peak_vector_performance / peak_vector_bandwidth, color='black', linestyle='--', label='Ridge Point')

# Plotting the extended peak bandwidth line
plt.plot(extended_oi, peak_bandwidth * extended_oi, label='Peak Bandwidth', color='black', linestyle=':')
plt.plot(extended_oi, peak_vector_bandwidth * extended_oi, label='Peak Vector Bandwidth', color='black', linestyle=':')

# Scatter plot for given operational intensity and performance with a red cross
plt.scatter(operational_intensity, performance, color='red', s=100, zorder=5, label='Given Performance')

plt.yscale('log')
plt.xscale('log')
plt.xlim(0.01, 100)
plt.ylim(0.01, 100)
plt.xlabel('Operational Intensity (Operations/Byte)')
plt.ylabel('Performance (GINTOP/s)')
plt.title('Roofline Model')

# Annotate the peak bandwidth line with "Peak Scalar L1 Bandwidth" moved further down to fit inside the plot
plt.text(2, peak_bandwidth * 1.2, 'Peak Scalar L1 Bandwidth', color='black', fontsize=12, rotation=0)
plt.text(2, peak_vector_bandwidth * 0.5, 'Peak Vector L1 Bandwidth', color='black', fontsize=12, rotation=0)

# Annotate the peak performance line with "Peak Scalar Add"
plt.text(1, peak_performance, 'Peak Scalar Add', color='black', fontsize=12, ha='left', va='bottom')
plt.text(1, peak_vector_performance, 'Peak Vector Add', color='black', fontsize=12, ha='left', va='bottom')

# Annotate the regions as "memory bound" and "compute bound"
ridge_point = peak_performance / peak_bandwidth
plt.text(ridge_point / 3, 0.1, 'Memory Bound', color='grey', fontsize=12, ha='center', va='center')
plt.text(ridge_point * 10, 0.1, 'Compute Bound', color='grey', fontsize=12, ha='center', va='center')

# Setting the background color to gray
plt.gca().set_facecolor('lightgray')
plt.legend(loc='lower right')
plt.grid(True, which="both", ls="--")

plt.savefig('roofline_plot.png')
