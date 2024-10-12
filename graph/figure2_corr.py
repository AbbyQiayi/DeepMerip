import matplotlib.pyplot as plt
import numpy as np

# Set the font globally to Roboto and set general font size to 14
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 14

# Redefined data with proper labeling for millions (M)
reads = ['0.6M', '1.2M', '1.8M', '3M', '6M']
ExomePeak2 = [0.4212, 0.4937, 0.5390, 0.5990, 0.6858]
Transformer = [0.6675, 0.7059, 0.7311, 0.7639, 0.8110]
ResNet = [0.7186, 0.7462, 0.7645, 0.7903, 0.8224]

# Define the bar width
bar_width = 0.25

# Set position of bar on X axis
r1 = np.arange(len(ExomePeak2))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Start of the enhancements for the graph
fig, ax = plt.subplots(figsize=(16, 9))

# Create the bars with improved aesthetics
ax.bar(r1, ExomePeak2, color='#628D56', width=bar_width, label='Subsampled', edgecolor='white', linewidth=0.7)
ax.bar(r2, Transformer, color='#E9C61D', width=bar_width, label='Transformer', edgecolor='white', linewidth=0.7)
ax.bar(r3, ResNet, color='#489FA7', width=bar_width, label='DeepMerip', edgecolor='white', linewidth=0.7)

# Add labels, title and axes ticks
ax.set_xlabel('Reads (Millions)', fontweight='bold', fontsize=20)  # Adjusted font size
ax.set_ylabel('Pearson Correlation with clean (60M read) data', fontweight='bold', fontsize=20)  # Adjusted font size
ax.set_xticks([r + bar_width for r in range(len(ExomePeak2))])
ax.set_xticklabels(reads, fontsize=16)  # Adjusted font size

# Set y-axis to start at 0.4
ax.set_ylim(bottom=0.4)
ax.set_yticks(np.arange(0.4, ax.get_yticks()[-1], step=0.1))
ax.set_yticklabels(np.around(ax.get_yticks(), decimals=2), fontsize=16)  # Adjusted font size

# Adding a grid for better readability
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='-', alpha=0.7)

# Increase the font size of the legend
ax.legend(fontsize=18)  # Adjusted font size

# Set a title with a larger font size
ax.set_title('Pearson Correlation', fontweight='bold', fontsize=24)  # Adjusted font size

# Removing the spines to eliminate the border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)

# Set a tight layout to ensure everything fits without overlapping
plt.tight_layout()

# Show graphic
plt.show()
