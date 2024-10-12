import matplotlib.pyplot as plt
import numpy as np


# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 14
# Data from the image
reads = [0.6, 1.2, 1.8, 3, 6]
AUROC_exomepeak2 = [0.9357, 0.9364, 0.9385, 0.9410, 0.9501]
AUROC_transformer = [0.8199, 0.8477, 0.8633, 0.8780, 0.9008]
AUROC_resnet = [0.8459, 0.8642, 0.8750, 0.8896, 0.8992]
AUROC_macs2 = [0.9161, 0.9262, 0.9257, 0.9345, 0.9491]

auprc_exomepeak2 = [0.5025, 0.5082, 0.5268, 0.5484, 0.6256]
auprc_transformer = [0.7594, 0.7604, 0.7624, 0.7587, 0.7793]
auprc_resnet = [0.8026, 0.7988, 0.7949, 0.7909, 0.7927]
auprc_macs2 = [0.4834, 0.5016, 0.5014, 0.5402, 0.6074]

# Adjusting the font size for the axes labels and ticks

# Plotting AUROC and AUPRC in one figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plotting AUROC
ax1.plot(reads, AUROC_exomepeak2, 'o-', label='Subsampled+ExomePeak2', color='#628D56')
ax1.plot(reads, AUROC_transformer, 'o-', label='Subsampled+Transformer', color='#E9C61D')
ax1.plot(reads, AUROC_resnet, 'o-', label='Subsampled+DeepMerip', color='#489FA7')
ax1.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold',fontsize=14)
ax1.plot(reads, AUROC_macs2, 'o-', label='Subsampled+MACS2', color='#3A68AE')
ax1.set_ylabel('AUROC of peak calls', fontweight='bold',fontsize=14)
ax1.set_title('AUROC of peak calls', fontweight='bold',fontsize=16)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.arange(0, 1.1, 0.25))
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=12)
ax1.grid(True)

# Plotting AUPRC
ax2.plot(reads, auprc_exomepeak2, 'o-', label='Subsampled+ExomePeak2', color='#628D56')
ax2.plot(reads, auprc_transformer, 'o-', label='Subsampled+Transformer', color='#E9C61D')
ax2.plot(reads, auprc_resnet, 'o-', label='Subsampled+DeepMerip', color='#489FA7')
ax2.plot(reads, auprc_macs2, 'o-', label='Subsampled+MACS2', color='#3A68AE')
ax2.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold',fontsize=14)
ax2.set_ylabel('AUPRC of peak calls', fontweight='bold',fontsize=14)
ax2.set_title('AUPRC of peak calls', fontweight='bold',fontsize=16)
ax2.set_ylim(0, 1)
ax2.set_yticks(np.arange(0, 1.1, 0.25))
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.show()

