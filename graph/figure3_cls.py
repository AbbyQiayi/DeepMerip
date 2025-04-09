import matplotlib.pyplot as plt
import numpy as np

# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 14

# Data from the table
reads = [0.6, 1.2, 1.8, 3, 6]
AUROC_exomepeak2 = [0.936, 0.936, 0.939, 0.941, 0.950]
AUROC_resnet = [0.846, 0.864, 0.875, 0.890, 0.890]
AUROC_transformer = [0.820, 0.848, 0.863, 0.878, 0.901]
AUROC_macs2 = [0.483, 0.447, 0.447, 0.452, 0.459]
AUROC_deepmerip = [0.828, 0.848, 0.858, 0.884, 0.892]

auprc_exomepeak2 = [0.503, 0.508, 0.527, 0.548, 0.626]
auprc_resnet = [0.803, 0.799, 0.795, 0.791, 0.793]
auprc_transformer = [0.759, 0.760, 0.762, 0.759, 0.779]
auprc_macs2 = [0.228, 0.229, 0.231, 0.242, 0.249]
auprc_deepmerip = [0.788, 0.789, 0.805, 0.812, 0.821]

# Plotting AUROC and AUPRC in one figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting AUROC
ax1.plot(reads, AUROC_exomepeak2, 'o-', label='ExomePeak2', color='#628D56')
ax1.plot(reads, AUROC_resnet, 'o-', label='CNN (ResNet)', color='#E9C61D')
ax1.plot(reads, AUROC_transformer, 'o-', label='Transformer', color='#489FA7')
ax1.plot(reads, AUROC_macs2, 'o-', label='MACS2', color='#3A68AE')
ax1.plot(reads, AUROC_deepmerip, 'o-', label='DeepMerip', color='#7B68EE')
ax1.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax1.set_ylabel('AUROC of peak calls', fontweight='bold', fontsize=14)
ax1.set_title('AUROC of peak calls', fontweight='bold', fontsize=16)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.arange(0, 1.1, 0.25))
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=12)
ax1.grid(True)

# Plotting AUPRC
ax2.plot(reads, auprc_exomepeak2, 'o-', label='ExomePeak2', color='#628D56')
ax2.plot(reads, auprc_resnet, 'o-', label='CNN (ResNet)', color='#E9C61D')
ax2.plot(reads, auprc_transformer, 'o-', label='Transformer', color='#489FA7')
ax2.plot(reads, auprc_macs2, 'o-', label='MACS2', color='#3A68AE')
ax2.plot(reads, auprc_deepmerip, 'o-', label='DeepMerip', color='#7B68EE')
ax2.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax2.set_ylabel('AUPRC of peak calls', fontweight='bold', fontsize=14)
ax2.set_title('AUPRC of peak calls', fontweight='bold', fontsize=16)
ax2.set_ylim(0, 1)
ax2.set_yticks(np.arange(0, 1.1, 0.25))
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.show()
