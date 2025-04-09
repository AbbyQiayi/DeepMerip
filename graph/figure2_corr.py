import matplotlib.pyplot as plt
import numpy as np

# Data for the three plots
reads = ['0.6M', '1.2M', '1.8M', '3M', '6M']

# Pearson Correlation
ExomePeak2_Pearson = [0.4212, 0.4937, 0.5390, 0.5990, 0.6858]
Transformer_Pearson = [0.6675, 0.7059, 0.7311, 0.7639, 0.8110]
ResNet_Pearson = [0.7186, 0.7462, 0.7645, 0.7903, 0.8224]
DeepMerip_Pearson = [0.7141, 0.7537, 0.7755, 0.8008, 0.8354]

# Mutual Information
ExomePeak2_MI = [0.1029, 0.1233, 0.2397, 0.4938, 0.5858]
Transformer_MI = [0.1927, 0.2303, 0.2483, 0.2681, 0.2944]
ResNet_MI = [0.1343, 0.2325, 0.2494, 0.6643, 0.7873]
DeepMerip_MI = [0.4176, 0.5073, 0.5658, 0.6449, 0.7670]

# KL Divergence
ExomePeak2_KL = [2.9800, 2.1749, 1.8257, 1.8233, 1.7079]
Transformer_KL = [1.6553, 1.6635, 1.6676, 1.5218, 1.0343]
ResNet_KL = [1.9729, 1.8729, 1.8409, 1.3173, 1.3024]
DeepMerip_KL = [0.8157, 0.7935, 0.7730, 0.7571, 0.7296]

# Bar width
bar_width = 0.2

# Set position of bars on X axis
r1 = np.arange(len(ExomePeak2_Pearson))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create subplots (3 in total)
fig, axs = plt.subplots(3, 1, figsize=(16, 18))

# Pearson Correlation Plot
axs[0].bar(r1, ExomePeak2_Pearson, color='#628D56', width=bar_width, label='Subsampled', edgecolor='white', linewidth=0.7)
axs[0].bar(r2, Transformer_Pearson, color='#E9C61D', width=bar_width, label='Transformer', edgecolor='white', linewidth=0.7)
axs[0].bar(r3, ResNet_Pearson, color='#489FA7', width=bar_width, label='CNN (ResNet)', edgecolor='white', linewidth=0.7)
axs[0].bar(r4, DeepMerip_Pearson, color='#3A68AE', width=bar_width, label='DeepMerip', edgecolor='white', linewidth=0.7)

axs[0].set_xlabel('Reads (Millions)', fontsize=16)
axs[0].set_ylabel('Pearson Correlation', fontsize=16)
axs[0].set_xticks([r + bar_width*1.5 for r in range(len(ExomePeak2_Pearson))])
axs[0].set_xticklabels(reads, fontsize=14)
axs[0].legend(fontsize=12)
axs[0].set_title('Pearson Correlation', fontsize=18)

# Mutual Information Plot
axs[1].bar(r1, ExomePeak2_MI, color='#628D56', width=bar_width, label='Subsampled', edgecolor='white', linewidth=0.7)
axs[1].bar(r2, Transformer_MI, color='#E9C61D', width=bar_width, label='Transformer', edgecolor='white', linewidth=0.7)
axs[1].bar(r3, ResNet_MI, color='#489FA7', width=bar_width, label='CNN (ResNet)', edgecolor='white', linewidth=0.7)
axs[1].bar(r4, DeepMerip_MI, color='#3A68AE', width=bar_width, label='DeepMerip', edgecolor='white', linewidth=0.7)

axs[1].set_xlabel('Reads (Millions)', fontsize=16)
axs[1].set_ylabel('Mutual Information', fontsize=16)
axs[1].set_xticks([r + bar_width*1.5 for r in range(len(ExomePeak2_MI))])
axs[1].set_xticklabels(reads, fontsize=14)
axs[1].legend(fontsize=12)
axs[1].set_title('Mutual Information', fontsize=18)

# KL Divergence Plot
axs[2].bar(r1, ExomePeak2_KL, color='#628D56', width=bar_width, label='Subsampled', edgecolor='white', linewidth=0.7)
axs[2].bar(r2, Transformer_KL, color='#E9C61D', width=bar_width, label='Transformer', edgecolor='white', linewidth=0.7)
axs[2].bar(r3, ResNet_KL, color='#489FA7', width=bar_width, label='CNN (ResNet)', edgecolor='white', linewidth=0.7)
axs[2].bar(r4, DeepMerip_KL, color='#3A68AE', width=bar_width, label='DeepMerip', edgecolor='white', linewidth=0.7)

axs[2].set_xlabel('Reads (Millions)', fontsize=16)
axs[2].set_ylabel('KL Divergence', fontsize=16)
axs[2].set_xticks([r + bar_width*1.5 for r in range(len(ExomePeak2_KL))])
axs[2].set_xticklabels(reads, fontsize=14)
axs[2].legend(fontsize=12)
axs[2].set_title('KL Divergence', fontsize=18)

# Set tight layout
plt.tight_layout()

# Show the plot
plt.show()
