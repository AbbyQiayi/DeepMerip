import matplotlib.pyplot as plt
import numpy as np

# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'
plt.rcParams['font.size'] = 14

# GLORI benchmark data
reads = [0.6, 1.2, 1.8, 3, 6]

# AUROC values
AUROC_exomepeak2 = [0.4392, 0.4421, 0.4435, 0.4793, 0.4825]
AUROC_resnet = [0.4406, 0.4415, 0.4466, 0.4584, 0.4662]
AUROC_transformer = [0.4032, 0.4223, 0.4634, 0.4705, 0.4717]
AUROC_macs2 = [0.3315, 0.3446, 0.3474, 0.3513, 0.3792]
AUROC_deepmerip = [0.4286, 0.4385, 0.4494, 0.4817, 0.4885]

# AUPRC values
auprc_exomepeak2 = [0.1773, 0.1815, 0.1973, 0.1982, 0.2015]
auprc_resnet = [0.2872, 0.2894, 0.3212, 0.3265, 0.3536]
auprc_transformer = [0.2335, 0.2454, 0.2462, 0.2745, 0.2814]
auprc_macs2 = [0.1503, 0.1545, 0.1576, 0.1612, 0.1685]
auprc_deepmerip = [0.2884, 0.2984, 0.3346, 0.3517, 0.3548]

# Plotting AUROC and AUPRC in one figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting AUROC
ax1.plot(reads, AUROC_exomepeak2, 'o-', label='ExomePeak2', color='#628D56')
ax1.plot(reads, AUROC_resnet, 'o-', label='CNN (ResNet)', color='#E9C61D')
ax1.plot(reads, AUROC_transformer, 'o-', label='Transformer', color='#489FA7')
ax1.plot(reads, AUROC_macs2, 'o-', label='MACS2', color='#3A68AE')
ax1.plot(reads, AUROC_deepmerip, 'o-', label='DeepMerip', color='#7B68EE')
ax1.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold')
ax1.set_ylabel('AUROC of peak calls', fontweight='bold')
ax1.set_title('AUROC of peak calls (GLORI Benchmark)', fontweight='bold', fontsize=16)
ax1.set_ylim(0.3, 0.6)
ax1.set_yticks(np.arange(0.3, 0.61, 0.05))
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=12)
ax1.grid(True)

# Plotting AUPRC
ax2.plot(reads, auprc_exomepeak2, 'o-', label='ExomePeak2', color='#628D56')
ax2.plot(reads, auprc_resnet, 'o-', label='CNN (ResNet)', color='#E9C61D')
ax2.plot(reads, auprc_transformer, 'o-', label='Transformer', color='#489FA7')
ax2.plot(reads, auprc_macs2, 'o-', label='MACS2', color='#3A68AE')
ax2.plot(reads, auprc_deepmerip, 'o-', label='DeepMerip', color='#7B68EE')
ax2.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold')
ax2.set_ylabel('AUPRC of peak calls', fontweight='bold')
ax2.set_title('AUPRC of peak calls (GLORI Benchmark)', fontweight='bold', fontsize=16)
ax2.set_ylim(0.1, 0.4)
ax2.set_yticks(np.arange(0.1, 0.41, 0.05))
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.show()
