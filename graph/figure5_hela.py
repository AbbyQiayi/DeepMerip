import matplotlib.pyplot as plt
import numpy as np

# Updated data
reads = [0.6, 1.2, 1.8, 3]

exomepeak2 = {
    "Pearson Correlation": [0.4378, 0.5195, 0.5689, 0.7202],
    "AUROC": [0.9239, 0.9244, 0.9258, 0.9385],
    "AUPRC": [0.5022, 0.5062, 0.5171, 0.6114],
}

resnet = {
    "Pearson Correlation": [0.7520, 0.7814, 0.7977, 0.8020],
    "AUROC": [0.8534, 0.8689, 0.8762, 0.8926],
    "AUPRC": [0.8092, 0.8453, 0.8534, 0.8625],
}

transformer = {
    "Pearson Correlation": [0.7164, 0.7489, 0.7688, 0.8298],
    "AUROC": [0.8373, 0.8584, 0.8701, 0.9017],
    "AUPRC": [0.8324, 0.8404, 0.8445, 0.8500],
}

deepmerip = {
    "Pearson Correlation": [0.7521, 0.7853, 0.8028, 0.8198],
    "AUROC": [0.8439, 0.8630, 0.8693, 0.8977],
    "AUPRC": [0.8532, 0.8794, 0.8855, 0.8890],
}

# Create the subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot Pearson Correlation
ax1.plot(reads, exomepeak2["Pearson Correlation"], 'o-', label='ExomePeak2', color='#628D56')
ax1.plot(reads, resnet["Pearson Correlation"], 'o-', label='CNN (ResNet)', color='#489FA7')
ax1.plot(reads, transformer["Pearson Correlation"], 'o-', label='Transformer', color='#E9C61D')
ax1.plot(reads, deepmerip["Pearson Correlation"], 'o-', label='DeepMerip', color='#C46210')
ax1.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax1.set_ylabel('Pearson Correlation', fontweight='bold', fontsize=14)
ax1.set_title('Pearson Correlation of Peak Calls', fontweight='bold', fontsize=16)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.arange(0, 1.1, 0.25))
ax1.legend(loc='lower right', fontsize=12)
ax1.grid(True)

# Plot AUROC
ax2.plot(reads, exomepeak2["AUROC"], 'o-', label='ExomePeak2', color='#628D56')
ax2.plot(reads, resnet["AUROC"], 'o-', label='CNN (ResNet)', color='#489FA7')
ax2.plot(reads, transformer["AUROC"], 'o-', label='Transformer', color='#E9C61D')
ax2.plot(reads, deepmerip["AUROC"], 'o-', label='DeepMerip', color='#C46210')
ax2.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax2.set_ylabel('AUROC', fontweight='bold', fontsize=14)
ax2.set_title('AUROC of Peak Calls', fontweight='bold', fontsize=16)
ax2.set_ylim(0.8, 1)
ax2.set_yticks(np.arange(0.8, 1.05, 0.05))
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True)

# Plot AUPRC
ax3.plot(reads, exomepeak2["AUPRC"], 'o-', label='ExomePeak2', color='#628D56')
ax3.plot(reads, resnet["AUPRC"], 'o-', label='CNN (ResNet)', color='#489FA7')
ax3.plot(reads, transformer["AUPRC"], 'o-', label='Transformer', color='#E9C61D')
ax3.plot(reads, deepmerip["AUPRC"], 'o-', label='DeepMerip', color='#C46210')
ax3.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax3.set_ylabel('AUPRC', fontweight='bold', fontsize=14)
ax3.set_title('AUPRC of Peak Calls', fontweight='bold', fontsize=16)
ax3.set_ylim(0.4, 1)
ax3.set_yticks(np.arange(0.4, 1.1, 0.1))
ax3.legend(loc='lower right', fontsize=12)
ax3.grid(True)

plt.tight_layout()
plt.show()
