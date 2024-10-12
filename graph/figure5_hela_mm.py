import matplotlib.pyplot as plt
import numpy as np


# Data for plotting
reads = [0.6, 1.2, 1.8, 3]

resnet = {
    "Pearson Correlation": [0.75199203, 0.781384348, 0.797700085, 0.801955461],
    "AUROC": [0.85335508, 0.868949729, 0.876203358, 0.892551176],
    "AUPRC": [0.809219214, 0.845340905, 0.853445243, 0.86246756],
}

transformer = {
    "Pearson Correlation": [0.716448985, 0.748911614, 0.768787969, 0.829791882],
    "AUROC": [0.837300751, 0.858376236, 0.870073629, 0.901732915],
    "AUPRC": [0.832412561, 0.840413345, 0.84450345, 0.850007597],
}

exomepeak2 = {
    "Pearson Correlation": [0.437806357, 0.519529523, 0.568949881, 0.720203328],
    "AUROC": [0.923852776, 0.924377169, 0.925787876, 0.93850094],
    "AUPRC": [0.502201339, 0.506240348, 0.517056582, 0.611391205],
}

# Create the subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

# Plot Pearson Correlation
ax1.plot(reads, exomepeak2["Pearson Correlation"], 'o-', label='Subsampled+ExomePeak2', color='#628D56')
ax1.plot(reads, resnet["Pearson Correlation"], 'o-', label='Subsampled+DeepMerip', color='#489FA7')
ax1.plot(reads, transformer["Pearson Correlation"], 'o-', label='Subsampled+Transformer', color='#E9C61D')
ax1.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax1.set_ylabel('Pearson Correlation', fontweight='bold', fontsize=14)
ax1.set_title('Pearson Correlation of peak calls', fontweight='bold', fontsize=16)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.arange(0, 1.1, 0.25))
ax1.legend(loc='lower left', fontsize=14)
ax1.grid(True)

ax2.plot(reads, exomepeak2["AUROC"], 'o-', label='Subsampled+ExomePeak2', color='#628D56')
ax2.plot(reads, resnet["AUROC"], 'o-', label='Subsampled+DeepMerip', color='#489FA7')
ax2.plot(reads, transformer["AUROC"], 'o-', label='Subsampled+Transformer', color='#E9C61D')
ax2.set_ylabel('AUROC of peak calls', fontweight='bold', fontsize=14)
ax2.set_title('AUROC of peak calls', fontweight='bold', fontsize=16)
ax2.set_ylim(0, 1)
ax2.set_yticks(np.arange(0, 1.1, 0.25))
ax2.legend(loc='lower left', fontsize=14)
ax2.grid(True)

# Plot AUPRC
ax3.plot(reads, exomepeak2["AUPRC"], 'o-', label='Subsampled+ExomePeak2', color='#628D56')
ax3.plot(reads, resnet["AUPRC"], 'o-', label='Subsampled+DeepMerip', color='#489FA7')
ax3.plot(reads, transformer["AUPRC"], 'o-', label='Subsampled+Transformer', color='#E9C61D')
ax3.set_xlabel('Sequencing Depth (Millions of Reads)', fontweight='bold', fontsize=14)
ax3.set_ylabel('AUPRC of peak calls', fontweight='bold', fontsize=14)
ax3.set_title('AUPRC of peak calls', fontweight='bold', fontsize=16)
ax3.set_ylim(0, 1)
ax3.set_yticks(np.arange(0, 1.1, 0.25))
ax3.legend(loc='lower left', fontsize=14)
ax3.grid(True)

plt.tight_layout()
plt.show()
