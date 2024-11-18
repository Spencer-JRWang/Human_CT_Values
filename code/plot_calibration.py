import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
data_path = "data/Predict"
files = ['A vs B.txt', 'A vs C.txt', 'A vs D.txt', 'B vs C.txt', 'B vs D.txt', 'C vs D.txt']
colors = plt.cm.get_cmap('tab10', len(files))
plt.figure(figsize=(8, 4))
for idx, file in enumerate(files):
    data = pd.read_csv(os.path.join(data_path, file), sep='\t', header=None)
    prob_true, prob_pred = calibration_curve(data.iloc[:, 0], data.iloc[:, 1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', color=colors(idx), label=f"{file.removesuffix('.txt')}", linewidth = 2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth = 2)
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration plot of all groups',fontweight='bold')
plt.legend(loc="upper left", fontsize=8)
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
plt.tight_layout()
plt.savefig("figure/CT/Calibration_all.pdf")
plt.show()
plt.close()
