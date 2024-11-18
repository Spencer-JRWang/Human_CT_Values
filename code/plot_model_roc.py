import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
data_path = "data/Predict"
files = ['A vs B.txt', 'A vs C.txt', 'A vs D.txt', 'B vs C.txt', 'B vs D.txt', 'C vs D.txt']
colors = plt.cm.get_cmap('tab10', len(files))
plt.figure(figsize=(5, 5))
for idx, file in enumerate(files):
    file_path = os.path.join(data_path, file)
    data = pd.read_csv(file_path, sep='\t', header=None)
    fpr, tpr, thresholds = roc_curve(data.iloc[:, 0], data.iloc[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors(idx), lw=2, label=f"{file.removesuffix('.txt')} (AUROC = {roc_auc:.3f})")
    plt.fill_between(fpr, tpr, color=colors(idx), alpha=0)
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity', fontsize=10)
plt.ylabel('Sensitivity', fontsize=10)
plt.title('ROC Plot of Train Data',fontweight='bold')
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
output_path = "figure/CT/Model_Construction/ROC_all.pdf"
plt.savefig(output_path)
plt.show()
plt.close()
print(f"ROC plot saved to {output_path}")