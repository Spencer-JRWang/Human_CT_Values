import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
import numpy as np
data_path = "data/Predict"
files = ['A vs B.txt', 'A vs C.txt', 'A vs D.txt', 'B vs C.txt', 'B vs D.txt', 'C vs D.txt']
colors = plt.cm.get_cmap('tab10', len(files))
plt.figure(figsize=(5, 5))
for idx, file in enumerate(files):
    data = pd.read_csv(os.path.join(data_path, file), sep='\t', header=None)
    precision, recall, _ = precision_recall_curve(data.iloc[:, 0], data.iloc[:, 1])
    auprc = auc(recall, precision)
    print(auprc)
    plt.plot(recall, precision, color=colors(idx), lw=2, label=f"{file.removesuffix('.txt')} (AUPRC = {auprc:.3f})")
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Plot of Train Data',fontweight='bold')
plt.legend(loc="lower left", fontsize=8)
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
plt.tight_layout()
plt.savefig("figure/CT/Model_Construction/PR_all.pdf")
plt.show()
plt.close()
