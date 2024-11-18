import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import seaborn as sns
data_path = "data/Predict"
files = ['A vs B.txt', 'A vs C.txt', 'A vs D.txt', 'B vs C.txt', 'B vs D.txt', 'C vs D.txt']
colors = plt.cm.get_cmap('tab10', len(files))
plt.figure(figsize=(8, 4))
for idx, file in enumerate(files):
    data = pd.read_csv(os.path.join(data_path, file), sep='\t', header=None)
    plt.hist(data.iloc[:, 1][data.iloc[:, 0] == 0], bins=30, alpha=0.3, color=colors(idx), density=False, label=f"{file.removesuffix('.txt')} - Class {file.removesuffix('.txt').split(' vs ')[0]}")
    plt.hist(data.iloc[:, 1][data.iloc[:, 0] == 1], bins=30, alpha=0.75, color=colors(idx), density=False, label=f"{file.removesuffix('.txt')} - Class {file.removesuffix('.txt').split(' vs ')[1]}")
plt.xlabel('Predicted Score')
plt.ylabel('Count')
plt.title('Histogram of Predicted Scores',fontweight='bold')
plt.legend(loc=[0.55, 0.3], fontsize=8)
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
plt.tight_layout()
plt.savefig("figure/CT/Histogram_all.pdf")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
for idx, file in enumerate(files):
    data = pd.read_csv(os.path.join(data_path, file), sep='\t', header=None)
    
    sns.kdeplot(
        data.iloc[:, 1],
        color=colors(idx),
        fill=True,
        alpha=0.1,
        linewidth=3,
        label=file.removesuffix('.txt')
    )

plt.xlabel('Predicted Score')
plt.ylabel('Density')
plt.title('Density Plot of Predicted Scores', fontweight='bold')
plt.legend(loc='upper right', fontsize=8)
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
plt.tight_layout()
plt.savefig("../figure/CT/Density_all.pdf")
plt.show()
plt.close()