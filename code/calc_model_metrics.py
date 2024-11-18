import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
data_path = "data/Predict"
files = ['A vs B.txt', 'A vs C.txt', 'A vs D.txt', 'B vs C.txt', 'B vs D.txt', 'C vs D.txt']
metrics_dict = {'File': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}

plt.figure(figsize=(10, 5))
for idx, file in enumerate(files):
    data = pd.read_csv(os.path.join(data_path, file), sep='\t', header=None)
    y_true = data.iloc[:, 0]
    y_pred = data.iloc[:, 1]
    accuracy = accuracy_score(y_true, y_pred.round())
    precision = precision_score(y_true, y_pred.round(), zero_division=0)
    recall = recall_score(y_true, y_pred.round())
    f1 = f1_score(y_true, y_pred.round())

    metrics_dict['File'].append(file.removesuffix('.txt'))
    metrics_dict['Accuracy'].append(accuracy)
    metrics_dict['Precision'].append(precision)
    metrics_dict['Recall'].append(recall)
    metrics_dict['F1'].append(f1)

metrics_df = pd.DataFrame(metrics_dict)
print(metrics_df)

