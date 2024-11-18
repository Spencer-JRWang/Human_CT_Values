










#################### Dependency Plots ####################
import shap
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
df_all = pd.read_csv('data/Human_CT_Values.txt', sep='\t')
df_all['Gender'] = df_all['Gender'].astype('category')
Cat_A = "A"
Cat_B = "C"
f = ['L1', 'L4', 'L1-L2_3', 'L2-L3_4', 'L2-L3_5', 'L3-L4_1', 'L4-L5_1', 'L5-S1_4']
df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
X = df.drop("Disease",axis = 1)
X = X[f]
y = df['Disease']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
shuffle_index = np.random.permutation(X.index)
X = X.iloc[shuffle_index]
y = y.iloc[shuffle_index]
y = y.map({Cat_A: 0, Cat_B: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "boost_from_average": True,
    "early_stopping_rounds": 50,
    "verbose_eval": 1000
}
model = lgb.train(
    params,
    d_train,
    1000,
    valid_sets=[d_test],
)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
start_color = (0.2, 0.8, 0.2)
middle_color = (1, 1, 0)
end_color = (1, 0, 0)
cmap = plt.get_cmap('RdYlBu_r',12)

fig, ax = plt.subplots(figsize=(5, 4))
plt.title("L4 & L1 Dependence Plot", fontweight='bold', fontsize=10)
ax.grid(linestyle="--", color="gray", linewidth=0.3, zorder=0, alpha=0.3)
shap.plots.scatter(shap_values[:, "L4"], color=shap_values[:, "L1"], cmap=cmap, ax=ax, show=False, dot_size = 25)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
plt.savefig(f"../figure/CT/Exp/Dependence/L4 & L1 Dependence Plot.pdf", bbox_inches='tight')
plt.close()
###################################################################################################
Cat_A = "B"
Cat_B = "D"
f = ['L1', 'L5', 'S1', 'L1-L2_5', 'L1-L2_6', 'L2-L3_1', 'L2-L3_2', 'L2-L3_3', 'L2-L3_6', 'L3-L4_1', 'L3-L4_6', 'L4-L5_1', 'L4-L5_3']
df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
X = df.drop("Disease",axis = 1)
X = X[f]
y = df['Disease']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
shuffle_index = np.random.permutation(X.index)
X = X.iloc[shuffle_index]
y = y.iloc[shuffle_index]
y = y.map({Cat_A: 0, Cat_B: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "boost_from_average": True,
    "early_stopping_rounds": 50,
    "verbose_eval": 1000
}
model = lgb.train(
    params,
    d_train,
    1000,
    valid_sets=[d_test],
)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
start_color = (0.2, 0.8, 0.2)
middle_color = (1, 1, 0)
end_color = (1, 0, 0)
cmap = plt.get_cmap('RdYlBu_r',12)

fig, ax = plt.subplots(figsize=(5, 4))
plt.title("S1 & L3 Dependence Plot", fontweight='bold', fontsize=10)
ax.grid(linestyle="--", color="gray", linewidth=0.3, zorder=0, alpha=0.3)
shap.plots.scatter(shap_values[:, "S1"], color=shap_values[:, "L3"], cmap=cmap, ax=ax, show=False, dot_size = 25)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
plt.savefig(f"../figure/CT/Exp/Dependence/L5 & L3 Dependence Plot.pdf", bbox_inches='tight')
plt.close()
###################################################################################################
Cat_A = "A"
Cat_B = "D"
f = ['L1', 'L2', 'L4', 'L5', 'L1-L2_3', 'L1-L2_4', 'L1-L2_6', 'L2-L3_3', 'L2-L3_5', 'L3-L4_5', 'L4-L5_1', 'L4-L5_3', 'L4-L5_4', 'L4-L5_5']
df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
X = df.drop("Disease",axis = 1)
X = X[f]
y = df['Disease']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
shuffle_index = np.random.permutation(X.index)
X = X.iloc[shuffle_index]
y = y.iloc[shuffle_index]
y = y.map({Cat_A: 0, Cat_B: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "boost_from_average": True,
    "early_stopping_rounds": 50,
    "verbose_eval": 1000
}
model = lgb.train(
    params,
    d_train,
    1000,
    valid_sets=[d_test],
)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
start_color = (0.2, 0.8, 0.2)
middle_color = (1, 1, 0)
end_color = (1, 0, 0)
cmap = plt.get_cmap('RdYlBu_r',12)

fig, ax = plt.subplots(figsize=(5, 4))
plt.title("L5 & L4 Dependence Plot", fontweight='bold', fontsize=10)
ax.grid(linestyle="--", color="gray", linewidth=0.3, zorder=0, alpha=0.3)
shap.plots.scatter(shap_values[:, "L5"], color=shap_values[:, "L4"], cmap=cmap, ax=ax, show=False, dot_size = 25)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
plt.savefig(f"../figure/CT/Exp/Dependence/L5 & L4 Dependence Plot.pdf", bbox_inches='tight')
plt.close()
###################################################################################################
Cat_A = "B"
Cat_B = "D"
f = ['L1', 'L5', 'S1', 'L1-L2_5', 'L1-L2_6', 'L2-L3_1', 'L2-L3_2', 'L2-L3_3', 'L2-L3_6', 'L3-L4_1', 'L3-L4_6', 'L4-L5_1', 'L4-L5_3']
df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
X = df.drop("Disease",axis = 1)
X = X[f]
y = df['Disease']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
shuffle_index = np.random.permutation(X.index)
X = X.iloc[shuffle_index]
y = y.iloc[shuffle_index]
y = y.map({Cat_A: 0, Cat_B: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "boost_from_average": True,
    "early_stopping_rounds": 50,
    "verbose_eval": 1000
}
model = lgb.train(
    params,
    d_train,
    1000,
    valid_sets=[d_test],
)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
start_color = (0.2, 0.8, 0.2)
middle_color = (1, 1, 0)
end_color = (1, 0, 0)
cmap = plt.get_cmap('RdYlBu_r',12)

fig, ax = plt.subplots(figsize=(5, 4))
plt.title("S1 & L1-L2_5 Dependence Plot", fontweight='bold', fontsize=10)
ax.grid(linestyle="--", color="gray", linewidth=0.3, zorder=0, alpha=0.3)
shap.plots.scatter(shap_values[:, "S1"], color=shap_values[:, "L1-L2_5"], cmap=cmap, ax=ax, show=False, dot_size = 25)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
plt.savefig(f"../figure/CT/Exp/Dependence/S1 & L1-L2_5 Dependence Plot.pdf", bbox_inches='tight')
plt.close()
###################################################################################################
Cat_A = "B"
Cat_B = "D"
f = ['L1', 'L5', 'S1', 'L1-L2_5', 'L1-L2_6', 'L2-L3_1', 'L2-L3_2', 'L2-L3_3', 'L2-L3_6', 'L3-L4_1', 'L3-L4_6', 'L4-L5_1', 'L4-L5_3']
df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
X = df.drop("Disease",axis = 1)
X = X[f]
y = df['Disease']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
shuffle_index = np.random.permutation(X.index)
X = X.iloc[shuffle_index]
y = y.iloc[shuffle_index]
y = y.map({Cat_A: 0, Cat_B: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "boost_from_average": True,
    "early_stopping_rounds": 50,
    "verbose_eval": 1000
}
model = lgb.train(
    params,
    d_train,
    1000,
    valid_sets=[d_test],
)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
start_color = (0.2, 0.8, 0.2)
middle_color = (1, 1, 0)
end_color = (1, 0, 0)
cmap = plt.get_cmap('RdYlBu_r',12)
###################################################################################################
fig, ax = plt.subplots(figsize=(5, 4))
plt.title("L5 & L3-L4_1 Dependence Plot", fontweight='bold', fontsize=10)
ax.grid(linestyle="--", color="gray", linewidth=0.3, zorder=0, alpha=0.3)
shap.plots.scatter(shap_values[:, "L5"], color=shap_values[:, "L3-L4_1"], cmap=cmap, ax=ax, show=False, dot_size = 25)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
plt.savefig(f"../figure/CT/Exp/Dependence/L5 & L3-L4_1 Dependence Plot.pdf", bbox_inches='tight')
plt.close()

import shap
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
Cat_A = "C"
Cat_B = "D"
f = ['L1', 'L2', 'L5', 'S1', 'L1-L2_2', 'L1-L2_3', 'L1-L2_4', 'L1-L2_6', 'L2-L3_2', 'L2-L3_3', 'L2-L3_4', 'L3-L4_1', 'L3-L4_2', 'L3-L4_3', 'L3-L4_6', 'L4-L5_2', 'L4-L5_5', 'L5-S1_1', 'L5-S1_4']
df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
X = df.drop("Disease",axis = 1)
X = X[f]
y = df['Disease']
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
shuffle_index = np.random.permutation(X.index)
X = X.iloc[shuffle_index]
y = y.iloc[shuffle_index]
y = y.map({Cat_A: 0, Cat_B: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "boost_from_average": True,
    "early_stopping_rounds": 50,
    "verbose_eval": 1000
}
model = lgb.train(
    params,
    d_train,
    1000,
    valid_sets=[d_test],
)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
start_color = (0.2, 0.8, 0.2)
middle_color = (1, 1, 0)
end_color = (1, 0, 0)
cmap = plt.get_cmap('RdYlBu_r',12)

fig, ax = plt.subplots(figsize=(5, 4))
plt.title("L1-L2_2 & L1-L2_3 Dependence Plot", fontweight='bold', fontsize=10)
ax.grid(linestyle="--", color="gray", linewidth=0.3, zorder=0, alpha=0.3)
shap.plots.scatter(shap_values[:, "L1-L2_2"], color=shap_values[:, "L1-L2_3"], cmap=cmap, ax=ax, show=False, dot_size = 25)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
plt.savefig(f"../figure/CT/Exp/Dependence/L1-L2_2 & L1-L2_3 Dependence Plot.pdf", bbox_inches='tight')
plt.close()