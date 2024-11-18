import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(fpr, tpr, auc, filename):
    """
    Plot ROC curve with improved aesthetics and transparent area under the curve.

    Args:
        fpr (list): List of false positive rates.
        tpr (list): List of true positive rates.
        auc (float): Area under the ROC curve.
        filename (str): Name of the file to save the plot.
    """
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
    plt.fill_between(fpr, tpr, color='#1f77b4', alpha=0.2)  # Transparent fill under the curve
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity', fontsize=10)
    plt.ylabel('Sensitivity', fontsize=10)
    plt.title(filename.split('/')[-1])
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename + '.pdf')
    plt.close()


def plot_roc_for_disease_pairs(file_path, output_dir_pdf, output_dir_png):
    """
    Plot ROC curves for each pair of diseases with all features.

    Parameters:
    file_path (str): Path to the input data file.
    output_dir (str): Directory to save the output PDF files.

    Returns:
    None
    """
    print("...Generating feature ROC plots...")
    
    # Read the txt file
    data = pd.read_csv(file_path, delimiter='\t')
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Get unique disease categories
    diseases = data['Disease'].unique()
    
    # Generate pairs of diseases
    disease_pairs = [(diseases[i], diseases[j]) for i in range(len(diseases)) for j in range(i + 1, len(diseases))]
    
    # Plot for each disease pair
    for disease_pair in disease_pairs:
        # Create a figure and axis
        data_current = data[data["Disease"].isin(list(disease_pair))]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
        
        # Dictionary to store AUC values for each feature
        auc_dict = {}

        # Plot ROC curves for each feature
        for feature in data.columns[1:]:
            # Extract feature data and labels, dropping NaN values
            feature_data = data_current[[feature, 'Disease']].dropna()
            if len(feature_data) < 50:
                pass
            else:
                disease_counts = feature_data['Disease'].value_counts()

                # Find most frequent disease
                most_common_disease = disease_counts.idxmax()

                # 0-1 encoding
                feature_data['Disease'] = feature_data['Disease'].apply(lambda x: 1 if x == most_common_disease else 0)
                X = feature_data[[feature]]
                y = feature_data['Disease']

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y, X)
                roc_auc = auc(fpr, tpr)

                # If AUC <= 0.5 then reverse the y
                if roc_auc <= 0.5:
                    y = [0 if m == 1 else 1 for m in y]
                    fpr, tpr, _ = roc_curve(y, X)
                    roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                ax.plot(fpr, tpr, label=f'{feature} (AUC = {roc_auc:.3f})', lw = 1.5)

                # Store AUC value
                auc_dict[feature] = roc_auc
        # Sort legend labels by AUC values
        handles, labels = ax.get_legend_handles_labels()
        # print(handles)
        # print(labels)
        # print(auc_dict)
        labels_and_aucs = [(label, auc_dict[label.split()[0]]) for label in labels]
        labels_and_aucs_sorted = sorted(labels_and_aucs, key=lambda x: x[1], reverse=True)
        labels_sorted = [x[0] for x in labels_and_aucs_sorted]
        handles_sorted = [handles[labels.index(label)] for label in labels_sorted]
        print(f"=========={disease_pair[0]} vs {disease_pair[1]}==========")
        for n in labels_and_aucs:
            print(n[0])
        ax.legend(handles_sorted[:15], labels_sorted[:15], loc='lower right', fontsize=6)

        # Set title and axis labels
        plt.title(f'ROC Curve for {disease_pair[0]} vs {disease_pair[1]}',fontweight='bold')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
        plt.tight_layout()
        # Save as PDF file
        output_path = os.path.join(output_dir_pdf, f'{disease_pair[0]}_vs_{disease_pair[1]}_ROC.pdf')
        output_path_2 = os.path.join(output_dir_png, f'{disease_pair[0]}_vs_{disease_pair[1]}_ROC.png')
        plt.savefig(output_path, format='pdf')
        plt.savefig(output_path_2, format='png')
        plt.close(fig)
    print(f'files saved to {output_dir_pdf} and {output_dir_png}')


def plot_pca_for_disease_groups(file_path, output_dir):
    """
    Plot PCA for each disease group.

    Parameters:
    file_path (str): Path to the input data file.
    output_dir (str): Directory to save the output PDF files.

    Returns:
    None
    """
    print("...Generating PCA plots...")

    # Read the txt file
    data = pd.read_csv(file_path, delimiter='\t')

    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # Extract feature columns (assuming 'Disease' is the first column)
    features = data.columns[1:]

    # Remove columns with any NaN values
    data_no_nan = data.dropna(axis=1, how='any')
    features_no_nan = data_no_nan.columns[1:]  # Recalculate features without NaNs

    if len(features_no_nan) < 2:
        raise ValueError("Not enough features without NaN values for PCA.")

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_no_nan[features_no_nan])

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Disease'] = data_no_nan['Disease']

    # Plot PCA
    plt.figure(figsize=(12, 4))
    palette = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Disease',
        palette=palette,
        data=pca_df,
        legend='full',
        alpha=1
    )

    # Plot ellipses
    for disease, color in zip(pca_df['Disease'].unique(), palette):
        subset = pca_df[pca_df['Disease'] == disease]
        if subset.shape[0] > 2:
            cov = np.cov(subset[['PC1', 'PC2']].values, rowvar=False)
            mean = subset[['PC1', 'PC2']].mean().values
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # Convert to degrees
            ell = Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=0.5)
            plt.gca().add_patch(ell)

    # Set title and axis labels
    # plt.title('PCA of Diseases')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
    plt.title(f'PCA of Disease Groups',fontweight='bold')
    # Save as PDF file
    output_path = os.path.join(output_dir, 'PCA_Diseases_2D.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def interaction_network(file, save1, save2, save3):
    
    print("Generating Interaction Network")
    df = pd.read_excel(file)
    
    # 创建无向图
    G = nx.DiGraph()

    # 添加节点和边
    for index, row in df.iterrows():
        node1, node2, edge_type = row
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2, edge_type=edge_type)

    # 设置节点颜色
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        if node in ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']:
            node_colors.append('#FABB6E')
            node_sizes.append(1300)
        else:
            node_colors.append('#92C2DD')
            node_sizes.append(1200)

    # 获取边的颜色映射
    edge_color_map = {
        'A vs B': '#1f77b4',  # Deep Blue
        'A vs C': '#ff7f0e',  # Orange
        'A vs D': '#2ca02c',  # Green
        'B vs C': '#d62728',  # Red
        'B vs D': '#9467bd',  # Purple
        'C vs D': '#7f7f7f'    # Gray
    }
    
    edge_colors = []
    edge_labels = []
    for edge in G.edges(data=True):
        _, _, edge_data = edge
        edge_type = edge_data['edge_type']
        edge_colors.append(edge_color_map.get(edge_type, '#7f7f7f'))
        edge_labels.append(edge_type)

    # 绘制图形
    pos = nx.spring_layout(G, k=1)
    plt.figure(figsize=(12, 12))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    # 绘制边
    unique_edge_types = set(edge_labels)
    for edge_type in unique_edge_types:
        edges = [e for e, l in zip(G.edges(), edge_labels) if l == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_color_map[edge_type], width=2.5, alpha=0.7)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=9)

    # 添加图注
    plt.legend(handles=[
        plt.Line2D([0], [0], color='#1f77b4', lw=2, label='A vs B'),
        plt.Line2D([0], [0], color='#ff7f0e', lw=2, label='A vs C'),
        plt.Line2D([0], [0], color='#2ca02c', lw=2, label='A vs D'),
        plt.Line2D([0], [0], color='#d62728', lw=2, label='B vs C'),
        plt.Line2D([0], [0], color='#9467bd', lw=2, label='B vs D'),
        plt.Line2D([0], [0], color='#7f7f7f', lw=2, label='C vs D')
    ], loc='best')

    plt.title('Bone-Muscle Interaction Graph')
    plt.tight_layout()
    plt.savefig(save1, format='pdf')
    plt.show()
    plt.close()
    print(f"Graph saved as {save1}")

    # 计算节点的betweenness, closeness, degree, clustering coefficient
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree = dict(G.degree())
    clustering = nx.clustering(G)

    # 对统计量进行排序
    betweenness_sorted = dict(sorted(betweenness.items(), key=lambda item: item[1], reverse=True))
    closeness_sorted = dict(sorted(closeness.items(), key=lambda item: item[1], reverse=True))
    degree_sorted = dict(sorted(degree.items(), key=lambda item: item[1], reverse=True))
    clustering_sorted = dict(sorted(clustering.items(), key=lambda item: item[1], reverse=True))

    # 绘制统计图
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.bar(betweenness_sorted.keys(), betweenness_sorted.values(), color='skyblue')
    plt.title('Betweenness Centrality')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 2)
    plt.bar(closeness_sorted.keys(), closeness_sorted.values(), color='lightgreen')
    plt.title('Closeness Centrality')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 3)
    plt.bar(degree_sorted.keys(), degree_sorted.values(), color='salmon')
    plt.title('Degree')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 4)
    plt.bar(clustering_sorted.keys(), clustering_sorted.values(), color='#FABB6E')
    plt.title('Clustering Coefficient')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.title('Centrality distribution plot')
    plt.savefig(f'{save2}', format='pdf')
    plt.show()
    plt.close()
    print(f"Centrality saved as {save2}")
    
    # 生成邻接矩阵
    adj_matrix = nx.to_numpy_matrix(G)

    # 设置无连接的节点为白色，有连接的节点为蓝色
    adj_matrix = np.where(adj_matrix > 0, 1, 0)

    # 绘制邻接矩阵热图，添加灰色网格线
    plt.figure(figsize=(10, 10))
    sns.heatmap(adj_matrix, cmap=sns.color_palette(['white', '#1f77b4']), annot=False, cbar=False, square=True,
                xticklabels=G.nodes(), yticklabels=G.nodes(), linewidths=0.7, linecolor='gray')
    plt.title('Adjacency Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(save3, format='pdf')
    plt.show()
    plt.close()
    print(f"Adjacency Matrix Heatmap saved as {save3}")


def plot_network(file, save1, save2, save3):
    
    print("Generating Interaction Network")
    df = pd.read_excel(file)
    
    # 创建无向图
    G = nx.Graph()

    # 添加节点和边
    for index, row in df.iterrows():
        node1, node2, edge_type = row
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2, edge_type=edge_type)

    # 设置节点颜色
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        if node in ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']:
            node_colors.append('#FABB6E')
            node_sizes.append(1300)
        else:
            node_colors.append('#92C2DD')
            node_sizes.append(1200)

    # 获取边的颜色映射
    edge_color_map = {
        'A vs B': '#1f77b4',  # Deep Blue
        'A vs C': '#ff7f0e',  # Orange
        'A vs D': '#2ca02c',  # Green
        'B vs C': '#d62728',  # Red
        'B vs D': '#9467bd',  # Purple
        'C vs D': '#7f7f7f'    # Gray
    }
    
    edge_colors = []
    edge_labels = []
    for edge in G.edges(data=True):
        _, _, edge_data = edge
        edge_type = edge_data['edge_type']
        edge_colors.append(edge_color_map.get(edge_type, '#7f7f7f'))
        edge_labels.append(edge_type)

    # 绘制图形
    pos = nx.spring_layout(G, k=1)
    plt.figure(figsize=(12, 12))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    # 绘制边
    unique_edge_types = set(edge_labels)
    for edge_type in unique_edge_types:
        edges = [e for e, l in zip(G.edges(), edge_labels) if l == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_color_map[edge_type], width=2.5, alpha=0.7)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=9)

    # 添加图注
    plt.legend(handles=[
        plt.Line2D([0], [0], color='#1f77b4', lw=2, label='A vs B'),
        plt.Line2D([0], [0], color='#ff7f0e', lw=2, label='A vs C'),
        plt.Line2D([0], [0], color='#2ca02c', lw=2, label='A vs D'),
        plt.Line2D([0], [0], color='#d62728', lw=2, label='B vs C'),
        plt.Line2D([0], [0], color='#9467bd', lw=2, label='B vs D'),
        plt.Line2D([0], [0], color='#7f7f7f', lw=2, label='C vs D')
    ], loc='best')

    plt.title('Bone-Muscle Interaction Graph')
    plt.tight_layout()
    plt.savefig(save1, format='pdf')
    plt.show()
    plt.close()
    print(f"Graph saved as {save1}")

    # 计算节点的betweenness, closeness, degree, clustering coefficient
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree = dict(G.degree())
    clustering = nx.clustering(G)

    # 对统计量进行排序
    betweenness_sorted = dict(sorted(betweenness.items(), key=lambda item: item[1], reverse=True))
    closeness_sorted = dict(sorted(closeness.items(), key=lambda item: item[1], reverse=True))
    degree_sorted = dict(sorted(degree.items(), key=lambda item: item[1], reverse=True))
    clustering_sorted = dict(sorted(clustering.items(), key=lambda item: item[1], reverse=True))

    # 绘制统计图
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.bar(betweenness_sorted.keys(), betweenness_sorted.values(), color='skyblue')
    plt.title('Betweenness Centrality')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 2)
    plt.bar(closeness_sorted.keys(), closeness_sorted.values(), color='lightgreen')
    plt.title('Closeness Centrality')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 3)
    plt.bar(degree_sorted.keys(), degree_sorted.values(), color='salmon')
    plt.title('Degree')
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 4)
    plt.bar(clustering_sorted.keys(), clustering_sorted.values(), color='#FABB6E')
    plt.title('Clustering Coefficient')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.title('Centrality distribution plot')
    plt.savefig(f'{save2}', format='pdf')
    plt.show()
    plt.close()
    print(f"Centrality saved as {save2}")
    
    # 生成邻接矩阵
    adj_matrix = nx.to_numpy_matrix(G)

    # 设置无连接的节点为白色，有连接的节点为蓝色
    adj_matrix = np.where(adj_matrix > 0, 1, 0)

    # 绘制邻接矩阵热图，添加灰色网格线
    plt.figure(figsize=(10, 10))
    sns.heatmap(adj_matrix, cmap=sns.color_palette(['white', '#1f77b4']), annot=False, cbar=False, square=True,
                xticklabels=G.nodes(), yticklabels=G.nodes(), linewidths=0.7, linecolor='gray')
    plt.title('Adjacency Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(save3, format='pdf')
    plt.show()
    plt.close()
    print(f"Adjacency Matrix Heatmap saved as {save3}")


from sklearn.cross_decomposition import PLSRegression

def plot_pls_da_for_disease_groups(file_path, output_dir):
    """
    Plot PLS-DA for each disease group.

    Parameters:
    file_path (str): Path to the input data file.
    output_dir (str): Directory to save the output PDF files.

    Returns:
    None
    """
    print("...Generating PLS-DA plots...")

    # Read the txt file
    data = pd.read_csv(file_path, delimiter='\t')

    # Convert all feature columns to numeric
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Remove columns with more than 25% NA values
    threshold = len(data) * 0.1
    data_filtered = data.loc[:, data.isna().sum() <= threshold]

    # Remove rows with any NA values
    data_cleaned = data_filtered.dropna()

    # Extract feature columns and disease labels
    X = data_cleaned.iloc[:, 1:]  # Features
    y = data_cleaned.iloc[:, 0]   # Disease labels

    # Perform one-hot encoding on disease labels
    y_encoded = pd.get_dummies(y)

    # Check if enough features exist for PLS-DA
    if X.shape[1] < 2:
        raise ValueError("Not enough features without NaN values for PLS-DA.")

    # Perform PLS-DA
    pls_da = PLSRegression(n_components=2)
    pls_da.fit(X, y_encoded)
    pls_components = pls_da.transform(X)

    # Create a DataFrame with the PLS-DA results
    pls_df = pd.DataFrame(data=pls_components, columns=['PLS1', 'PLS2'])
    pls_df['Disease'] = data_cleaned.iloc[:, 0]

    # Plot PLS-DA
    plt.figure(figsize=(12, 4))
    palette = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]
    sns.scatterplot(
        x='PLS1', y='PLS2',
        hue='Disease',
        palette=palette,
        data=pls_df,
        legend='full',
        alpha=0.7
    )

    # Plot ellipses
    for disease, color in zip(pls_df['Disease'].unique(), palette):
        subset = pls_df[pls_df['Disease'] == disease]
        if subset.shape[0] > 2:
            cov = np.cov(subset[['PLS1', 'PLS2']].values, rowvar=False)
            mean = subset[['PLS1', 'PLS2']].mean().values
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # Convert to degrees
            ell = Ellipse(v[0], v[1], 180.0 + angle, color=color, alpha=0.5)
            plt.gca().add_patch(ell)

    # Set title and axis labels
    plt.xlabel('PLS Component 1')
    plt.ylabel('PLS Component 2')
    plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
    plt.title('PLS-DA of Disease Groups', fontweight='bold')

    # Save as PDF and PNG files
    output_path = os.path.join(output_dir, 'PLS_DA_Diseases_2D.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

