import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Data points for each dataset
avsb = [0.5579, 0.5449, 0.6559, 0.6265, 0.6520, 0.6481, 0.6955, 0.7296, 0.7208, 0.7381, 0.7249, 0.7038, 0.7252, 0.7167, 0.7081, 0.6994, 0.7167, 0.6954, 0.7081, 0.6824, 0.6952, 0.6824, 0.6821, 0.6524, 0.6652, 0.6438, 0.6481, 0.6694, 0.6607, 0.6567, 0.6781, 0.6611, 0.6611, 0.6611, 0.6611, 0.6611]
avsc = [0.5735, 0.5515, 0.7487, 0.7702, 0.7809, 0.7704, 0.7647, 0.8468, 0.8193, 0.8139, 0.8141, 0.7980, 0.7867, 0.8084, 0.8193, 0.8249, 0.7975, 0.7920, 0.8088, 0.8141, 0.8033, 0.8142, 0.7977, 0.7975, 0.8195, 0.8085, 0.8031, 0.8033, 0.7866, 0.8033, 0.7866, 0.7867, 0.7867, 0.7867, 0.7867, 0.7867]
avsd = [0.5384, 0.5976, 0.6693, 0.8518, 0.8390, 0.8901, 0.8857, 0.8857, 0.8774, 0.8902, 0.8815, 0.8814, 0.8814, 0.8941, 0.8729, 0.8856, 0.8687, 0.8815, 0.8772, 0.8772, 0.8772, 0.8603, 0.8815, 0.8645, 0.8604, 0.8688, 0.8560, 0.8561, 0.8645, 0.8645, 0.8645, 0.8645, 0.8603, 0.8603, 0.8603, 0.8603]
bvsc = [0.5861, 0.5527, 0.6159, 0.6012, 0.6108, 0.5626, 0.6440, 0.6297, 0.6539, 0.6439, 0.6491, 0.6537, 0.6827, 0.6588, 0.6585, 0.6345, 0.6534, 0.6440, 0.6489, 0.6441, 0.6540, 0.6250, 0.6492, 0.6347, 0.6298, 0.6107, 0.6298, 0.6059, 0.6105, 0.6057, 0.6056, 0.5864, 0.5864, 0.5864, 0.5864, 0.5864]
bvsd = [0.5171, 0.5784, 0.6169, 0.6359, 0.6323, 0.6977, 0.7165, 0.7052, 0.7128, 0.7052, 0.7546, 0.7546, 0.7623, 0.7586, 0.7432, 0.7165, 0.7277, 0.7280, 0.7163, 0.7127, 0.6935, 0.6896, 0.6973, 0.6972, 0.7009, 0.7203, 0.6819, 0.7050, 0.6858, 0.6936, 0.7089, 0.7089, 0.7050, 0.7089, 0.7050, 0.7050]
cvsd = [0.5922, 0.5823, 0.6065, 0.5832, 0.6496, 0.6774, 0.6589, 0.6875, 0.6876, 0.6969, 0.6970, 0.7016, 0.7157, 0.6918, 0.7014, 0.7063, 0.6969, 0.7299, 0.7535, 0.7299, 0.7394, 0.7061, 0.7155, 0.7249, 0.7062, 0.6872, 0.6776, 0.6872, 0.7109, 0.6781, 0.6732, 0.6825, 0.6826, 0.6826, 0.6826, 0.6826]

# X values based on the number of data points
x_values = list(range(len(avsb)))
x_values = [i + 1 for i in x_values]
# Define colors from the Set1 palette
colors = plt.cm.get_cmap('tab10', 6)
# Create the plot
plt.figure(figsize=(12, 4))

# Plot each dataset
plt.plot(x_values, avsb, marker='o', linestyle='-', color=colors(0), label='A vs B')
plt.plot(x_values, avsc, marker='o', linestyle='-', color=colors(1), label='A vs C')
plt.plot(x_values, avsd, marker='o', linestyle='-', color=colors(2), label='A vs D')
plt.plot(x_values, bvsc, marker='o', linestyle='-', color=colors(3), label='B vs C')
plt.plot(x_values, bvsd, marker='o', linestyle='-', color=colors(4), label='B vs D')
plt.plot(x_values, cvsd, marker='o', linestyle='-', color=colors(5), label='C vs D')

def highlight_max(x, y, color='#C82423'):
    max_idx = y.index(max(y))
    plt.plot(x[max_idx], y[max_idx], 'o', color=color)

highlight_max(x_values, avsb)
highlight_max(x_values, avsc)
highlight_max(x_values, avsd)
highlight_max(x_values, bvsc)
highlight_max(x_values, bvsd)
highlight_max(x_values, cvsd)

# Add labels, title, and legend
plt.xlabel('Number of Features')
plt.ylabel('Cross Validation Score')
plt.title('Feature Selection of Overall Data', fontweight='bold')
plt.legend(fontsize=6)
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)

# Optimize layout and show the plot
plt.tight_layout()
plt.savefig('figure/CT/Model_Construction/Feature_Selection.pdf', format = 'pdf')
plt.show()