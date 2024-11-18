import matplotlib.pyplot as plt

plt.figure(figsize=(3, 3))
# data
categories = ['A', 'B', 'C', 'D']
values = [104, 129, 79, 132]

# color
color_map = {
    'A': '#8ECFC9',
    'B': '#FFBE7A',
    'C': '#FA7F6F',
    'D': '#82B0D2'
}
colors = [color_map[cat] for cat in categories]
plt.barh(categories, values, color=colors)
plt.xlabel('Value')
plt.ylabel('Stage')
plt.title('Group Counts', fontweight='bold')
plt.tight_layout()
plt.savefig("figure/CT/Group_count.pdf")