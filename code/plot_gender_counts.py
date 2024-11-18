import matplotlib.pyplot as plt

plt.figure(figsize=(3, 3))
# data
categories = ['Male', 'Female']
values = [129, 325]

# color
color_map = {
    'Male': '#1f77b4',
    'Female':'#d03045',
}
colors = [color_map[cat] for cat in categories]
plt.barh(categories, values, color=colors)
plt.xlabel('Value')
plt.ylabel('Stage')
plt.title('Gender Counts', fontweight='bold')
plt.tight_layout()
plt.savefig("figure/CT/Gender_count.pdf")