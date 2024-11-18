import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read data from an Excel file
data = pd.read_excel('data/original_data.xlsx')

# Define a function to calculate the peak value and 95% confidence interval
def peak_and_ci(group_data, ci=95):
    """
    Calculate the peak value and confidence interval for a given dataset.

    Parameters:
        group_data (array-like): The dataset for which the calculations are performed.
        ci (float): Confidence interval level (default is 95%).

    Returns:
        tuple: A tuple containing the peak value and a tuple of the confidence interval (lower bound, upper bound).
    """
    # Compute the kernel density estimate (KDE)
    kde = stats.gaussian_kde(group_data)
    
    # Find the peak point of the KDE
    x_range = np.linspace(min(group_data), max(group_data), 1000)  # Generate values within the data range
    kde_values = kde(x_range)  # Evaluate KDE at these points
    peak_value = x_range[np.argmax(kde_values)]  # Peak corresponds to the maximum KDE value
    
    # Compute the confidence interval
    lower_bound = np.percentile(group_data, (100 - ci) / 2)  # Lower bound of the confidence interval
    upper_bound = np.percentile(group_data, 100 - (100 - ci) / 2)  # Upper bound of the confidence interval
    
    return peak_value, (lower_bound, upper_bound)

# Create a dictionary to store the peak value and confidence interval for each group
group_peak_ci = {}

# Retrieve unique groups from the dataset
groups = data['Group'].unique()

# Compute peak and confidence intervals for each group
for group in groups:
    group_data = data[data['Group'] == group]['age']  # Filter data for the current group
    peak, (ci_lower, ci_upper) = peak_and_ci(group_data)  # Calculate peak and CI
    group_peak_ci[group] = {'peak': peak, '95% CI': (ci_lower, ci_upper)}  # Store results in the dictionary

# Print the peak and 95% confidence interval for each group
for group, values in group_peak_ci.items():
    print(f"Group: {group}, Peak: {values['peak']:.2f}, 95% CI: ({values['95% CI'][0]:.2f}, {values['95% CI'][1]:.2f})")

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set a custom color palette for the plots
custom_palette = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]

# Create a figure for the plots
plt.figure(figsize=(8, 4))

# Plot histograms and density plots for each group
for group, color in zip(data['Group'].unique(), custom_palette):
    # Plot histogram with transparency to avoid overlapping with the density plot
    sns.histplot(data[data['Group'] == group], x='age', color=color, kde=False, 
                 bins=10, stat="density", alpha=0.2, element="bars", linewidth=0)
    # Plot density (KDE) with a filled curve
    sns.kdeplot(data[data['Group'] == group], x='age', color=color, fill=True, 
                alpha=0.1, linewidth=3, label=f'Group {group}')

# Add legend and title to the plot
plt.title('Density Plot with Histogram of Age by Disease Groups', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Density')
plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)  # Add grid lines for better readability
plt.legend(title='Group', loc='upper right')
plt.tight_layout()  # Adjust layout to prevent overlapping elements

# Save the plot as a PDF file
plt.savefig("../figure/CT/Age_with_Histogram_and_Density.pdf")

# Display the plot
plt.show()
