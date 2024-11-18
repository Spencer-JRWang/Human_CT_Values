# Load necessary packages
library(ggplot2)

# Read the dataset from a tab-delimited file
data <- read.csv("data/Human_CT_Values.txt", sep = "\t", header = TRUE)

# Convert all columns (except the first) to numeric type
data[, 2:ncol(data)] <- lapply(data[, 2:ncol(data)], as.numeric)

# Calculate the proportion of missing values (NA) for each column
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)

# Filter out columns with more than 20% missing values
data_filtered <- data[, c(TRUE, na_percentage <= 0.03)]

# Separate labels (first column) and features (remaining columns)
labels <- data_filtered[, 1]
features <- data_filtered[, -1]

# Impute missing values using the mean of the corresponding group (defined by labels)
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}

# Perform Principal Component Analysis (PCA) with scaled data
pca_result <- prcomp(data_filled, scale = TRUE)

# Prepare data for PCA plot
pca_data <- data.frame(Sample = rownames(data_filled),  # Add sample names
                       PC1 = pca_result$x[, 1],         # First principal component
                       PC2 = pca_result$x[, 2],         # Second principal component
                       Labels = labels)                # Corresponding labels for samples

# Define custom colors for the plot
custom_colors <- c("#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2")

# Create a PCA plot and add ellipses for confidence intervals
p <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Labels)) +
  geom_point(size = 1.5) +  # Plot points for PCA data
  stat_ellipse(level = 0.95, aes(fill = Labels), alpha = 0.2, geom = "polygon", show.legend = FALSE) +  # Add 95% confidence ellipses
  theme_minimal() +  # Apply minimal theme
  scale_color_manual(values = custom_colors) +  # Use custom colors for points
  scale_fill_manual(values = custom_colors) +  # Use custom colors for ellipses
  labs(title = "PCA Plot of Disease Groups", x = "PC1", y = "PC2") +  # Add title and axis labels
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),  # Center and bold the title
        legend.position = "None")  # Remove legend

# Save the PCA plot as a PDF file
ggsave("figure/CT/Reduct_Dimension/PCA.pdf", plot = p, width = 4, height = 4)
