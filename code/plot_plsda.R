# Load necessary packages
library(ggplot2)  # For data visualization
library(pls)      # For Partial Least Squares (PLS) modeling
library(MASS)     # For plotting confidence ellipses

# Read the dataset from a tab-delimited file
data <- read.csv("data/Human_CT_Values.txt", sep = "\t", header = TRUE)

# Convert all columns (except the first) to numeric type
data[, 2:ncol(data)] <- lapply(data[, 2:ncol(data)], as.numeric)

# Calculate the proportion of missing values (NA) for each column
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)

# Filter out columns with more than 3% missing values
data_filtered <- data[, c(TRUE, na_percentage <= 0.03)]

# Separate labels (first column) and features (remaining columns)
labels <- data_filtered[, 1]
features <- data_filtered[, -1]

# Impute missing values using the mean of the corresponding group (defined by labels)
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}

# Convert the label column to a factor type
labels <- as.factor(labels)

# Convert factor labels to numeric values for modeling
labels_numeric <- as.numeric(labels)

# Build a PLS-DA model
pls_model <- plsr(labels_numeric ~ ., data = data_filled, validation = "LOO", scale = TRUE)  # Use Leave-One-Out (LOO) cross-validation

# Extract scores for the first two principal components
scores <- scores(pls_model, ncomp = 2)

# Prepare a data frame for plotting
scores_df <- data.frame(
  PC1 = scores[, 1],  # First principal component
  PC2 = scores[, 2],  # Second principal component
  Label = labels      # Corresponding group labels
)

# Define custom colors for the plot
custom_colors <- c("#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2")

# Create the PLS-DA plot
p <- ggplot(scores_df, aes(x = PC1, y = PC2, color = Label)) +
  geom_point(size = 1.5) +  # Plot points for PLS-DA scores
  stat_ellipse(level = 0.95, aes(fill = Label), alpha = 0.2, geom = "polygon", show.legend = FALSE) +  # Add 95% confidence ellipses
  scale_color_manual(values = custom_colors) +  # Use custom colors for points
  scale_fill_manual(values = custom_colors) +  # Use custom colors for ellipses
  theme_minimal() +  # Apply minimal theme
  labs(title = "PLS-DA Plot of Disease Groups", x = "PC1", y = "PC2") +  # Add title and axis labels
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),  # Center and bold the title
        legend.position = "none")  # Remove legend

# Save the plot as a PDF file
ggsave("figure/CT/Reduct_Dimension/PLS-DA.pdf", plot = p, width = 4, height = 4)
