# Load necessary libraries
library(ggplot2)
library(dplyr)
library(broom)

# Read the data file
data <- read.csv("data/data_for_ml_ct.txt", sep = "\t", header = TRUE)

# Convert all columns (except the first) to numeric type
data[, 2:ncol(data)] <- lapply(data[, 2:ncol(data)], as.numeric)

# Calculate the percentage of missing values (NA) in each column
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)

# Filter columns with less than or equal to 2% missing values
data_filtered <- data[, c(TRUE, na_percentage <= 0.02)]

# Extract labels (first column) and features (remaining columns)
labels <- data_filtered[, 1]
features <- data_filtered[, -1]

# Fill missing values with the mean of each group based on the label
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, 
                            FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}

# Load additional data from an Excel file
library(readxl)
df <- read_excel("data/original_data.xlsx", sheet = 1)

# Compute average bone and muscle values across specified columns
data_filled$avgBone <- rowMeans(data_filled[, 1:6])
data_filled$avgMuscle <- rowMeans(data_filled[, 7:16])

# Add additional demographic and group information
data_filled$Age <- df$age
data_filled$Gender <- df$gender
data_filled$Group <- df$Group

# Replace the original dataset with the filled dataset
data <- data_filled

# Set custom colors for plots
custom_colors <- c('female' = '#d03045', 'male' = '#1f77b4', "A" = "#8ECFC9", 
                   "B" = "#FFBE7A", "C" = "#FA7F6F", "D" = "#82B0D2")

# Extract feature names for plotting (assuming first 18 columns are features)
features <- colnames(data)[1:18]

# Loop through each feature to generate plots
for (feature in features) {
  # Filter rows where the feature value is less than or equal to 400
  filter_data <- data %>% filter(!!sym(feature) <= 400)
  
  # Build a regression model of the feature against Age
  model <- lm(reformulate("Age", feature), data = data)
  r_squared <- summary(model)$r.squared  # Extract R-squared value
  
  # Create a scatter plot with confidence intervals and regression lines
  p <- ggplot(filter_data, aes(x = Age, y = !!sym(feature))) +
    geom_point(aes(shape = Gender, color = Group), size = 2.3) +  # Scatter plot by gender and group
    geom_smooth(method = "lm", se = TRUE, aes(color = Gender), alpha = 0.2) +  # Regression line by gender
    geom_smooth(method = "lm", se = TRUE, color = "black", alpha = 0.5) +  # Overall regression line
    scale_color_manual(values = custom_colors) +
    scale_shape_manual(values = c("female" = 16, "male" = 15)) +  # Set point shapes
    theme_minimal() +
    labs(title = paste("Scatter Plot of Age vs", feature),
         x = "Age", y = feature) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),  # Bold and center-align title
      legend.title = element_blank(),  # Remove legend title
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      legend.position = "top"
    ) +
    annotate("text", x = Inf, y = -Inf, label = paste("R² =", round(r_squared, 2)),
             hjust = 1.1, vjust = -0.5, size = 4, color = "black")  # Annotate R² value in bottom-right
  
  # Save the plot as a PDF
  pdf_filename <- paste0('figure/CT/Combined/', feature, "_scatter_plot.pdf")
  ggsave(pdf_filename, plot = p, width = 4, height = 4.5)
}

# Load necessary libraries for violin and box plots
library(ggplot2)
library(ggpubr)

# Prepare data for gender comparison plots
data <- data.frame(
  Gender = data_filled$Gender,
  Bone = data_filled$avgBone,
  Muscle = data_filled$avgMuscle
)

# Set custom colors for gender
custom_colors <-  c('female' = '#d03045', 'male' = '#1f77b4')

# Create violin plot, box plot, and scatter plot for Bone values
p1 <- ggplot(data, aes(x = Gender, y = Bone, fill = Gender, color = Gender)) +
  geom_violin(trim = FALSE, alpha = 0.5, color = NA, width = 0.8) +
  geom_boxplot(width = 0.2, position = position_dodge(0.75), fill = "white", 
               color = "black", outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1, color = "#0f559b") +
  scale_fill_manual(values = custom_colors) +
  labs(title = "Bone Comparison", y = "Bone Value", fill = "Bone") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 10),  # Bold and center-align title
    legend.title = element_blank(),  # Remove legend title
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    legend.position = "none"
  )

# Create violin plot, box plot, and scatter plot for Muscle values
p2 <- ggplot(data, aes(x = Gender, y = Muscle, fill = Gender, color = Gender)) +
  geom_violin(trim = FALSE, alpha = 0.5, color = NA, width = 0.8) +
  geom_boxplot(width = 0.2, position = position_dodge(0.75), fill = "white", 
               color = "black", outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1, color = "#0f559b") +
  scale_fill_manual(values = custom_colors) +
  labs(title = "Muscle Comparison", y = "Muscle Value", fill = "Muscle") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 10),  # Bold and center-align title
    legend.title = element_blank(),  # Remove legend title
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    legend.position = "none"
  )

# Arrange the two plots side by side
final_plot <- ggarrange(p1, p2, ncol = 2, nrow = 1)

# Save the combined plot as a PDF
ggsave(final_plot, file = "figure/CT/gender_compare.pdf", height = 3, width = 6)
