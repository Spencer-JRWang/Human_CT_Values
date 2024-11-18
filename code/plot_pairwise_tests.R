# Load necessary libraries
library(readxl)
library(ggplot2)
library(reshape2)
library(dplyr)
library(patchwork)
library(openxlsx)  # For saving results to Excel

# Set working directory and file paths
setwd("/Users/wangjingran/Desktop/Bone_Muscle_Interaction")
file_path <- "data/original_data.xlsx"

# Load and preprocess data
original_ct <- read_excel(file_path, sheet = 'CT', na = "")
ct <- original_ct[, 5:41]

if(!"BMD" %in% colnames(original_ct)){
  stop("BMD column not found.")
}

ct_long <- melt(ct, id.vars = "BMD", variable.name = "Parameter", value.name = "Value")
ct_long <- na.omit(ct_long)

# Centrum and muscle
ct_long_cen <- ct_long[ct_long$Parameter %in% c("L1", "L2", "L3", "L4", "L5", "S1"), ]
ct_long_mus <- anti_join(ct_long, ct_long_cen, by = c("Parameter", "BMD", "Value"))

# Function to perform pairwise comparisons
perform_pairwise_tests <- function(data) {
  bmd_groups <- unique(data$BMD)
  pairwise_combinations <- combn(bmd_groups, 2, simplify = FALSE)
  
  results <- data %>%
    group_by(Parameter) %>%
    do({
      param_data <- .
      test_results <- lapply(pairwise_combinations, function(pair) {
        group1 <- param_data %>% filter(BMD == pair[1])
        group2 <- param_data %>% filter(BMD == pair[2])
        
        n1 <- nrow(group1)
        n2 <- nrow(group2)
        test_type <- ifelse(n1 > 30 & n2 > 30, "t-test", "Wilcoxon")
        
        p_value <- if (test_type == "t-test") {
          t.test(group1$Value, group2$Value)$p.value
        } else {
          wilcox.test(group1$Value, group2$Value)$p.value
        }
        
        data.frame(
          Parameter = unique(param_data$Parameter),
          Group1 = pair[1],
          Group2 = pair[2],
          Test = test_type,
          p_value = p_value
        )
      })
      do.call(rbind, test_results)
    })
  
  return(results)
}

# Perform tests
test_results_cen <- perform_pairwise_tests(ct_long_cen)
test_results_mus <- perform_pairwise_tests(ct_long_mus)

# Save results to Excel
wb <- createWorkbook()
addWorksheet(wb, "Centrum Tests")
addWorksheet(wb, "Muscle Tests")

writeData(wb, sheet = "Centrum Tests", test_results_cen)
writeData(wb, sheet = "Muscle Tests", test_results_mus)

saveWorkbook(wb, file = "data/test_results_CT.xlsx", overwrite = TRUE)

# Plotting
# Custom colors
color_temp <- c("#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2")

# Count of non-NA values per parameter
param_counts_cen <- ct_long_cen %>%
  group_by(Parameter) %>%
  summarise(count = n())

param_counts_mus <- ct_long_mus %>%
  group_by(Parameter) %>%
  summarise(count = n())

# Plot for Centrum
p_cen <- ggplot(ct_long_cen, aes(x = factor(BMD), y = Value, fill = factor(BMD))) +
  geom_violin(trim = FALSE, color = NA, alpha = 0.7) +
  geom_boxplot(width = 0.3, position = position_dodge(0.9), fill = "white", outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.25, size = 1, color = "#0f559b") +
  facet_wrap(~ Parameter, scales = "free", ncol = 6) +
  scale_fill_manual(values = color_temp) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    # strip.background = element_rect(fill = "gray90"),
    panel.grid.major = element_line(color = "gray", size = 0.25, linetype = "dashed"),
    panel.grid.minor = element_line(color = "gray", size = 0.25, linetype = "dashed"),
    panel.background = element_rect(fill = "white")
  ) +
  labs(x = "BMD Group", y = "Value")

ggsave(p_cen, file = "figure/CT/CT_paras_cen.pdf", height = 3, width = 15)

# Plot for Muscle
p_mus <- ggplot(ct_long_mus, aes(x = factor(BMD), y = Value, fill = factor(BMD))) +
  geom_violin(trim = FALSE, color = NA, alpha = 0.8) +
  geom_boxplot(width = 0.3, position = position_dodge(0.9), fill = "white", outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.25, size = 1, color = "#0f559b") +
  facet_wrap(~ Parameter, scales = "free", ncol = 6) +
  scale_fill_manual(values = color_temp) +
  scale_color_manual(values = color_temp) +  # 确保散点的颜色与小提琴图匹配
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    # strip.background = element_rect(fill = "gray90"),
    panel.grid.major = element_line(color = "gray", size = 0.25, linetype = "dashed"),
    panel.grid.minor = element_line(color = "gray", size = 0.25, linetype = "dashed"),
    panel.background = element_rect(fill = "white")
  ) +
  labs(x = "BMD Group", y = "Value")


# Bar plot for Muscle counts
p_bar_mus <- ggplot(param_counts_mus, aes(x = reorder(Parameter, -count), y = count, group = 1)) +
  geom_bar(stat = "identity", fill = "#82B0D2") +
  geom_line(color = "#7615ab", size = 0.7) + # 添加折线图，颜色选择橙红色
  geom_point(color = "#7615ab", size = 2.5) + # 在折线图上添加点
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    plot.margin = margin(0, 0, 0, 0)
  ) +
  labs(title = "Number of Records", x = NULL, y = NULL)


# Combine plots for Muscle
combined_mus <- p_mus / p_bar_mus + plot_layout(heights = c(8, 1))

ggsave(combined_mus, file = "figure/CT/CT_paras_mus.pdf", height = 16, width = 15)