# 加载必要的包
library(ggplot2)
library(pls)  # PLS包
library(MASS)  # 用于绘制置信椭圆

# 读取数据
data <- read.csv("data/data_ml.txt", sep = "\t", header = TRUE)

# 将列转换为数值类型
data[, 2:ncol(data)] <- lapply(data[, 2:ncol(data)], as.numeric)

# 计算每列的NA比例
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)

# 过滤NA比例大于20%的列
data_filtered <- data[, c(TRUE, na_percentage <= 0.03)]

# 提取标签和特征
labels <- data_filtered[, 1]
features <- data_filtered[, -1]

# 填充缺失值
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}
labels <- as.factor(labels)

# 将因子标签转换为数值型
labels_numeric <- as.numeric(labels)

# PLS-DA建模
pls_model <- plsr(labels_numeric ~ ., data = data_filled, validation = "LOO", scale = TRUE)

# 提取主成分得分
scores <- scores(pls_model, ncomp = 2)  # 提取前两个主成分的得分

# 提取前两个主成分得分
scores_df <- data.frame(
  PC1 = scores[, 1],
  PC2 = scores[, 2],
  Label = labels
)

# 自定义颜色
custom_colors <- c("#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2")

# 绘图
p <- ggplot(scores_df, aes(x = PC1, y = PC2, color = Label)) +
  geom_point(size = 1.5) +
  stat_ellipse(level = 0.95, aes(fill = Label), alpha = 0.2, geom = "polygon", show.legend = FALSE) +  # 添加置信椭圆
  scale_color_manual(values = custom_colors) +  # 自定义颜色
  scale_fill_manual(values = custom_colors) +
  theme_minimal() +
  labs(title = "PLS-DA Plot of Disease Groups", x = "PC1", y = "PC2") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),  # 标题居中并加粗
        legend.position = "none") 

ggsave("figure/CT/PCA/PLS-DA.pdf", plot = p, width = 4, height = 4)