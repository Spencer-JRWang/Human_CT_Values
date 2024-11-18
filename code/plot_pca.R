# 加载必要的包
library(ggplot2)

# 读取数据
data <- read.csv("data/data_ml.txt", sep = "\t", header = TRUE)

# 将列转换为数值类型
data[, 2:ncol(data)] <- lapply(data[, 2:ncol(data)], as.numeric)

# 计算每列的NA比例
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)

# 过滤NA比例大于20%的列
data_filtered <- data[, c(TRUE, na_percentage <= 0.1)]

# 提取标签和特征
labels <- data_filtered[, 1]
features <- data_filtered[, -1]

# 填充缺失值
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}

# 进行PCA分析
pca_result <- prcomp(data_filled, scale = TRUE)

# 创建PCA图数据
pca_data <- data.frame(Sample = rownames(data_filled),
                       PC1 = pca_result$x[, 1],
                       PC2 = pca_result$x[, 2],
                       Labels = labels)

# 自定义颜色
custom_colors <- c("#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2")

# 绘制PCA图并添加椭圆
p <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Labels)) +
  geom_point(size = 1.5) +  # 数据点
  stat_ellipse(level = 0.95, aes(fill = Labels), alpha = 0.2, geom = "polygon", show.legend = FALSE) +  # 添加置信椭圆
  theme_minimal() +
  scale_color_manual(values = custom_colors) +  # 自定义颜色
  scale_fill_manual(values = custom_colors)+
  labs(title = "PCA Plot of Disease Groups", x = "PC1", y = "PC2") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),  # 标题居中并加粗
        legend.position = "None") 

# 保存图形为PDF
ggsave("figure/CT/PCA/PCA.pdf", plot = p, width = 4, height = 4)