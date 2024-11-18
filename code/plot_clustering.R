library(ggplot2)
library(pheatmap)

# read data
data <- read.csv("data/data_for_cluster_ct.txt", sep = "\t", header = TRUE)
data[, 2:ncol(data)] <- lapply(data[, 2:ncol(data)], as.numeric)

# process NAs
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)
data_filtered <- data[, c(TRUE, na_percentage <= 0.05)]
labels <- data_filtered[, 1]
features <- data_filtered[, -1]
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}
labels <- as.factor(labels)
data_filled <- t(data_filled)

annotation_col = data.frame(Group =c(rep("A",101), rep("B",129),rep("C",79),rep("D",127)))
colnames(data_filled) <- c(paste0("A", 1:101), paste0("B",1:129), paste0("C",1:79),paste0("D",1:127))
rownames(annotation_col) <- colnames(data_filled)
colnames(annotation_col) <- "Stage"

annotation_row = data.frame(
  Type = factor(rep(c("Bone", "Muscle"), c(6, nrow(data_filled) - 6)))
)
rownames(annotation_row) = rownames(data_filled)

ann_colors = list(
  Type = c(Bone = "#4DAF4A", Muscle = "#1F78B4"),
  Stage = c(A = "#8ECFC9", B = "#FFBE7A",C =  "#FA7F6F", D = "#82B0D2")
)

# Define all methods
distance_methods <- c("euclidean", "maximum", "manhattan", "canberra", "minkowski")
clustering_methods <- c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid")
for (distance in distance_methods) {
  for (method in clustering_methods) {
    pdf_filename <- paste0("figure/CT/PCA/Cluster2_", method, "_", distance, ".pdf")
    pdf(pdf_filename, width = 10, height = 5)
    pheatmap(data_filled,scale = "row",
             color = colorRampPalette(colors = c("blue","white","red"))(100),
             angle = 45,
             clustering_distance_cols = distance,
             clustering_method = method,
             border = "white",
             show_colnames = FALSE,
             show_rownames = TRUE,
             legend = FALSE,
             cluster_cols = TRUE, treeheight_col = 20,
             cluster_rows = TRUE, treeheight_row = 20,
             annotation_colors = ann_colors,
             annotation_col = annotation_col,
             annotation_row = annotation_row,
             title = "Bone Muscle Cluster Plot"
    )
    dev.off()
  }
}