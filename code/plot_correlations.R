library(corrplot)
# read data
data <- read.table("data/Human_CT_Values.txt",sep="\t",header=TRUE)

# process NAs
na_percentage <- sapply(data[, 2:ncol(data)], function(col) mean(is.na(col)))
print(na_percentage)
data <- data[, c(TRUE, na_percentage <= 0.26)]
labels <- data[, 1]
features <- data[, -1]

# fill NAs
data_filled <- features
for (col in 1:ncol(features)) {
  data_filled[, col] <- ave(features[, col], labels, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}
cor_colors <- colorRampPalette(c("#131963", "white", "#b1201b"))(100)
correlations <- cor(data_filled, method = "spearman")

# Draw Correlation Heatmap
pdf("figure/CT/Correlations.pdf", height = 6.7, width = 6.7)
#corrplot(correlations, method = "color", tl.col = "black", tl.srt = 45,tl.cex = 1,col = cor_colors)
corrplot(correlations, 
         method = "color", 
         #addCoef.col = "black",  
         col = cor_colors,       
         tl.col = "black",      
         order = "hclust",
         tl.cex = 0.8,      
         number.cex = 1,
         addrect = 6)
dev.off()