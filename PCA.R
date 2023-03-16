rm(list=ls())
options(stringsAsFactors=F)
dat <- read.table("ReliefF.txt", sep="\t", header=T, check.names=F)
dat <- dat[,1:3]
dat <- cbind(scale(dat[,2:ncol(dat)]), dat[,1])
colnames(dat)[ncol(dat)] <- "class"
dat <- as.data.frame(dat)
dat$class <- ifelse(dat$class==0, "normal", "tumor")


library("FactoMineR")#画主成分分析图需要加载这两个包
library("factoextra") 
# The variable group_list (index = 54676) is removed
# before PCA analysis
pdf("pca.pdf")
dat.pca <- PCA(dat[,-ncol(dat)],graph = FALSE)#现在dat最后一
fviz_pca_ind(dat.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = dat$class, # color by groups
             addEllipses = TRUE, # Concentration ellipses
             ellipse.level=0.95,
             legend.title = "Groups",
             ggtheme = theme_minimal()
)
dev.off()

