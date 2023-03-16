rm(list=ls())
options(stringsAsFactors=F)
rt=read.table("ReliefF.txt",sep="\t",header=T,check.names=F)
rt <- rt[,1:3]
rt <- t(rt)
rt[1,] <- ifelse(rt[1,]==0, "normal", "tumor")
colnames(rt) <- rt[1,]
rt <- rt[-1,]

exp=rt[,1:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
rt=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
# rt=log2(rt+0.001)
#rt[rt>15]=15

library(pheatmap)

type=factor(colnames(rt),levels = c("normal","tumor"))
ann=data.frame(type)
data <- rt
colnames(data) <- rownames(ann)

	   
pdf(file="heatmap.pdf",height=3,width=6)

# c("limegreen","khaki1","red3")
pheatmap(data, annotation_col=ann,
         color = colorRampPalette(c("limegreen","khaki1","red3"))(50),scale="row",show_rownames=T,show_colnames=F,
         fontsize = 6)
         
dev.off()


