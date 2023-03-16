rm(list = ls())
library(AnnoProbe)
library(ggpubr)
suppressPackageStartupMessages(library(GEOquery))
library(impute)
library(limma)


logFoldChange=1.5
adjustP=0.05

eSet=getGEO('GSE137140', destdir=".", AnnotGPL = F, getGPL = F)[[1]]

probes_expr <- exprs(eSet)

phenoDat <- pData(eSet)
phenoDat <- cbind(id=rownames(phenoDat), phenoDat)
phenoDat <- phenoDat[,c(1,ncol(phenoDat)-2)]
colnames(phenoDat)[2] <- "group"
phenoDat <- phenoDat[phenoDat$group!="Lung cancer, post-operation",]
phenoDat$group <- ifelse(phenoDat$group=="Non-cancer control","normal","cancer")
phenoDat <- phenoDat[order(phenoDat$group,decreasing=T),]
table(phenoDat$group)    #normal:2178,tumor:1566

probes_expr <- probes_expr[,phenoDat$id]
probes_expr <- cbind(id=rownames(probes_expr), probes_expr)

GPL=eSet@annotation
probes_anno <- getGEO(GPL, destdir=".")
probes_anno <- probes_anno@dataTable@table
colnames(probes_anno)[1] <- "id"
expr <- as.matrix(merge(probes_anno, probes_expr, how="inner", by="id"))
rownames(expr) <- expr[,3]
expr <- expr[,-c(1:3)]
expr <- as.data.frame(avereps(expr))
rownames(expr) <- gsub(",",";",rownames(expr))

dimnames=list(rownames(expr),colnames(expr))
expr=matrix(as.numeric(as.matrix(expr)),nrow=nrow(expr),dimnames=dimnames)

mat=impute.knn(expr)
rt=mat$data

rt=avereps(rt)     #基因对应多个探针取均值

mat=impute.knn(expr)
rt=mat$data

rt=avereps(rt)     #基因对应多个探针取均值
#normalize
pdf(file="rawBox.pdf")
boxplot(rt,col = "blue",xaxt = "n",outline = F)
dev.off()
rt=normalizeBetweenArrays(as.matrix(rt))
pdf(file="normalBox.pdf")
boxplot(rt,col ="red",xaxt = "n",outline = F)
dev.off()



#differential
#class <- c("con","con","treat","con","treat","treat")
class <- c(rep("con",2178),rep("treat",1566))    #需要修改
# class <- c(rep(c("con","treat"),12),rep("con",3))
design <- model.matrix(~0+factor(class))
colnames(design) <- c("con","treat")
fit <- lmFit(rt,design)
cont.matrix<-makeContrasts(treat-con,levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2)

allDiff=topTable(fit2,adjust='fdr',number=200000)
write.table(allDiff,file="limmaTab.xls",sep="\t",quote=F)
allLimma=allDiff
allLimma=allLimma[order(allLimma$logFC),]
allLimma=rbind(Gene=colnames(allLimma),allLimma)
write.table(allLimma,file="limmaTab.txt",sep="\t",quote=F,col.names=F)

#write table
diffSig <- allDiff[with(allDiff, (abs(logFC)>logFoldChange & adj.P.Val < adjustP )), ]
write.table(diffSig,file="diff.xls",sep="\t",quote=F)
diffUp <- allDiff[with(allDiff, (logFC>logFoldChange & adj.P.Val < adjustP )), ]
write.table(diffUp,file="up.xls",sep="\t",quote=F)
diffDown <- allDiff[with(allDiff, (logFC<(-logFoldChange) & adj.P.Val < adjustP )), ]
write.table(diffDown,file="down.xls",sep="\t",quote=F)

#write expression level of diff gene
hmExp=rt[rownames(diffSig),]
input <- as.data.frame(c(rep(0,2178),rep(1,1566)))
input <- cbind(input,t(hmExp))
colnames(input)[1] <- "class"
write.table(input,"input.csv",sep=",",quote=F,row.names=F)
diffExp=rbind(id=colnames(hmExp),hmExp)
write.table(diffExp,file="diffExp.txt",sep="\t",quote=F,col.names=F)

#volcano
pdf(file="vol.pdf")
xMax=400
# xMax=max(-log10(allDiff$P.Value))
yMax=max(abs(allDiff$logFC))
plot(-log10(allDiff$adj.P.Val), allDiff$logFC, xlab="-log10(adj.P.Val)",ylab="logFC", main="GSE137140", xlim=c(0,xMax),ylim=c(-yMax,yMax),yaxs="i",pch=20, cex=0.4, cex.main=1.8, cex.lab=1.5)
diffSub=subset(allDiff, adj.P.Val<adjustP & logFC>logFoldChange)
points(-log10(diffSub$adj.P.Val), diffSub$logFC, pch=20, col="red",cex=0.4)
diffSub=subset(allDiff, adj.P.Val<adjustP & logFC<(-logFoldChange))
points(-log10(diffSub$adj.P.Val), diffSub$logFC, pch=20, col="green",cex=0.4)
abline(h=0,lty=2,lwd=3)
dev.off()
