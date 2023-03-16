library(ggpubr)
dataset=read.table("ReliefF.txt",sep="\t",header=T,check.names=F)
dataset$class <- ifelse(dataset$class==0,"normal","tumor")


pdf("miR-5100.pdf")
ggplot(dataset, aes(x=class, y=`genes`)) + geom_violin(aes(fill=class), trim=FALSE) + geom_boxplot(width=0.2,position=position_dodge(0.9)) + scale_fill_manual(values=c("#009ad6","#f15b6c")) + stat_summary(fun.data="mean_sdl", fun.args = list(mult=1), geom="pointrange", color = "red") + stat_compare_means(method = "t.test",label="p.signif") + theme_bw()
dev.off()
