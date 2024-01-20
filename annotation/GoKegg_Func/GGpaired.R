suppressPackageStartupMessages(library(ggpubr))

args <-commandArgs(T)
InFile   <- args[1]
OUFile   <- args[2]


Gpair <-function(InFile, out){
  Order <- c('HEK293T', 'U2OS.CCS', 'U2OS', 'COLON', 'PC3', 'PDAC')
  InFile$Cellline <- factor(InFile$Cellline, levels=Order)
  l <- length(unique(InFile$Cellline))
  p <- ggpaired(InFile, cond1='oncogene', cond2='genome', id="DNA", 
                color = "Cellline", facet.by = "Cellline",shape = "Cellline",
                palette = "lancet", line.color = "gray", 
                line.size = 0.3,  ncol = 6,
                font.label = list(size = 11, face = "bold", color ="black"),
                width = 0.5, panel.labs=NULL,
                #xlab = opt$xlab, ylab = opt$xlab,
                short.panel.labs = FALSE, add = "jitter") +
    labs( y = 'oncogene vs genome gene') + 
    scale_shape_manual(values=seq(1,l)) +
    stat_compare_means(method = 'wilcox.test',label = "p.format", label.x=1.25, label.y.npc="top", paired = TRUE) +
    facet_wrap(~Cellline, scales = "free_y",ncol = 6, as.table=TRUE,labeller = label_value) +
    theme(text= element_text(color = "black"), panel.grid.major = element_blank(), axis.title.x=element_blank(),
          legend.text=element_text(color = "black",size = 11),legend.position ='right') 
  p + ggsave(filename=out,width =14, height = 4)
  
}
InFile <- as.data.frame(read.csv(InFile, sep='\t',header=TRUE))
Gpair(InFile, OUFile)

