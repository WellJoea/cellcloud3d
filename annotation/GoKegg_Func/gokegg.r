library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)
library(ggplot2)
library(gridExtra)
library("RColorBrewer")
keytypes(org.Hs.eg.db) 

args <-commandArgs(T)
INFile   <- args[1]
OUTDir   <- args[2]

#INFile <- '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/Circos/20210405/data/cancer.gene.txt'
#OUTDir <- '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/20210405/GoKegg'
geneFile <- read.csv(INFile, header=F)
dir.create(OUTDir, showWarnings = TRUE, recursive = TRUE)
setwd(OUTDir)

Genes <- bitr(geneFile[,1], 
            fromType="SYMBOL",
            toType=c("ENSEMBL", "ENTREZID"), 
            OrgDb="org.Hs.eg.db")
head(Genes,2)

###########backgroundGeneset
#data(geneList, package="DOSE") #富集分析的背景基因集
#gene <- names(geneList)[abs(geneList) > 2]
#gene.df <- bitr(gene, fromType = "ENTREZID", toType = c("ENSEMBL", "SYMBOL"), OrgDb = org.Hs.eg.db)
#head(gene.df,2)
#####################GO
ggo <- groupGO(gene = Genes$ENTREZID, 
              OrgDb = org.Hs.eg.db,
              ont = "CC",
              level = 3,
              readable = TRUE)

ego_ALL <- enrichGO(gene = Genes$ENTREZID, 
                    #universe = names(geneList),
                    OrgDb = 'org.Hs.eg.db', #organism="human"
                    keyType = 'ENTREZID',
                    ont = "ALL", #CC, BP, MF 
                    pAdjustMethod = "BH",
                    pvalueCutoff =0.05,
                    qvalueCutoff = 0.2,
                    readable = TRUE) #Gene ID to gene Symbol

head(ego_ALL,2)
dim(ego_ALL[ego_ALL$ONTOLOGY=='BP',])
dim(ego_ALL[ego_ALL$ONTOLOGY=='CC',])
dim(ego_ALL[ego_ALL$ONTOLOGY=='MF',])

#######################KEGG
kegg <- enrichKEGG(gene = Genes$ENTREZID,
                   organism = 'hsa',  ## hsa为人的简写，bta是牛的简写 
                   keyType = 'kegg', 
                   pvalueCutoff = 0.05,
                   pAdjustMethod = 'BH', 
                   minGSSize = 3,
                   maxGSSize = 500,
                   qvalueCutoff = 0.2,
                   use_internal_data = FALSE)

OUT <- function(Enrich, type='GO', showC=20){
    Enrich <- setReadable(Enrich, 'org.Hs.eg.db', 'ENTREZID')
    write.csv(as.data.frame(summary(Enrich)), paste0(type, '.enrich.csv'), row.names =FALSE)

    dotplot(Enrich, title= paste0(type, '_dot'), showCategory=showC) + 
        ggsave( filename= paste0(type, '_dot.pdf'),   width = 10, height = 6)

    barplot(Enrich, title= paste0(type, '_bar'), showCategory=showC, drop=T) + 
        ggsave( filename= paste0(type, '_bar.pdf'), width = 10, height = 6)

    cnetplot(Enrich, title= paste0(type, '_cnet'), showCategory=showC, circular = TRUE, colorEdge =TRUE) + 
        ggsave( filename= paste0(type, '_cnet.pdf'), width = 10, height = 7)

    hm.palette <- colorRampPalette(rev(brewer.pal(9, 'YlOrRd')), space='Lab')
    heatplot(Enrich, showCategory=showC) +
        ggplot2::coord_flip() +
        ggplot2::scale_fill_gradientn(colours = hm.palette(100)) +
        ggplot2::ggsave( filename= paste0(type, '_heatmap.pdf'), width = 10, height = 10)

    if (type=='KEGG'){
        emapplot(Enrich, title= paste0(type, '_emapplot'), showCategory=showC, pie_scale=1.5,layout="kk") +
            ggsave(paste0(type, "_emapplot.pdf"), width = 9, height = 9)
    }
}
OUT(ego_ALL)
#OUT(kegg, type='KEGG')
