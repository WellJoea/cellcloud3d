library(DOSE)
library(org.Hs.eg.db)
library(org.Mmu.eg.db)
library(org.Mm.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)
library(ggplot2)
library(gridExtra)
library("RColorBrewer")
keytypes(org.Hs.eg.db) 

#library(AnnotationHub)
#hub <- AnnotationHub()


GKplot <- function(Enrich0, FList=NULL, outpre='GO', OrgDb="org.Hs.eg.db", method='go', showCategory=40, pform='png'){
    Enrich <- setReadable(Enrich0, OrgDb, 'ENTREZID')
    write.csv(summary(Enrich), paste0(outpre, method, '.enrich.csv'), row.names =FALSE)
    
    try({
        dp <- enrichplot::dotplot(Enrich, title= paste0(outpre, method, '_dot'), 
                                  font.size=11,showCategory=showCategory)
        plot(dp)
        ggplot2::ggsave( filename= paste0(outpre, method, '_dot.', pform), dp, width = 8, height = 15)
    })

    try({
        bp <- barplot(Enrich, title= paste0(outpre, method, '_bar'), font.size=11, showCategory=showCategory, drop=T)
        plot(bp)
        ggsave( filename= paste0(outpre, method, '_bar.', pform), bp, width = 8, height = 15)
    })

    try({
        cp <- enrichplot::cnetplot(Enrich, title= paste0(outpre, method, '_cnet'), 
                                   foldChange=FList, 
                                    showCategory=10, circular = FALSE, colorEdge =TRUE)
        plot(cp)
        ggplot2::ggsave( filename= paste0(outpre, method, '_cnet.', pform), cp, width = 15, height = 12)
    })

    try({
        hm.palette <- colorRampPalette(rev(brewer.pal(9, 'YlOrRd')), space='Lab')
        hp <- enrichplot::heatplot(Enrich, showCategory=showCategory, foldChange=FList) +
                ggplot2::coord_flip() +
                ggplot2::scale_fill_gradientn(colours = hm.palette(100)) +
                ggplot2::ggtitle(paste0(outpre, method, '_heat'))+
                ggplot2::theme_gray(base_size = 14) +
                ggplot2::theme(axis.text.x=element_text(angle=60, hjust=1))
        plot(hp)
        ggplot2::ggsave( filename= paste0(outpre, method, '_heatmap.', pform), hp, width = 20, height = 15)
    })

    try({
        Enrich2 <- enrichplot::pairwise_termsim(Enrich)
        p1 <- enrichplot::treeplot(Enrich2, showCategory=showCategory, hclust_method = "ward.D") + 
                ggtitle(paste0(outpre, method, 'tree_ward.D'))
        p2 <- enrichplot::treeplot(Enrich2, showCategory=showCategory, hclust_method = "average")+ 
                ggtitle(paste0(outpre, method, 'tree_average'))
        #tp <- aplot::plot_list(list(p1, p2)) + plot_annotation(tag_levels='A')
        tp <- grid.arrange(grobs = list(p1, p2), ncol = 1)
        ggplot2::ggsave( filename= paste0(outpre, method, '_treeplot.', pform), tp, width = 15, height = 15)
    })

    try({
        up <- enrichplot::upsetplot(Enrich)
        plot(up)
        ggplot2::ggsave( filename= paste0(outpre, method, '_upset.', pform), up, width = 15, height = 15)
    })

    try({
        if (method=='kegg'){
            Enrich2 <- enrichplot::pairwise_termsim(Enrich)
            ep <- enrichplot::emapplot(Enrich2, title= paste0(outpre, method, '_emap'), showCategory=showCategory,
                                       pie_scale=1.5,layout="kk") 
            plot(ep)
            ggsave(paste0(outpre, method, "_emapplot.", pform), ep, width = 9, height = 9)
        }
    })
}

GetGOKegg <-function(Glist, FList=NULL, method='go', organism = 'hsa',
                     ont='ALL',pool=FALSE, outpre=NULL,...){
    ###########backgroundGeneset
    #data(geneList, package="DOSE") #backgrouds
    #gene <- names(geneList)[abs(geneList) > 2]
    #gene.df <- bitr(gene, fromType = "ENTREZID", toType = c("ENSEMBL", "SYMBOL"), OrgDb = org.Hs.eg.db)
    #head(gene.df,2)
    #####################GO
    if (organism %in% c('hsa', 'human')){
        OrgDb = org.Hs.eg.db
    }else if(organism == 'mcc'){
        OrgDb = org.Mmu.eg.db
    }else if(organism == 'mmu'){
        OrgDb = org.Mm.eg.db
    }
    print(columns(OrgDb))
    Genes <- bitr(Glist, fromType="SYMBOL", toType=c("ENSEMBL", "ENTREZID"),  OrgDb=OrgDb)

    ggo <- groupGO(gene = Genes$ENTREZID, 
              OrgDb = OrgDb,
              ont = "CC",
              level = 3,
              readable = TRUE)
    
    if (method == 'go'){
        GK <- clusterProfiler::enrichGO(gene = unique(Genes$ENTREZID), 
                #universe = names(geneList),
                OrgDb = OrgDb, #organism="human"
                keyType = 'ENTREZID',
                ont = ont, #CC, BP, MF 
                pAdjustMethod = "BH",
                pvalueCutoff =0.05,
                qvalueCutoff = 0.2,
                pool = pool,
                readable = TRUE) #Gene ID to gene Symbol 
    }
    if (method == 'kegg'){
        #######################KEGG
        library(R.utils);
        kegg_link <- function (target_db, source_db) 
        {
            R.utils::setOption("clusterProfiler.download.method",'wget')
            options(clusterProfiler.download.method = "wget")
            getOption("clusterProfiler.download.method")

            url <- paste0("http://rest.kegg.jp/link/", target_db, "/", 
                source_db, collapse = "")
            print(url)
            clusterProfiler:::kegg_rest(url)
        }

        R.utils::setOption("clusterProfiler.download.method",'wget')
        options(clusterProfiler.download.method = "wget")
        reassignInPackage("kegg_link", pkgName="clusterProfiler", kegg_link)

        GK <- enrichKEGG(gene = unique(Genes$ENTREZID),
                        organism = organism,  ## hsa: human
                        keyType = 'kegg', 
                        pvalueCutoff = 0.05,
                        pAdjustMethod = 'BH', 
                        minGSSize = 3,
                        maxGSSize = 500,
                        qvalueCutoff = 0.2,
                        use_internal_data = FALSE)
    }

    if(!is.null(outpre)){
        try({GKplot(GK, FList=FList, method=method, outpre=outpre, OrgDb=OrgDb, ...)})
    }
    return(GK)
}
