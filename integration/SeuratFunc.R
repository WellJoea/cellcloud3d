suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(hash))

#Sys.unsetenv("GITHUB_PAT")
##################function region
#####IO
Read10XData <- function(IN, proj='seurat', ish5=FALSE){
    print(IN)
    pbmc.counts <- ifelse( ish5, Read10X_h5(IN), Read10X(data.dir = IN))
    pbmc.counts <- CreateSeuratObject(
                            counts = pbmc.counts,
                            project= proj,
                            assay  = "RNA",
                            min.cells = 3, 
                            min.features = 10)
    return(pbmc.counts)
}

READ10X <- function(IN, proj='seurat', ish5=FALSE){
    print(IN)
    pbmc.counts <- ifelse( ish5, Read10X_h5(IN), Read10X(data.dir = IN))
    colnames(pbmc.counts ) = paste(colnames(pbmc.counts), 
                                   unlist(lapply(strsplit(colnames(pbmc.counts),'-'), function(x) x[2])), 
                                   sep='-')
    pbmc.counts <- CreateSeuratObject(
                            counts = pbmc.counts,
                            project= proj,
                            assay  = "RNA",
                            min.cells = 3, 
                            min.features = 10)
    return(pbmc.counts)
}
                                          
#####Add infor
ccscoreNew <-function(seur.obj, Nomal=TRUE){
    seur.obj.Raw <- seur.obj
    cc.genes.updated.2019 <- readRDS('/home/gpfs/home/wulab17/WorkSpace/11Project/02DRG/01Analysis/cc.genes.updated.2019.20210519.rds')
    if (Nomal){seur.obj <- NormalizeData(seur.obj)}
    seur.obj <- CellCycleScoring(seur.obj,
                    s.features = cc.genes.updated.2019$s.genes,
                    g2m.features = cc.genes.updated.2019$g2m.genes,
                    set.ident = TRUE)
    OO <- rownames(seur.obj@meta.data)
    RR <- rownames(seur.obj.Raw@meta.data)
    print(OO[OO!=RR])
    seur.obj.Raw[['S.Score']] <- seur.obj[['S.Score']]
    seur.obj.Raw[['G2M.Score']] <- seur.obj[['G2M.Score']]
    seur.obj.Raw[['CC.Diff']] <- seur.obj[['S.Score']] - seur.obj[['G2M.Score']]
    seur.obj.Raw[['Phase']] <- seur.obj[['Phase']]
    return(seur.obj.Raw)
}
                                                                                                                           

Doublettect <-function(seur.obj, SCT=FALSE, doublet.rate=0.075, NPC=50){
    library(DoubletFinder)
    seur.obj.Raw <- seur.obj
    if (SCT){
        seur.obj <- SCTransform(seur.obj)
    }else{
        seur.obj <- NormalizeData(seur.obj)
        seur.obj <- FindVariableFeatures(seur.obj, selection.method = "vst", nfeatures = 2000)
        options(future.globals.maxSize = 300 * 1024 ^ 3)
        plan("multicore", workers = 6)
        seur.obj <- ScaleData(seur.obj)
    }
    seur.obj <- RunPCA(seur.obj)
    seur.obj <- FindNeighbors(seur.obj, reduction = "pca", dims = 1:NPC)
    seur.obj <- FindClusters(seur.obj, resolution = 1, verbose =FALSE)
    seur.obj <- RunUMAP(seur.obj, dims = 1:NPC)
    #seur.obj <- RunTSNE(seur.obj, dims = 1:NPC)

    ## pK Identification (no ground-truth) ---------------------------
    sweep.res.list <- paramSweep_v3(seur.obj, PCs = 1:NPC, sct = SCT)
    sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
    bcmvn <- find.pK(sweep.stats)
    pk_v <- as.numeric(as.character(bcmvn$pK))
    pk_good <- pk_v[bcmvn$BCmetric==max(bcmvn$BCmetric)]

    ## Homotypic Doublet Proportion Estimate ------------------------
    annotations <- seur.obj@meta.data$seurat_clusters
    homotypic.prop <- modelHomotypic(annotations)      ## ex: annotations <- seur.obj@meta.data$ClusteringResults
    nExp_poi <- round(doublet.rate*nrow(seur.obj@meta.data))  ## Assuming 7.5% doublet formation rate - tailor for your dataset
    nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))

    ## Run DoubletFinder with varying classification stringencies ----
    pn_set <- 0.25
    DFname <- paste('DF.classifications', pn_set, pk_good, nExp_poi.adj, sep='_') ##name check
    DFscore<- paste('pANN', pn_set, pk_good, nExp_poi.adj, sep='_')
    
    seur.obj <- doubletFinder_v3(seur.obj, PCs = 1:NPC, pN = pn_set, pK = pk_good, nExp = nExp_poi.adj, sct = SCT)
    OO <- rownames(seur.obj@meta.data)
    RR <- rownames(seur.obj.Raw@meta.data)
    print(OO[OO!=RR])
    #rename(PANN =starts_with("pANN_"))
    seur.obj.Raw[['DF.Score']] <- seur.obj@meta.data[,DFscore]
    seur.obj.Raw[['DF.Class']] <- seur.obj@meta.data[,DFname]
    return(seur.obj.Raw)
}

QCplot <- function(seur.obj, header='before', group.by='sampleid', pt.size=0.05,
                   FTs=c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.ribo","percent.hb")){
    pl.count <- VlnPlot(seur.obj, features = FTs, ncol = 5)
    ggsave( paste0(header, ".count.vln.pdf"), pl.count, width = 18, height = 10) 

    pf1 <-ggplot(data=seur.obj@meta.data, aes(nFeature_RNA,..density..))+
                geom_histogram(color='white',fill='gray60',size=0.005, binwidth = 10)+
                geom_line(stat='density')
    pf11<-ggplot(data=seur.obj@meta.data, aes(nCount_RNA,..density..))+
                geom_histogram(color='white',fill='gray60',size=0.005, binwidth = 10)+
                geom_line(stat='density')
    pf2 <-ggplot(data=seur.obj@meta.data, aes_string(x='nFeature_RNA', color=group.by))+
            geom_density(alpha=0.6, size=0.5)
    pf22<-ggplot(data=seur.obj@meta.data, aes_string(x='nCount_RNA', color=group.by))+
            geom_density(alpha=0.6, size=0.5)
    options(repr.plot.height = 12, repr.plot.width = 15)
    pf.dens<- grid.arrange(grobs = list(pf1, pf2, pf11,pf22), ncol = 2)
    ggsave( paste0(header, ".feature.density.pdf"), pf.dens, width = 15, height = 12) 

    options(repr.plot.height = 20, repr.plot.width = 30)
    pl.countg <- VlnPlot(seur.obj, features = FTs, ncol = 3, group.by=group.by, pt.size=pt.size) 
    ggsave( paste0(header, ".count.vln.groupby.pdf"), pl.countg, width = 30, height = 20) 

    plot1 <- FeatureScatter(seur.obj, feature1 = "nCount_RNA", feature2 = "percent.mt", group.by=group.by)
    plot2 <- FeatureScatter(seur.obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA", group.by=group.by) 
    options(repr.plot.height = 10, repr.plot.width = 18)
    pl.fscatter <- grid.arrange(grobs = list(plot1, plot2), ncol = 2)
    ggsave( paste0(header, ".feature.scatter.pdf"), pl.fscatter, width = 18, height = 10)
    pl.countg
}

Charplot <-function(scrnaObj, header='before', group.by='sampleid', chartype='DF.Class'){ 
    K = scrnaObj@meta.data
    K = K %>% 
            group_by(K[group.by]) %>% 
            count(K[chartype], name="cellnumber") %>%
            mutate(cellfrq= cellnumber/sum(cellnumber))
    p1 <- ggplot(K, aes_string(x=group.by, y="cellnumber", fill=chartype)) + 
            geom_bar(stat="identity",position = "stack")+
            theme(axis.text.x=element_text(angle=90, hjust=1))
    p2 <- ggplot(K, aes_string(x=group.by, y="cellnumber", fill=chartype)) + 
            geom_bar(stat="identity",position = "fill")+
            theme(axis.text.x=element_text(angle=90, hjust=1))
    pl.fscatter <- grid.arrange(grobs = list(p1, p2), ncol = 2)
    ggsave( paste(header, chartype, group.by, "feature.barplot.pdf", sep='.'), pl.fscatter, width = 12, height = 8) 
}

MarkerCheck <- function(seur.obj, features, group.by='sampleid', header='before',...){
    pv <- VlnPlot(seur.obj, features = features, group.by=group.by,...)
    ggsave( paste(header, group.by, "feature.VlnPlot.png", sep='.'), pv , width = 42, height = 36, limitsize = FALSE)
    pl <- VlnPlot(seur.obj, features = features, group.by=group.by, log=TRUE, ...)
    ggsave( paste(header, group.by, "feature.VlnPlot.log.png", sep='.'), pl , width = 42, height = 36, limitsize = FALSE)
}
######PreProcessing-standardization-classical##############
'%!in%' <- function(x,y)!('%in%'(x,y))
                                          
GuessNfeat <-function(seur.obj, featues){
    cellNum <- dim(seur.obj)[2]
    if ( is.null(featues) || featues==0 ){ return( NULL ) }
    if (class(featues)=='character'){
        if(featues=='all.gene'){ return(rownames(seur.obj))}
        if(featues=='var'){ return(Seurat::VariableFeatures(seur.obj)) }
        if(featues=='guess'){ return(floor(1000*log(cellNum/1000, exp(1)+0.5))) }
        if(featues=='Guess'){ return(floor((cellNum/1000/exp(1))^(1/2)*1000)) }
        if(length(featues)>1){return(featues)}
    }
    if (class(featues)=='numeric'){
        if (featues >0 && featues <=1){return(floor(cellNum*featues)) }
        if (featues >1){ return(floor(featues)) }
    }
}

colrows<- function(ncell, ncols=NULL, nrows=NULL, soft=TRUE){
    if (is.null(ncols) & is.null(nrows)){
        ncols = 3
    }
    if (!is.null(ncols)){
        nrows = ceiling(ncell/ncols)
        ncols = min(ncell, ncols)
    }else if(!is.null(nrows)){
        ncols = ceiling(ncell/nrows)
        nrows = min(ncell, nrows)
    }

    if ((soft) & (ncell> 1) & (ncell - ncols*(nrows-1)<=1)){
        ncols = ncols + 1
        nrows = nrows - 1
    }
    return( c(nrows, ncols) )
}

                                          
CNomal <- function(seur.obj){
    seur.obj <- NormalizeData(seur.obj, 
                                normalization.method = "LogNormalize",
                                margin = 1,
                                scale.factor = 10000)
    return (seur.obj)
}

HGVS <- function(seur.obj, nfeatures='Guess'){
    nfeatures <- GuessNfeat(seur.obj, nfeatures)
    print(paste0('The feature used for normalizeData is ', nfeatures))
    seur.obj <- FindVariableFeatures(seur.obj, 
        selection.method = "vst", 
        loess.span = 0.3,
        clip.max = "auto",
        num.bin = 20,
        binning.method = "equal_width",
        nfeatures = nfeatures,
        mean.cutoff = c(0.1, 8),
        dispersion.cutoff = c(1, Inf),
        verbose = TRUE)
    topn <- head(VariableFeatures(seur.obj), 10)
    plot1 <- VariableFeaturePlot(seur.obj)
    plot2 <- LabelPoints(plot = plot1, points = topn, repel = TRUE)
    pl.topnhvd <- ggpubr::ggarrange(plot1, plot2, ncol = 2)
    pl.topnhvd
    ggsave( paste0("FindVariableFeatures.", nfeatures, ".pdf"), pl.topnhvd, width = 15, height = 9) 
    return(seur.obj)
}

VariableFeaturePlot.list <- function(seur.list, ncols=4, pscale=4){
    plots <- lapply(X = seq_along(seur.list), FUN = function(x){
               VariableFeaturePlot(seur.list[[x]]) + labs(title=names(seur.list[x]))
    })
    CR <- colrows(length(seur.list), ncols=ncols)
    ncol <- CR[2]
    nrow <- CR[1]
    options(repr.plot.height = nrow*pscale, repr.plot.width = ncol*pscale)
    pall <- grid.arrange(grobs = plots,ncol = ncol)
    pall
}                                        

HGVSbatch <- function(seur.obj, batch.by='sampleid', anchor.features = 'Guess', features='Guess',
                      partial.featues = NA,
                      batch.com=NULL, min.features = 200,
                      isnormal=FALSE, defassay='RNA', dropMT=FALSE, dropRi=FALSE, dropHb=FALSE){
    DefaultAssay(seur.obj) <- defassay
    if(isnormal){seur.obj <- NormalizeData(seur.obj)}
    anchor.features <- GuessNfeat(seur.obj, anchor.features)
    
    if (length(unique(seur.obj@meta.data[,batch.by]))==1){
        print(sprintf('Only one batch id in %s!', batch.by))
        seur.obj <- HGVS(seur.obj, nfeatures=anchor.features)
        select.features <- VariableFeatures(seur.obj)
        coun.features <- sort(table(select.features), decreasing = TRUE)
    }else{
        seur.list <- SplitObjectO(seur.obj, split.by = batch.by)
        seur.list <- lapply(X = names(seur.list), FUN = function(x){
                if (is.na(partial.featues) | is.na(partial.featues[x])){
                    ifeatures <- features
                }else{
                    ifeatures <- partial.featues[[x]]
                }
                i.seur <- seur.list[[x]]
                ifeatures <- GuessNfeat(i.seur, ifeatures)
                print(sprintf('selct %s feature in batch id %s.', ifeatures, x))
                i.seur <- FindVariableFeatures(i.seur, selection.method = "vst", nfeatures = ifeatures)
        })
        all.features <- c()
        for (iseu in seur.list){ all.features <- c(all.features, VariableFeatures(iseu)) }
        coun.features <- sort(table(all.features), decreasing = TRUE)

        if (!is.null(batch.com)) {
            select.features <- names(coun.features)[coun.features>=batch.com]
            if (length(select.features)<= min.features){
                warning(sprintf('Only get %s final features, lower than %s!', length(select.features), min.features))
            }
        }else{
            select.features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
        }
    }
    select.features <- dropfeature(select.features, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
    print(sprintf('final features: %s, in %s batches.', length(select.features), min(coun.features[select.features])))

    VariableFeatures(seur.obj) <- select.features
    vst.variable <- rownames(seur.obj@assays[[defassay]]@meta.features) %in% select.features
    seur.obj@assays[[defassay]]@meta.features$vst.variable <- vst.variable
    seur.obj@assays[[defassay]]@meta.features$vst.variable.counts <- as.vector(coun.features[rownames(seur.obj@assays[[defassay]]@meta.features)])
    
    seur.obj
}
                        
dropHGVSfeature <- function(seur.obj, dropMT=FALSE, dropRi=FALSE, dropHb=FALSE){
    mito.genes <- grep(pattern = "^MT-|^mt-|^Mt-", rownames(seur.obj), value = TRUE)
    ribo.genes <- grep(pattern = "^RP[SL]|^Rp[sl]", rownames(seur.obj), value = TRUE)
    hb.genes   <- grep(pattern = "^HB[^(P)]|^Hb[^(p)]", rownames(seur.obj), value = TRUE)
    features <- VariableFeatures(seur.obj)
    
    if(dropMT) features <- setdiff(features, mito.genes)
    if(dropRi) features <- setdiff(features, ribo.genes)
    if(dropHb) features <- setdiff(features, hb.genes)
    VariableFeatures(seur.obj) <- features
    seur.obj
}

dropfeature <- function(features, dropMT=FALSE, dropRi=FALSE, dropHb=FALSE){
    flen <- length(features)
    mito.genes <- grep(pattern = "^MT-|^mt-|^Mt-", features, value = TRUE)
    ribo.genes <- grep(pattern = "^RP[SL]|^Rp[sl]", features, value = TRUE)
    hb.genes   <- grep(pattern = "^HB[^(P)]|^Hb[^(p)]", features, value = TRUE)
    
    if(dropMT) features <- setdiff(features, mito.genes)
    if(dropRi) features <- setdiff(features, ribo.genes)
    if(dropHb) features <- setdiff(features, hb.genes)
    klen <- length(features)
    print(sprintf("drop %s features..", flen - klen))
    features
}


SplitObjectO <- function(seur.obj, split.by='sampleid'){
    LEV <- tryCatch({levels(droplevels(seur.obj@meta.data[,split.by]))},
                    error = function(e){unique(seur.obj@meta.data[,split.by])}
                   ) %>% as.character()
    print(LEV)
    seu.list = list()
    seu.split= Seurat::SplitObject(seur.obj, split.by = split.by)
    for (ilv in LEV){
        seu.list[[ilv]] = seu.split[[ilv]]
    }
    rm(seu.split)
    return(seu.list)
}
  
CScale <- function(seur.obj, features=NULL,
                   vars.to.regress=c( "nCount_RNA","percent.mt"), ... ){
                    # "nCount_RNA","nFeature_RNA", "CC.Diff", "percent.mt" ##overcorrected!!!
    all.genes <- GuessNfeat(seur.obj, features)
    #memory.limit(30000)
    options(future.globals.maxSize = 50 * 1024 ^ 10)
    plan("multicore", workers = 10)
    seur.obj <- ScaleData(seur.obj,
                        features = all.genes,
                        model.use = "linear",
                        vars.to.regress = vars.to.regress,
                        use.umi = FALSE,
                        scale.max = 10,
                        block.size = 1000,
                        do.scale = TRUE,
                        do.center = TRUE,
                        min.cells.to.block = 3000,
                        verbose = TRUE, ...)
    return(seur.obj)
}

CSCT <- function(seur.obj, nfeatures='Guess', ncells=5000,
                    vars.to.regress=c( "nCount_RNA", "percent.mt"),...){
    nfeatures <- GuessNfeat(seur.obj, nfeatures)
    seur.obj <- SCTransform(seur.obj, vars.to.regress = vars.to.regress, method = "glmGamPoi",
                            variable.features.n=nfeatures, ncells=ncells, verbose = FALSE, ...)
    return(seur.obj)
}
                           
rawcheack <-function(seur.obj, defaltas = 'integrated', groubpy='sampleid', headname='Notbatch', NPCS=50, DIMS=30, k.param=20 ){
        #DefaultAssay(seur.obj) <- defaltas
        seur.obj <- RunPCA(seur.obj, verbose = FALSE, npcs=NPCS)

        seur.obj<- FindNeighbors(seur.obj, reduction = "pca", k.param= k.param,  dims = 1:DIMS)
        seur.obj<- FindClusters(seur.obj, resolution = 0.8, verbose =FALSE)

        seur.obj <- RunUMAP(seur.obj, dims = 1:DIMS)
        seur.obj <- RunTSNE(seur.obj, dims = 1:DIMS)

        p1 <- DimPlot(seur.obj, reduction = "umap", raster =FALSE, group.by = groubpy)
        p2 <- DimPlot(seur.obj, reduction = "umap", raster =FALSE, group.by = "seurat_clusters", label = TRUE,  repel = TRUE)
        ggsave(paste(headname, groubpy, 'fastcheack.umap.groups.DimPlot.pdf', sep='_'), 
               grid.arrange(grobs = list(p1, p2), ncol = 2), width = 18, height = 9)

        p3 <- DimPlot(seur.obj, reduction = "tsne", raster =FALSE, group.by = groubpy)
        p4 <- DimPlot(seur.obj, reduction = "tsne", raster =FALSE, group.by = "seurat_clusters", label = TRUE,  repel = TRUE)
        ggsave(paste(headname, groubpy, 'fastcheack.tsne.groups.DimPlot.pdf', sep='_'), 
               grid.arrange(grobs = list(p3, p4), ncol = 2), width = 18, height = 9)
}

PCACLS <- function(seur.obj, NPCS=100, header='cheack', dimheatpc=6, dodimheat=FALSE, group.by = NULL, features=NULL, JSplot=FALSE){
    features <- GuessNfeat(seur.obj, features)
    seur.obj <- RunPCA(seur.obj, 
                    features = features,
                    npcs = NPCS,
                    rev.pca = FALSE,
                    weight.by.var = TRUE,
                    verbose = TRUE,
                    ndims.print = 1:5,
                    nfeatures.print = 5,
                    reduction.key = "PC_",
                    seed.use = 42,
                    approx = TRUE,
        )

    pl.pca <- DimPlot(seur.obj, dims = c(1, 2), group.by=group.by, raster =FALSE, reduction = "pca")
    pl.elb <- ElbowPlot(seur.obj, ndims=NPCS)
    pall<-grid.arrange(grobs = list(pl.pca, pl.elb),ncol = 2) 
    ggsave(paste(header, NPCS, "pca.ElbowPlot.pdf", sep='_'), pall, width = 16, height = 8)
    
    pl.pcV <- VizDimLoadings(seur.obj, dims = 1:dimheatpc, ncol=3, reduction = "pca")
    ggsave(paste(header, NPCS, "pca.ElbowPlot.pdf", sep='_'), pl.pcV, width = 25, height = 30)

    if(dodimheat){
        pdf(paste(header, NPCS, "pca.DimHeatmap.png", sep='_'), width = 18, height = 12)
        DimHeatmap(seur.obj, dims = 1:dimheatpc, ncol = 3, balanced = TRUE)
        dev.off()
    }
    ##Determining
    if (JSplot){
        options(future.globals.maxSize = 50 * 1024 ^ 7)
        plan("multicore", workers = 7)
        seur.obj <- JackStraw(seur.obj, dims = NPCS, num.replicate = 50)  #not SCTransform

        seur.obj <- ScoreJackStraw(seur.obj, dims = 1:NPCS)
        pl.jsp <- JackStrawPlot(seur.obj, dims = 1:NPCS)
        ggsave( paste(header, NPCS, "_JackStrawPlot.pdf", sep="_"), pl.jsp, width = 12, height = 10) 
    }
    return(seur.obj)
}

CLUSTER <- function(seur.obj, DIMS=50, RES = c(0.2, 0.6, 0.8, 1, 2, 3), reduction='pca',  min.dist=0.3,
                    n.neighbors=30L, algorithm=1, runtsne=FALSE, 
                    metric='cosine', umap.method="uwot", 
                     defassay = 'RNA', header='cheack',  k.param=20, run3d=FALSE){
    print(sprintf('Do FindNeighbors (%s)...', DIMS))
    seur.obj <- FindNeighbors(seur.obj, reduction = reduction, k.param=k.param, dims = 1:DIMS, verbose = FALSE)
    #SO <- FindNeighbors(SO,reduction = "umap", dims = 1:2, force.recalc = T)
    print('Do FindClusters...')
    if(length(RES)>1){
        options(future.globals.maxSize = 50 * 1024 ^ 10)
        plan("multicore", workers = 10)
        seur.obj <- FindClusters( seur.obj, resolution = RES, algorithm=algorithm, verbose =FALSE)
    }else{
        seur.obj <- FindClusters( seur.obj, resolution = RES, algorithm=algorithm, verbose =FALSE)
    }
    sprintf('Do RunUMAP (%s)...', DIMS)
    seur.obj <- RunUMAP(seur.obj, reduction = reduction, metric=metric,
                        n.neighbors=n.neighbors, dims = 1:DIMS, min.dist=min.dist, 
                        verbose = FALSE,
                        umap.method=umap.method)
    if (runtsne){
         seur.obj <- RunTSNE(seur.obj, reduction = reduction, dims = 1:DIMS, check_duplicates = FALSE)
    }

    if (run3d){
        seur.obj <- RunUMAP(seur.obj, reduction = reduction, dims = 1:DIMS, min.dist=min.dist,
                            umap.method=umap.method, metric=metric,
                            n.neighbors=n.neighbors, n.components=3L,reduction.name = "umap3d", reduction.key = "UMAP3D_")
        if (runtsne){
            seur.obj <- RunTSNE(seur.obj, reduction = reduction, dims = 1:DIMS,
                            dim.embed = 3, reduction.name = "tsne3d", reduction.key = "tSNE3D_")
        }
    }
    return(seur.obj)
}

CLUSTERquick <- function(seur.obj, DIMS=50, RES = c(0.2, 0.6, 0.8, 1, 2, 3), reduction='pca', min.dist=0.3,
                         n.neighbors=30L, algorithm=1,
                         metric='cosine', umap.method="uwot", defassay = 'RNA',
                         header='cheack', k.param=20, runtsne=FALSE, run3d=FALSE){
    print(sprintf('Do FindNeighbors (%s)...', DIMS))
    seur.obj <- FindNeighbors(seur.obj, reduction = reduction, k.param=k.param,  dims = 1:DIMS)
    print('Do FindClusters...')
    if(length(RES)>1){
        options(future.globals.maxSize = 50 * 1024 ^ 10)
        plan("multicore", workers = 10)
        seur.obj <- FindClusters( seur.obj, resolution = RES, algorithm=algorithm, verbose =FALSE)
    }else{
        seur.obj <- FindClusters( seur.obj, resolution = RES, algorithm=algorithm, verbose =FALSE)
    }
    print(sprintf('Do RunUMAP (%s)...', DIMS))
    seur.obj <- RunUMAP(seur.obj, reduction = reduction, min.dist=min.dist, metric=metric,
                        n.neighbors=n.neighbors, dims = 1:DIMS, umap.method=umap.method)
    if (runtsne){
         seur.obj <- RunTSNE(seur.obj, reduction = reduction, dims = 1:DIMS, check_duplicates = FALSE)
    }
    if (run3d){
        seur.obj <- RunUMAP(seur.obj, reduction = reduction, dims = 1:DIMS, n.components=3L, 
                            min.dist=min.dist, umap.method=umap.method, metric=metric,
                            reduction.name = "umap3d", reduction.key = "UMAP3D_")
     }
    return(seur.obj)
}

Harmony <-function(seur.obj, group.by='sampleid', DIMS=NULL, headname='batch'){
    library(harmony)
    if (!is.null(DIMS)){    
        seur.obj@reductions$pca@cell.embeddings  <- seur.obj@reductions$pca@cell.embeddings[,1:DIMS]
        seur.obj@reductions$pca@feature.loadings <- seur.obj@reductions$pca@feature.loadings[,1:DIMS]
    }
    seur.obj <- RunHarmony(seur.obj, group.by.vars=group.by, plot_convergence = TRUE)
    harmony_embeddings <- Embeddings(seur.obj, 'harmony')

    p1 <- DimPlot(object = seur.obj, reduction = "pca", pt.size = .1, raster =FALSE, group.by = group.by)
    p2 <- VlnPlot(object = seur.obj, features = "PC_1", group.by = group.by, pt.size = .1)
    p3 <- DimPlot(object = seur.obj, reduction = "harmony", pt.size = .1, raster =FALSE, group.by = group.by)
    p4 <- VlnPlot(object = seur.obj, features = "harmony_1", group.by = group.by, pt.size = .1)
    ph <- grid.arrange(grobs = list(p1, p2, p3, p4), ncol = 2)
    ggsave(paste(headname, group.by, 'harmony.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 18)
    return(seur.obj)
}

SeuratMnn <- function(seur.obj, group.by='sampleid', headname='batch', features='guess', NPCS=50){
    features <- GuessNfeat(seur.obj, features)
    orders <- levels(seur.obj@meta.data[,group.by])
    seur.obj <- SeuratWrappers::RunFastMNN(object.list = SplitObject(seur.obj, split.by = group.by), features=features, d=NPCS)
    seur.obj@meta.data[,group.by] <- factor(seur.obj@meta.data[,group.by], levels=orders)
    p1 <- DimPlot(object = seur.obj, reduction = "mnn", pt.size = .1, raster =FALSE, group.by = group.by)
    p2 <- VlnPlot(object = seur.obj, features = "mnn_1", group.by = group.by, pt.size = .1)
    ph <- grid.arrange(grobs = list(p1, p2), ncol = 2)
    ggsave(paste(headname, group.by, 'fastmnn.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 9)
    return(seur.obj)
}

FastMnn <- function(seur.obj, headname='batch', DIMS=NULL, group.by='sampleid', mnn.k=20, ...){
    batch_info <- as.factor(seur.obj@meta.data[,group.by])
    if (!is.null(DIMS)){    
        seur.obj@reductions$pca@cell.embeddings  <- seur.obj@reductions$pca@cell.embeddings[,1:DIMS]
        seur.obj@reductions$pca@feature.loadings <- seur.obj@reductions$pca@feature.loadings[,1:DIMS]
    }
    pca_coord  <- seur.obj@reductions$pca@cell.embeddings
    pca_corrected <- batchelor::reducedMNN(pca_coord,batch = batch_info, k=mnn.k, ...)
    seur.obj@reductions$pca@cell.embeddings <- pca_corrected$corrected

    p1 <- DimPlot(object = seur.obj, reduction = "pca", pt.size = .1, raster =FALSE, group.by = group.by)
    p2 <- VlnPlot(object = seur.obj, features = "PC_1", group.by = group.by, pt.size = .1)
    ph <- grid.arrange(grobs = list(p1, p2), ncol = 2)
    ggsave(paste(headname, group.by, 'FastMNN.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 9)
    return(seur.obj)
}

correctMnnX <- function(seur.obj, batch.by='sampleid', anchor.features = 'Guess', features='Guess',
                         isnormal=FALSE, defassay='RNA', dropMT=FALSE, dropRi=FALSE, dropHb=FALSE,...){
    #https://bioinformatics-core-shared-training.github.io/cruk-summer-school-2021/scRNAseq/Markdowns/batchCorrection.html
    DefaultAssay(seur.obj) <- defassay
    if(isnormal){seur.obj <- NormalizeData(seur.obj)}
    anchor.features <- GuessNfeat(seur.obj, anchor.features)

    seur.list <- SplitObjectO(seur.obj, split.by = batch.by)
    seur.list <- lapply(X = seur.list, FUN = function(x){
            features <- GuessNfeat(x, features)
            x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = features)
    })  
    features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
    features <- dropfeature(features, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
    
    batchs = names(seur.list)
    seur.list <- lapply(batchs,  FUN = function(x){
        S <- subset(seur.list[[x]], features = features)
        S@assays[[defassay]]@data
    })
    names(seur.list) = batchs
    mnncor = batchelor::mnnCorrect(seur.list,...)
    return(mnncor)
}
                                          
CCA <- function(seur.obj, group.by='sampleid', features='guess', anchor.features='guess', 
                batch.com=NULL, min.features = 200, local.features=NULL, isnormal=TRUE, iNPCS=NULL,
                k.anchor=5,  k.filter = 200, k.score = 30, k.weight = 100, max.features=200,
                     normalization.method='LogNormalize', aDIMS=30, iDIMS=50, NPCS=100, headname='batch',
                     dropMT=FALSE, dropRi=FALSE, dropHb=FALSE,
                     vars.to.regress=c("nCount_RNA","percent.mt"), ...){
        if(isnormal){seur.obj <- NormalizeData(seur.obj)}
    
        seur.list <- SplitObjectO(seur.obj, split.by = group.by)
        if (is.null(local.features)){
            seur.list <- lapply(X = seur.list, FUN = function(x) {
                features <- GuessNfeat(x, features)
                x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = features)
                x <- dropHGVSfeature(x, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
                x
             }) 
            
            VariableFeaturePlot.list(seur.list)
            anchor.features <- GuessNfeat(seur.obj, anchor.features)
            all.features <- c()
            for (iseu in seur.list){ all.features <- c(all.features, VariableFeatures(iseu)) }
            coun.features <- sort(table(all.features), decreasing = TRUE)
            
            if (!is.null(batch.com)) {
                select.features <- names(coun.features)[coun.features>=batch.com]
                if (length(select.features)<= min.features){
                    warning(sprintf('Only get %s final features, lower than %s!', length(select.features), min.features))
                }
            }else{
                select.features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
            }
            print(sprintf('final features: %s, in %s batches.', length(select.features), min(coun.features[select.features])))
        }else{
            select.features <- local.features
            print(sprintf('final features: %s.', length(select.features)))
        }
        seur.list <- lapply(X = seur.list, FUN = function(x) {
            VariableFeatures(x) <- select.features
            x
        })
        seur.anchors <- FindIntegrationAnchors(object.list = seur.list, anchor.features = select.features,
                                               k.filter = k.filter, k.score=k.score, k.anchor = k.anchor,
                                               max.features=max.features,
                                               reduction = "cca", dims= 1:aDIMS, ...)
        cell.anchors <- table(seur.anchors@anchors[c('dataset1', 'dataset2')])
        print(cell.anchors)
        rm(seur.list)
    
        seur.combined <- IntegrateData(anchorset = seur.anchors, new.assay.name = "integrated",
                                       k.weight = k.weight, 
                                       normalization.method=normalization.method, dims= 1:iDIMS)
    
        seur.combined@meta.data = seur.obj@meta.data[colnames(seur.combined),]
        #seur.combined@meta.data[,group.by] <- factor(seur.combined@meta.data[,group.by], levels=LEV)
        rm(seur.obj)
        rm(seur.anchors)

        DefaultAssay(seur.combined) <- "integrated"
        options(future.globals.maxSize = 50 * 1024 ^ 10)
        plan("multicore", workers = 10)
        seur.combined <- ScaleData(seur.combined, vars.to.regress = vars.to.regress, verbose = FALSE)
        seur.combined <- PCACLS(seur.combined, NPCS=NPCS, group.by=group.by)
        p1 <- DimPlot(object = seur.combined, reduction = "pca", pt.size = .1, raster =FALSE, group.by = group.by)
        p2 <- VlnPlot(object = seur.combined, features = "PC_1", group.by = group.by, pt.size = .1)
        #ph <- grid.arrange(grobs = list(p1, p2),ncol = 2)
        ph <- CombinePlots(plots = list(p1, p2), ncol=2)
        plot(ph)
        ggsave(paste(headname, group.by, 'CCA.groups.DimPlot.pdf', sep='_'), ph, width = 6, height = 12)
        return(seur.combined)
}

CCASCT <- function(seur.obj, group.by='sampleid', aDIMS=30, iDIMS=50, NPCS=100, 
                   iNPCS=NULL, local.features=NULL,
                   anchor.features = 'Guess', headname='batch',features='Guess',ncells=5000,
                   vars.to.regress=c("nCount_RNA", "percent.mt"),...){
        seur.list <- SplitObjectO(seur.obj, split.by = group.by)
        LEV <- levels(droplevels(seur.obj@meta.data[,group.by]))
        seur.list <- lapply(X = seur.list, FUN = function(x) {
                            features <- GuessNfeat(x, features)
                            x <- SCTransform(x, vars.to.regress = vars.to.regress, method = "glmGamPoi",
                                                variable.features.n=features, ncells=ncells) #, verbose = FALSE
                            #x <- dropHGVSfeature(x, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
                            x
        })
        VariableFeaturePlot.list(seur.list)
        anchor.features <- GuessNfeat(seur.obj, anchor.features)
        select.features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
        seur.list <- PrepSCTIntegration(object.list = seur.list, anchor.features = select.features )
        seur.anchors <- FindIntegrationAnchors(object.list = seur.list, anchor.features = select.features ,
                                               normalization.method = "SCT", 
                                               k.filter = k.filter, k.score=k.score, k.anchor = k.anchor,
                                               max.features=max.features,
                                               reduction = "cca", dims= 1:aDIMS)
        cell.anchors <- table(seur.anchors@anchors[c('dataset1', 'dataset2')])
        print(cell.anchors)
        seur.combined <- IntegrateData(anchorset = seur.anchors, new.assay.name = "integrated",
                                       k.weight = k.weight, 
                                       normalization.method = "SCT",
                                       dims= 1:iDIMS)
        seur.combined@meta.data = seur.obj@meta.data[colnames(seur.combined),]
        #seur.combined@meta.data[,group.by] <- factor(seur.combined@meta.data[,group.by], levels=LEV)

        DefaultAssay(seur.combined) <- "integrated"
        seur.combined <- PCACLS(seur.combined, NPCS=NPCS, group.by=group.by)
        p1 <- DimPlot(object = seur.combined, reduction = "pca", pt.size = .1, group.by = group.by)
        p2 <- VlnPlot(object = seur.combined, features = "PC_1", group.by = group.by, pt.size = .1)
        ph <- grid.arrange(grobs = list(p1, p2), ncol = 2)
        ggsave(paste(headname, group.by, 'CCA.SCT.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 9)
        return(seur.combined)
}

RPCA <- function(seur.obj, group.by='sampleid', features='Guess', anchor.features='Guess', local.features=NULL,
                 scale=TRUE, reference =NULL, 
                 batch.com=NULL, min.features = 200, isnormal=TRUE, normalization.method='LogNormalize', aDIMS=50, iDIMS=50, 
                 iNPCS=50, NPCS=100, headname='batch', k.anchor=5,  k.filter = 200, k.score = 30,
                 k.weight = 100, max.features=200, dropMT=FALSE, dropRi=FALSE, dropHb=FALSE,
                 vars.to.regress=c("nCount_RNA","percent.mt"), ...){

        if(isnormal){seur.obj <- NormalizeData(seur.obj)}
        seur.list <- SplitObjectO(seur.obj, split.by = group.by)    
        if (is.null(local.features)){
            seur.list <- lapply(X = seur.list, FUN = function(x) {
                features <- GuessNfeat(x, features)
                x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = features)
                x <- dropHGVSfeature(x, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
                x
             })
            VariableFeaturePlot.list(seur.list)
            anchor.features <- GuessNfeat(seur.obj, anchor.features)
            all.features <- c()
            for (iseu in seur.list){ all.features <- c(all.features, VariableFeatures(iseu)) }
            coun.features <- sort(table(all.features), decreasing = TRUE)
    
            if (!is.null(batch.com)) {
                select.features <- names(coun.features)[coun.features>=batch.com]
                if (length(select.features)<= min.features){
                    warning(sprintf('Only get %s final features, lower than %s!', length(select.features), min.features))
                }
            }else{
                select.features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
            }
            print(sprintf('final features: %s, in %s batches.', length(select.features), min(coun.features[select.features])))
        }else{
            select.features <- local.features
            print(sprintf('final features: %s.', length(select.features) ))
        }
        seur.list <- lapply(X = seur.list, FUN = function(x) {
            VariableFeatures(x) <- select.features
            x
        })
        seur.list <- lapply(X = seur.list, FUN = function(x) {
            x <- ScaleData(x, features = select.features ,  vars.to.regress=vars.to.regress, verbose = FALSE)
            x <- RunPCA(x, features = select.features , npcs = iNPCS, nfeatures.print = 1, verbose = FALSE)
            x
        })    

        seur.anchors <- FindIntegrationAnchors(object.list = seur.list, anchor.features = select.features ,
                                               reference = reference, 
                                               k.filter = k.filter, k.score=k.score, k.anchor = k.anchor,
                                               max.features=max.features, scale=scale,
                                               reduction = "rpca", dims= 1:aDIMS, ...)
        cell.anchors <- table(seur.anchors@anchors[c('dataset1', 'dataset2')])
        print(cell.anchors)
        rm(seur.list)

        seur.combined <- IntegrateData(anchorset = seur.anchors, new.assay.name = "integrated",
                                       k.weight = k.weight, 
                                       normalization.method=normalization.method, dims= 1:iDIMS)
        seur.combined@meta.data = seur.obj@meta.data[colnames(seur.combined),]
        rm(seur.obj)
        rm(seur.anchors)

        DefaultAssay(seur.combined) <- "integrated"
        options(future.globals.maxSize = 50 * 1024 ^ 10)
        plan("multicore", workers = 10)
        seur.combined <- ScaleData(seur.combined, vars.to.regress = NULL, verbose = FALSE)
        seur.combined <- PCACLS(seur.combined, NPCS=NPCS, group.by=group.by)
        p1 <- DimPlot(object = seur.combined, reduction = "pca", pt.size = .1, group.by = group.by)
        p2 <- VlnPlot(object = seur.combined, features = "PC_1", group.by = group.by, pt.size = .1)
        #ph <- grid.arrange(grobs = list(p1, p2), ncol = 2)
        ph <- CombinePlots(plots = list(p1, p2), ncol=2)
        plot(ph)
        ggsave(paste(headname, group.by, 'RPCA.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 9)
        return(seur.combined)
}

RPCASCT <- function(seur.obj, group.by='sampleid', features='Guess', anchor.features='Guess',
                    method='glmGamPoi', normalization.method='SCT', iDIMS=50, NPCS=100, iNPCS=50, 
                     k.anchor=5,  k.filter = 200, k.score = 30, k.weight = 100, max.features=200,
                     dropMT=FALSE, dropRi=FALSE, dropHb=FALSE, local.featur=NULL,
                     vars.to.regress=c("nCount_RNA","percent.mt"), headname='batch', ...){

        seur.list <- SplitObjectO(seur.obj, split.by = group.by)
        seur.list <- lapply(X = seur.list, FUN = function(x) {
            features <- GuessNfeat(x, features)
            x <- SCTransform(x, vars.to.regress =vars.to.regress,
                             variable.features.n=features,
                             method=method) #, verbose = FALSE
            #x <- dropHGVSfeature(x, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
            x
        })
        VariableFeaturePlot.list(seur.list)
        anchor.features <- GuessNfeat(seur.obj, anchor.features)
        select.features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
        seur.list <- PrepSCTIntegration(object.list = seur.list, anchor.features = select.features)
        seur.list <- lapply(X = seur.list, FUN = function(x) {
            x <- RunPCA(x, features = select.features, npcs = iNPCS, nfeatures.print = 1, verbose = FALSE)
            x
        })    

        seur.anchors <- FindIntegrationAnchors(object.list = seur.list, anchor.features = select.features,
                                               k.filter = k.filter, k.score=k.score, k.anchor = k.anchor,
                                               max.features=max.features,
                                               normalization.method=normalization.method,
                                               reduction = "rpca", dims= 1:aDIMS)
        cell.anchors <- table(seur.anchors@anchors[c('dataset1', 'dataset2')])
        print(cell.anchors)
        seur.combined <- IntegrateData(anchorset = seur.anchors, new.assay.name = "integrated",
                                       k.weight = k.weight, 
                                       normalization.method=normalization.method, dims= 1:iDIMS)

        seur.combined@meta.data = seur.obj@meta.data[colnames(seur.combined),]
        #DefaultAssay(seur.combined) <- "integrated"
        #options(future.globals.maxSize = 50 * 1024 ^ 10)
        #plan("multicore", workers = 10)
        #seur.combined <- ScaleData(seur.combined, vars.to.regress = vars.to.regress, verbose = FALSE)
        seur.combined <- PCACLS(seur.combined, NPCS=NPCS, group.by=group.by)
        p1 <- DimPlot(object = seur.combined, reduction = "pca", pt.size = .1, group.by = group.by)
        p2 <- VlnPlot(object = seur.combined, features = "PC_1", group.by = group.by, pt.size = .1)
        ph <- grid.arrange(grobs = list(p1, p2), ncol = 2)
        ggsave(paste(headname, group.by, 'RPCASCT.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 9)
        return(seur.combined)
}
                                          
NomalScale <-function(scrnafilt, Batchmeth='Harmony', isnomal=TRUE, isscale=TRUE, batch.by=NULL, hvgbatch=TRUE,
                      batch.com=NULL, min.features = 200,
                      group.by='sampleid', nfeatures='Guess', ifeatures='Guess', NPCS=75, defassay='RNA',
                      dropMT=FALSE, dropRi=FALSE, dropHb=FALSE, ...){
    if(Batchmeth %in% c('CCA')){
         scrnafilt <- CCA(scrnafilt, group.by=group.by,NPCS=NPCS, batch.com=batch.com, ...)
         DefaultAssay(scrnafilt) ='integrated'
    }else if(Batchmeth %in% c('CCASCT')){
            scrnafilt <- CCASCT(scrnafilt, group.by=group.by, defassay=defassay, NPCS=NPCS, ...)
            DefaultAssay(scrnafilt) ='integrated'
    }else if(Batchmeth %in% c('RPCA')){
        if (defassay=='integrated'){
             scrnafilt <- RPCA(scrnafilt, group.by=group.by,NPCS=NPCS, 
                               batch.com=batch.com, 
                               dropMT=dropMT,
                               dropRi=dropRi, 
                               dropHb=dropHb,...)
             DefaultAssay(scrnafilt) ='integrated'
        }
    }else if(Batchmeth %in% c('RPCASCT')){
        if (defassay=='integrated'){
             scrnafilt <- RPCASCT(scrnafilt, group.by=group.by,NPCS=NPCS,
                                   dropMT=dropMT,
                                   dropRi=dropRi, 
                                   dropHb=dropHb,...)
             DefaultAssay(scrnafilt) ='integrated'
        }
    }else if(Batchmeth %in% c('STACAS')){
        if (defassay=='integrated'){
             scrnafilt <- STACAS(scrnafilt, group.by=group.by,NPCS=NPCS,...)
             #DefaultAssay(scrnafilt) ='integrated'
        }else if(defassay=='SCT'){
            print('updating!')
        }
    }else{
        if (defassay %in% c('RNA', 'Spatial')){
            if (isnomal){scrnafilt <- CNomal(scrnafilt)}
            if(hvgbatch){
                if ( is.null(batch.by)){ batch.by = group.by} 
                scrnafilt <- HGVSbatch(scrnafilt, batch.by=batch.by, 
                                        anchor.features = nfeatures, 
                                        features=ifeatures,
                                        isnormal=FALSE, 
                                        dropMT=dropMT,
                                        dropRi=dropRi, 
                                        dropHb=dropHb,
                                        batch.com=batch.com, 
                                        min.features = min.features,
                                        defassay=defassay)
            }else{
                scrnafilt <- HGVS(scrnafilt, nfeatures=nfeatures)
            }
            if(isscale){scrnafilt <-CScale(scrnafilt, ...)}
            scrnafilt <- PCACLS(scrnafilt, group.by=group.by, NPCS=NPCS)
            #DefaultAssay(scrnafilt) ='RNA'
        }else if(defassay=='SCT'){
            scrnafilt <- CSCT(scrnafilt, nfeatures=nfeatures, ...)
            scrnafilt <- PCACLS(scrnafilt, group.by=group.by, NPCS=NPCS)
            #DefaultAssay(scrnafilt) ='SCT'
        }
    }
    return(scrnafilt)
}

BatchProce <-function(scrnafilt, Batchmeth='Harmony', group.by='sampleid', defassay='RNA', NPCS = 75, DIMS=NULL, mnn.k=20, ...){
    if(Batchmeth %in% c('CCA', 'CCASCT')){
        DefaultAssay(scrnafilt) <-'integrated'
        Reduct   <- 'pca'
        clustpre <- 'integrated_snn_res.'
    }else if(Batchmeth %in% c('STACAS')){
        DefaultAssay(scrnafilt) <-'integrated'
        Reduct   <- 'pca'
        clustpre <- 'integrated_snn_res.'
    }else if(Batchmeth=='RPCA'){
        DefaultAssay(scrnafilt) <-'integrated'
        Reduct   <- 'pca'
        clustpre <- 'integrated_snn_res.'
    }else if(Batchmeth=='RPCASCT'){
        DefaultAssay(scrnafilt) <-'integrated'
        Reduct   <- 'pca'
        clustpre <- 'integrated_snn_res.'
    }else if(Batchmeth=='SeuratMnn'){
        DefaultAssay(scrnafilt) <- defassay
        Reduct <- 'mnn'
        scrnafilt <- SeuratMnn(scrnafilt, group.by=group.by, NPCS=NPCS)
    }else if(Batchmeth=='FastMnn'){
        DefaultAssay(scrnafilt) <-defassay
        Reduct <- 'pca'
        scrnafilt <- FastMnn(scrnafilt, group.by=group.by, DIMS=DIMS, mnn.k=mnn.k, ...)
    }else if(Batchmeth=='Harmony'){
        DefaultAssay(scrnafilt) <-defassay
        Reduct <- 'harmony'
        scrnafilt <- Harmony(scrnafilt, group.by=group.by, DIMS=DIMS, ...)
    }else if(Batchmeth %in% c('nobatch','nointegrate')){
        DefaultAssay(scrnafilt) <-defassay
        Reduct   <- 'pca'
        clustpre=paste0(defassay, '_snn_res.')
    }else{
        DefaultAssay(scrnafilt) <-defassay
        Reduct   <- 'pca'
        clustpre=paste0(defassay, '_snn_res.')
    }
    return (list(Data=scrnafilt, Reduct=Reduct, Batchmeth=Batchmeth))
}
                                          
CennAnno <-function(seur.obj, CM, clustpre='RNA_snn_res.', ORDER=NULL, ctpre='CellType.', dRes=2){
    seur.obj@meta.data[,paste0(ctpre,dRes)] <- factor(seur.obj@meta.data[,paste0(clustpre,dRes)], 
                                                 levels=CM$Cluster,
                                                 labels=CM$CellType)  
    if (!is.null(ORDER)){
        seur.obj@meta.data[,paste0(ctpre,dRes)] <- factor(seur.obj@meta.data[,paste0(ctpre,dRes)],
                                                             levels=ORDER)
    }
    seur.obj
}
    

FINDAMarks <- function(seur.obj, TN=20, ident='RNA_snn_res.0.7', defassay='RNA', disp.min = -2.5,disp.max = NULL, addcor=FALSE,
                       runfind=TRUE,  slot='scale.data', DIMS=10, width = 30, height = 60, ...){
    DefaultAssay(seur.obj) <- "RNA"
    Idents(object = seur.obj) <- ident
    if (runfind){
        seur.obj.markers <- FindAllMarkers(seur.obj, only.pos = TRUE, min.pct = 0.1, slot = 'data', logfc.threshold = 0.25, ...)
        write.csv(seur.obj.markers , paste(DIMS, ident, 'FindAllMarkers.csv', sep='.'), sep=',')
    }else{
        seur.obj.markers <- read.csv( paste(DIMS, ident, 'FindAllMarkers.csv', sep='.'), sep=',')
        rownames(seur.obj.markers) = seur.obj.markers$X
    }
    topN <- seur.obj.markers %>% group_by(cluster) %>% top_n(n = TN, wt = avg_log2FC)
    
    DefaultAssay(seur.obj) <- defassay
    pl.stop <- DoHeatmap(seur.obj, features = topN$gene, slot=slot, disp.min=disp.min, disp.max = disp.max) #+ NoLegend()
    if (slot=='data'){ pl.stop = pl.stop + scale_fill_viridis_c()}
    ggsave( paste(DIMS, ident, 'Top', TN , slot, 'DoHeatmap.pdf', sep='.'), 
            pl.stop, width = width, height = height, limitsize = FALSE) 
    TN = 10
    topN <- seur.obj.markers %>% group_by(cluster) %>% top_n(n = TN, wt = avg_log2FC)
    pa <- DotPlot(seur.obj, features = unique(topN$gene), group.by=ident) + 
            theme(text = element_text(size=10), axis.text.x = element_text(angle=90, hjust=1)) 
    ggsave( paste(DIMS, ident, 'Top', TN , 'DotPlot.scale.data.pdf', sep='.'), 
                       pa, width = 40, height = 20, limitsize = FALSE) 
    return(seur.obj.markers)
}

get_conserved <- function(seur.obj, 
                          idents.1=NULL,
                          ident.by='CellType', 
                          grouping.var='species',
                          only.pos=TRUE, ...){
   DefaultAssay(seur.obj) <- "RNA"
   Idents(seur.obj) <- ident.by

   Idents <- tryCatch({levels(droplevels(seur.obj@meta.data[,ident.by]))},
                        error = function(e){unique(seur.obj@meta.data[,ident.by])}) %>% as.character()
   if (!is.null(idents.1)) Idents <- idents.1

   conserved_markers <- lapply(Idents, function(cluster){
        iconserv <- FindConservedMarkers(seur.obj,
                           ident.1 = cluster,
                           grouping.var = grouping.var,
                           only.pos = only.pos, ...) %>%
                    rownames_to_column(var = "gene") %>%
                    cbind(ident.id = cluster, .) 
        print(c(cluster, dim(iconserv)))
        iconserv
   })

   conserved_markers <- data.frame(do.call(rbind, conserved_markers))
   conserved_markers
}


splitCheck <-function(seur.obj, group.by, nfeatures='Guess', ncomp=75,DIR=getwd(),
                      ncells=5000, NPC=50, Res=5, defassay = 'RNA', k.param=20,
                      vars.to.regress=c('nCount_RNA',"percent.mt")){
    OUSD <- paste(DIR, 'SplitCheck', sep='/')
    dir.create(OUSD, recursive=TRUE)
    setwd(OUSD)
    
    options(repr.plot.height = 10, repr.plot.width = 15)
    if (is.factor(seur.obj@meta.data[, group.by])){
        Groups <- levels(seur.obj@meta.data[, group.by])
    }else{
        Groups <- unique(seur.obj@meta.data[, group.by])
    }
    seur.obj.R = seur.obj
    for (i in Groups){
        print(i)
        seur.obj = subset( seur.obj.R,  !!rlang::sym(group.by)==i)
        if (defassay=='SCT'){
            nfeatures <- GuessNfeat(seur.obj, nfeatures)
            seur.obj <- SCTransform(seur.obj, vars.to.regress = vars.to.regress, 
                                method = "glmGamPoi",variable.features.n=nfeatures, ncells=5000, verbose = FALSE)
            
        }else{
            seur.obj <- NormalizeData(seur.obj)
            nfeatures <- GuessNfeat(seur.obj, nfeatures)
            seur.obj <- FindVariableFeatures(seur.obj, selection.method = "vst", nfeatures = nfeatures)
            options(future.globals.maxSize = 50 * 1024 ^ 10)
            plan("multicore", workers = 10)
            seur.obj <- ScaleData(seur.obj, vars.to.regress = vars.to.regress)
        }
        DefaultAssay(seur.obj) <- defassay
        seur.obj <- RunPCA(seur.obj, npcs = ncomp)
        seur.obj <- FindNeighbors(seur.obj, reduction = "pca", dims = 1:NPC, k.param=k.param)
        seur.obj <- FindClusters(seur.obj, resolution = Res, verbose =FALSE)
        seur.obj <- RunUMAP(seur.obj, dims = 1:NPC)

        groupclu <- paste0(defassay,'_snn_res.', Res)
        p1 <- DimPlot(seur.obj, reduction ='umap', group.by=groupclu, label.size = 4, label=TRUE, pt.size=0.1)
        p2 <- FeaturePlot(seur.obj,'nCount_RNA',order=TRUE, label.size = 4, label=TRUE, pt.size=0.1)
        p3 <- FeaturePlot(seur.obj,'percent.mt',order=TRUE, label.size = 4, label=TRUE, pt.size=0.1)
        p4 <- ElbowPlot(seur.obj, ndims=ncomp)+ labs(title=i)
        pall<-grid.arrange(grobs = list(p1,p2,p3,p4),ncol = 2) 
        ggsave( paste(i, NPC, Res, "feature.density.pdf", sep='_'), pall, width = 14, height = 14) 
    }
    setwd(DIR)
}

QuickCheck <- function(scrnaObj, features=NULL, INDIR=getwd(), DIMSs=seq(10,60,2), MapCol=NULL, OUTpre='seurat',
                       defassay='RNA', dRes = 2, 
                       #RES = c(seq(0.5, 4, 0.5), seq(5, 8, 1)), 
                       RES = seq(0.5, 4, 0.5),
                       barc='classic', 
                    QC.by=c('nCount_RNA','nFeature_RNA','Phase','phase','percent.mt','percent.ribo','percent.hb'),
                       clustpre = 'RNA_snn_res.', RUN3D = TRUE, runtsne=FALSE, group.by='SID', savedata=FALSE,
                       workers=5, ... ){
    options(repr.plot.height = 10, repr.plot.width = 22)
    for (DIMS in DIMSs){
        OUSW <- paste(INDIR, DIMS, sep='/')
        dir.create(OUSW, recursive=TRUE)
        print(OUSW)
        setwd(OUSW)

        DefaultAssay(scrnaObj$Data)= defassay
        seu.obj <- CLUSTERquick(scrnaObj$Data, DIMS=DIMS, RES=RES, 
                                reduction = scrnaObj$Reduct, runtsne=runtsne, run3d=RUN3D, ...)
        
        if (savedata){saveRDS(seu.obj, file = paste(OUTpre, DIMS, 'rds', sep='.'))}
        if (is.null(MapCol)){ MapCol = c(paste0(clustpre, dRes), group.by) }
        pCol <- PlotCluster(seu.obj, DIMS=DIMS, group.by=MapCol, ncols=2)
        plot(pCol)
        #PlotResolt(seu.obj, clustpre=clustpre, DIMS=DIMS)   
        #groupDimplot(seu.obj, group.by=group.by, outpre=DIMS)
        splitDimplot(seu.obj, group.by=group.by, outpre=DIMS)
    
        DimScatter(seu.obj, outpre=DIMS, group.by=QC.by)
        #DefaultAssay(seu.obj)=ifelse(defassay=='SCT', 'SCT','RNA')
        if (!is.null(features)){
            MarkerFPoint(seu.obj, features, reduction='umap', barc=barc, DIMS=DIMS, ncol=8, raster=FALSE)
        }
        if(RUN3D){
            Plot3dp(seu.obj, MapCol, outpre=paste0(DIMS,'.scater3d'),workers=workers )
        }        
    }
    seu.obj
}

TellDetail <- function(scrnaObj, features=NULL, otherF=NULL, INDIR=getwd(), DIMS=19, defassay='integrated',
                       dRes = 2, Batchmeth=NULL,
                       RES = c(seq(0, 4, 0.5), seq(5, 8, 1)), clustpre = 'integrated_snn_res.', ncol=10, 
                       RUN3D = TRUE,
                       raster.dpi = c(1024, 1024), runtsne=FALSE,
                       QC.by=c('nCount_RNA','nFeature_RNA','Phase','phase','percent.mt','percent.ribo','percent.hb'), MapCol=NULL,
                       group.by='SID', OUTpre='seurat', dodeg=FALSE,dores=FALSE, savedata=TRUE, workers=5, ... ){
    options(repr.plot.height = 10, repr.plot.width = 22)

    OUSW <- paste(INDIR, DIMS, sep='/')
    dir.create(OUSW, recursive=TRUE)
    print(OUSW)
    setwd(OUSW)

    DefaultAssay(scrnaObj$Data)= defassay
    seu.obj <- CLUSTER(scrnaObj$Data, DIMS=DIMS, RES=RES, reduction = scrnaObj$Reduct, run3d=RUN3D, runtsne=runtsne, ...)
    if (savedata){saveRDS(seu.obj, file = paste(OUTpre, DIMS, 'rds', sep='.'))}
    
    #PlotAllResCluster(seu.obj, RES=RES, DIMS=DIMS, clustpre=clustpre)
    if (is.null(MapCol)){ MapCol = c(paste0(clustpre, dRes), group.by) }

    pCol <- PlotCluster(seu.obj, DIMS=DIMS, group.by=MapCol, ncols=2)
    plot(pCol)
    scatter2dp(seu.obj, MapCol, outpre=paste0(DIMS,'.scater2d'))
    #groupDimplot(seu.obj, group.by=group.by, outpre=DIMS)
    splitDimplot(seu.obj, group.by=group.by, outpre=DIMS)
    DimScatter(seu.obj, outpre=DIMS, group.by=QC.by)
    
    if(dores){ PlotResolt(seu.obj, clustpre=clustpre, DIMS=DIMS) }

    DefaultAssay(seu.obj)=ifelse(defassay=='SCT', 'SCT','RNA')
    if(!is.null(features)){
        MarkerFPoint(seu.obj, features, reduction='umap', barc='classic', DIMS=DIMS, ncol=ncol, raster.dpi=raster.dpi )
        if (runtsne){MarkerFPoint(seu.obj, features, reduction='tsne', barc='classic', DIMS=DIMS, ncol=ncol, raster.dpi=raster.dpi )}
    }
    if(dodeg){FINDAMarks(seu.obj, DIMS=DIMS, ident = paste0(clustpre, dRes), defassay=defassay)}

    if(!is.null(otherF)){
        MarkerFPoint(seu.obj, otherF, reduction='umap', barc='others', DIMS=DIMS, ncol=ncol, raster.dpi=raster.dpi )
        if(runtsne){MarkerFPoint(seu.obj, otherF, reduction='tsne', barc='others', DIMS=DIMS, ncol=ncol, raster.dpi=raster.dpi ) }
    }
    if(RUN3D){
        Plot3dp(seu.obj, MapCol, outpre=paste0(DIMS,'.scater3d'), workers=workers)
    }
    #MarkerEachPoint(seu.obj, ALLMK, groupby='sampleid', reduction='umap', DIMS=DIMS)
    #MarkerEachPoint(seu.obj, ALLMK, groupby='sampleid', reduction='tsne', DIMS=DIMS)
    #MarkerVPoint(seu.obj, c("nCount_RNA", "nFeature_RNA","percent.mt", "percent.ribo"), RES, clustpre = clustpre, ncol=2, width =20, height=14)
    #MarkerVPoint(seu.obj, AllM, RES, clustpre = 'RNA_snn_res.') 
    return(seu.obj)
}   

saveH5AD <- function(seu.obj, outpre, assay='RNA', init.scale=FALSE){
    DefaultAssay(seu.obj) <- assay
    if (init.scale){
        seu.obj@assays$RNA@scale.data  <- new(Class = "matrix")
    }
    if('pca' %in% names(subobj@reductions)){
        seu.obj@reductions$pca@global <-TRUE
    }

    SeuratDisk::SaveH5Seurat(seu.obj, filename = paste0(outpre, ".h5Seurat"), overwrite = TRUE)
    SeuratDisk::Convert(paste0(outpre, ".h5Seurat"), dest = "h5ad", overwrite = TRUE)
    file.remove(paste0(outpre, ".h5Seurat"))
    write.csv(seu.obj@meta.data, paste0(outpre, ".meta.data.csv"))
}

saveH5AD.v0 <- function(seuobj){
    #saveRDS(seuobj, 'seurat.50.RPCA.rds')

    DefaultAssay(seuobj) <- 'RNA'

    #slot(object = seuobj[['RNA']], name = "scale.data") <- new(Class = "matrix")
    seuobj@assays$RNA@scale.data  <- new(Class = "matrix")
    seuobj@reductions$pca@global <-TRUE

    SeuratDisk::SaveH5Seurat(seuobj, filename = "seurat.50.RPCA.h5Seurat", overwrite = TRUE)
    SeuratDisk::Convert("seurat.50.RPCA.h5Seurat", dest = "h5ad", 
                        assay = "RNA", overwrite = TRUE)
    write.csv(seuobj@assays$integrated@data, 'seurat.50.RPCA.correct.data.csv')
    write.csv(seuobj@meta.data, 'seurat.50.RPCA.meta.data.csv')
    
}                                    

    
toupperdim <- function(seu.obj){
    for (i in names(seu.obj@reductions)){
        colnames(seu.obj@reductions[[i]]@cell.embeddings) <- toupper(colnames(seu.obj@reductions[[i]]@cell.embeddings))
    }
    return(seu.obj)
}

MulitiFoldChange <- function(seuobj, group.by, groups=NULL, slot = 'data', 
                             na.rm =FALSE, mean.fxn = NULL){
    if (na.rm == TRUE){
        mean.fxn <- function(x, pseudocount.use=1, base=2){
            log(x = rowMeans(x = expm1(x = x), na.rm =TRUE) + pseudocount.use, base = base)
        }

        mean.fxn <- function(x, pseudocount.use=1, base=2){
            log(rowSums(expm1(x), na.rm =TRUE)/rowSums(x>0) + pseudocount.use, base = base)
        }
    }
    if (is.null(groups)){
        groups <- as.character(unique(seuobj@meta.data[, group.by]))
    }
    BranchFC <- lapply(groups, function(igroup){
        seuobj$fcgroup <- ifelse(seuobj@meta.data[, group.by]==igroup, igroup, 'others')
        seuobj$fcgroup <- factor(seuobj$fcgroup, levels=c(igroup, 'others'))
        gFC <- Seurat::FoldChange(seuobj, ident.1 = igroup, ident.2 = 'others', 
                                  group.by='fcgroup', slot = slot,
                                  mean.fxn=mean.fxn)
        iFC<- gFC$avg_log2FC
        names(iFC) <- rownames(gFC)
        iFC
    })
    BranchFC <- data.frame(do.call(cbind, BranchFC))
    colnames(BranchFC) <- groups
    BranchFC$group <- colnames(BranchFC)[apply(BranchFC,1,which.max)]
    return(BranchFC)
}

#######PLOT###
PlotCluster1 <- function(seur.obj, DIMS=50, group.by=c('RNA_snn_res.2','sampleid'), 
                        raster=NULL, size=10, error=1, pt.size=NULL, header='cheack',
                        ncols=NULL){
    #'integrated_snn_res'
    plist <- list()
    nrow  <- 0
    ncol  <- length(group.by)
    if ('umap' %in% names(seur.obj@reductions)){
        pu <- lapply(group.by, function(x) DimPlot(seur.obj, reduction ='umap',raster =raster,
                                                   group.by=x, label.size = 4, label=TRUE, pt.size=pt.size))
        plist = c(plist, pu)
        nrow  = nrow+1
    }
    if ('tsne' %in% names(seur.obj@reductions)){
        pt <- lapply(group.by, function(x) DimPlot(seur.obj, reduction ='tsne',raster =raster,
                                                   group.by=x, label.size = 4, label=TRUE, pt.size=pt.size))
        plist = c(plist, pt)
        nrow  = nrow+1
    }
    
    if (nrow>0){
        ncols <- ifelse(is.null(ncols), ncol, ncols )
        CR <- colrows(length(plist), ncols)
        ncol <- CR[2]
        nrow <- CR[1]
        
        pall<-grid.arrange(grobs = plist, ncol=ncol)
        ggsave( paste(c(header, DIMS, group.by, "feature.density.pdf"), collapse='_'), pall, 
                 width = ncol*(size+error), height = nrow*size, limitsize = FALSE)
        return (pall)
    } 
}

PlotCluster <- function(seur.obj, DIMS=50, group.by=c('RNA_snn_res.2','sampleid'), 
                        raster=NULL, size=10, error=3.5, pt.size=NULL, header='cheack',
                        ncols=2, ...){

    iCR <- colrows(length(group.by), ncols=ncols, soft=TRUE)
    irows <- iCR[1]
    icols <- iCR[2]
    
    plist <- list()
    nrow  <- 0
    if ('umap' %in% names(seur.obj@reductions)){
        pu <- DimPlot(seur.obj, group.by = group.by, ncol=icols,  reduction ='umap',
                      raster =raster, label.size = 4, label=TRUE, pt.size=pt.size, ...)
        nrow  <- nrow + 1
        plist[[nrow]] <- pu
    }
    if ('tsne' %in% names(seur.obj@reductions)){
        pt <- DimPlot(seur.obj, group.by = group.by, ncol=icols,  reduction ='tsne',
                      raster =raster, label.size = 4, label=TRUE, pt.size=pt.size, ...)
        nrow  <- nrow + 1
        plist[[nrow]] <- pt
    }
    pall <- CombinePlots(plots =plist, ncol=1)             
    if (nrow>0){
        nrows <- nrow*irows
        ncols <- icols
        ggsave( paste(c(header, DIMS, group.by, "feature.density.pdf"), collapse='_'), pall, 
                 width = ncols*(size+error), height = nrows*size, limitsize = FALSE)
    }
    pall
    return (pall)
}
                     
PlotAllResCluster <-function(seur.obj, RES, ...){
    for (i in RES){
        PlotCluster(seur.obj, Res=i, ...) 
    }
}

PlotResolt <- function(seur.obj, clustpre='RNA_snn_res.', DIMS=50, width=25, height=20, dostratum=FALSE){
    library(clustree)
    Q <- clustree(seur.obj@meta.data, prefix = clustpre)
    ggsave(paste0(DIMS,'.PlotResolt.clustree.pdf'), Q, width = width, height = height, limitsize = FALSE)
    
    if (dostratum){
        library(ggalluvial)
        library(tidyverse)
        Res = c(0.2, 0.4, 0.6, 0.8, 1, 1.4, 1.8, 2.2)
        P <- ggplot(seur.obj@meta.data, 
                aes_string( axis1 = paste0(clustpre, '0.2'), axis2 = paste0(clustpre, '0.4'), 
                            axis3 = paste0(clustpre, '0.6'), axis4 = paste0(clustpre, '0.8'), 
                            axis5 = paste0(clustpre, '1'), axis6 = paste0(clustpre, '1.4'),   
                            axis7 = paste0(clustpre, '1.8'), axis8 = paste0(clustpre, '2.2'))) + 
                scale_x_discrete(limits=c(paste0(clustpre, Res)), expand=c(0.01, 0.05))+
                geom_alluvium(aes_string(fill=paste0(clustpre, '2.2'))) +
                geom_stratum() +
                geom_text(stat="stratum", infer.label=TRUE)+
                theme(axis.text.x=element_text(angle=90, hjust=1))+
                ggtitle('cell number in each cluster')
        ggsave(paste0(DIMS,'.PlotResolt.alluvium.png'), P, width = 10, height = 18)
    }
}

MarkerEachPoint <- function(seur.obj, Markdf, group.by='sampleid', reduction='umap', DIMS=15){
    for (i in 1:nrow(Markdf)){
        imarkers <- unlist( strsplit(Markdf[i, 'markergene'], split=',') )
        icelltype<- Markdf[i, 'celltype']
        try({
            P <- FeaturePlot(seur.obj, imarkers, split.by=group.by,reduction=reduction, ncol=7,order=TRUE) + coord_fixed(ratio=1)
            ggsave(paste(DIMS, group.by, reduction, 'FeaturePlot', icelltype, 'pdf', sep='.'), 
                   P, width=21, height = 3*length(imarkers)) 
            ggsave(paste(DIMS, group.by, reduction, 'FeaturePlot', icelltype, 'png', sep='.'),
                   P, width=21, height = 3*length(imarkers)) 
        })
    }
}

MarkerVPoint <- function(seur.obj, features, Res, clustpre='integrated_snn_res.', barc='gene', ncol=6, width =56, height=36, ...){
    for (i in Res){
        pv <- VlnPlot(seur.obj, features = features, group.by=paste0(clustpre, i), ncol=ncol, ...)
        ggsave( paste(DIMS, i, barc, 'VlnPlot.png', sep='.'), pv , width = width, height = height, limitsize = FALSE)
    }
    pv
}

MarkerFPoint <- function(seur.obj, features, reduction='umap', barc='gene', 
                         #cols=c("lightgrey", "blue", "blue2",'red','yellow'),
                         cols=c("lightgrey", 'yellow', 'red','darkred'),
                         order=TRUE, DIMS=15, ncol=6, size=5, ...){
    W = ncol*size
    H = ceiling(length(features)/ncol)*size

    pf <- FeaturePlot(seur.obj, features, reduction=reduction, ncol=ncol, 
                      order=order,  cols=cols, ...)
    ggsave(paste(DIMS, reduction, barc, 'FeaturePlot.png', sep='.'), pf ,
           width = W, height = H, limitsize = FALSE) 
    pf
}

scatter3dp <-function(scrna.obj, group.by='sampleid', reduct.ty='umap', outpre='scater3d', 
                      slot='data',
                      colors=NULL, fontsize=10, show=TRUE, size=1, width=1500, height=1500, ...){
    library(plotly)
    redim <- scrna.obj@reductions[[reduct.ty]]@cell.embeddings
    cn <- group.by

    p3d <- Seurat::FetchData(object = scrna.obj, vars = c(cn), slot=slot, ...)
    l <- list(font = list(family = "sans-serif", color = "#000", fontsize = fontsize),
              itemsizing='constant', 
              borderwidth = 0,
              autosize = F, 
              width = width, 
              height = height)
    
    fig <- plot_ly(x = redim[,1], 
                   y = redim[,2], 
                   z = redim[,3],
                   color = p3d[,cn],
                   colors=colors,
                   type = "scatter3d",
                   mode = "markers", 
                   marker = list(size = size, #color = ,
                                 line = list(color = 'black', width = 0))
                  ) %>% layout(legend = l)
    if (!is.null(outpre)){
        htmlwidgets::saveWidget(fig, file=paste(outpre, reduct.ty, group.by, 'html', sep='.' ))
    }
    if (show){ return(fig) }
}
    
scatter2dp <-function(scrna.obj, group.by='sampleid', reduct.ty='UMAP', outpre='scater2d',
                      colors = NULL,
                      size=1.5, nrows=1, save=TRUE){
    library(plotly)
    xn <- paste(reduct.ty, 1, sep="_")
    yn <- paste(reduct.ty, 2, sep="_")
    cn <- group.by

    p3d <- Seurat::FetchData(object = scrna.obj, vars = c(xn, yn, cn))

    l <- list(font = list(color = "#000", size = 10), #family = "sans-serif",
              itemsizing='constant', 
              borderwidth = 0)
    
    figs <- lapply(group.by, function(x){
                plot_ly(x = p3d[,xn], 
                        y = p3d[,yn], 
                       color = p3d[,x],
                       mode = "markers", 
                       colors = colors,
                       marker = list(size = size, width=0)) %>% 
                layout(legend = l, autosize = T, template = "simple_white")
    })
    fig <- plotly::subplot(figs, nrows=nrows)
    if (save){
        htmlwidgets::saveWidget(fig, file=paste(c(outpre, reduct.ty, group.by, 'html'), collapse='.'))}
    fig   
}
                     
DimScatter <-function(scrna.obj, group.by=c('nCount_RNA','nFeature_RNA','Phase','phase','percent.mt','percent.ribo','percent.hb'),
                      outform='png', 
                      reduct.ty='UMAP', outpre='dim.scater', size=0.1, ncol=3, scale=5, widtherror = 0.2, out=NULL){
    group.by <- intersect(colnames(scrna.obj@meta.data), group.by)
    if (length(group.by)<=0){
        return(NULL)
    }
    xn <- paste(reduct.ty, 1, sep="_")
    yn <- paste(reduct.ty, 2, sep="_")
    cn <- group.by
    p3d <- Seurat::FetchData(object = scrna.obj, vars = c(xn, yn, cn))
    ggs <- lapply(group.by, function(x){
        P <- ggplot(p3d, aes_string(x=xn, y=yn, color=x)) +
                      geom_point(size=size) + 
                      ggtitle(x) + theme_bw()
        if (is.numeric(p3d[,x])){
            P + scale_colour_viridis_c()
        }else{
            P + guides(color = guide_legend(override.aes = list(size = 3)))
        }
    })
    A = length(group.by)
    C = ifelse(A - floor(A/ncol)*ncol ==1, ncol + 1, min(ncol, A))
    R = ceiling(A/C)
    plots <- grid.arrange( grobs =ggs, ncol = C, limitsize = FALSE)
    options(repr.plot.height = R*scale, repr.plot.width = C*(scale+widtherror))
    if (is.null(out)){
       ggsave(paste(outpre, reduct.ty, 'QCfeature.dim.scater.', outform, sep='.' ), plots,
              width = C*(scale+widtherror), height = R*scale)
    }else if(is.character(out)){
       ggsave(out, plots, width = C*scale, height = R*scale)
    }
    plots
}

ggscatter <-function(scrna.obj, group.by='sampleid', reduct.ty='umap', 
                     slot='data', outpre='scater2d',size=0.1, nrows=1, save=TRUE, ...){
    xn <- paste(reduct.ty, 1, sep="_")
    yn <- paste(reduct.ty, 2, sep="_")
    cn <- group.by
    p3d <- Seurat::FetchData(object = scrna.obj, vars = c(xn, yn, cn), slot=slot, ...)
    ggs <- lapply(group.by, function(x){
        ggplot(p3d, aes_string(x=xn, y=yn, color=x)) +
              geom_point(size=size) + 
              theme_bw() %>% ggplotly()
    })
    fig <- plotly::subplot(ggs, nrows=nrows) #, widths=c(0.5,0.5),heights=c(0.5)
    if (save){htmlwidgets::saveWidget(fig, file=paste(c(outpre, reduct.ty, group.by, 'ggplot.html'), collapse='.'))}
    fig
}

Plot3dp<-function(seur.obj, group.bys, basis=NULL, show=FALSE, workers=5, ...){

    if (workers>1){
        library("future.apply")
        options(future.globals.maxSize = 100 * 1024 ^ 3)
        plan("multicore", workers = workers)
        gg<- future_lapply(group.bys, function(group.by){
            try({
            if (! is.null(basis)){
                scatter3dp(seur.obj, group.by=group.by, reduct.ty=basis, ...)}
            else{
               if ('umap3d' %in% names(seur.obj@reductions)){ scatter3dp(seur.obj, group.by=group.by, reduct.ty='umap3d', ...) }
               if ('tsne3d' %in% names(seur.obj@reductions)){ scatter3dp(seur.obj, group.by=group.by, reduct.ty='tsne3d', ...) }
           }
           })
        })
    }else{
        gg<- lapply(group.bys, function(group.by){
            if (! is.null(basis)){
                scatter3dp(seur.obj, group.by=group.by, reduct.ty=basis, ...)}
            else{
               if ('umap3d' %in% names(seur.obj@reductions)){ scatter3dp(seur.obj, group.by=group.by, reduct.ty='umap3d', ...) }
               if ('tsne3d' %in% names(seur.obj@reductions)){ scatter3dp(seur.obj, group.by=group.by, reduct.ty='tsne3d', ...) }
           }
        })
    }

    if (show){return(gg)}                   
}

PlotCellType <- function(seur.obj, DIMS=50, clustpre='RNA_snn_res.', Res=0.7, reduction='pca', header='cheack'){
    #'integrated_snn_res'
    groupclu <- paste0(clustpre,Res)
    cellclu <- paste0('CellType.',Res)
    p0 <- DimPlot(seur.obj, reduction ='umap', group.by=groupclu, label.size = 6, label=T,raster =FALSE, pt.size=0.01)
    p1 <- DimPlot(seur.obj, reduction ='umap', group.by=cellclu, label.size = 6, label=F,raster =FALSE, pt.size=0.01)
    p2 <- DimPlot(seur.obj, reduction ='umap', group.by='AGE',label.size = 6, label=F,raster =FALSE, pt.size=0.01)
    p3 <- DimPlot(seur.obj, reduction ='umap', group.by='GW',label.size = 6, label=F,raster =FALSE, pt.size=0.01)
    
    p6 <- DimPlot(seur.obj, reduction ='tsne', group.by=groupclu, label.size = 6, label=T,raster =FALSE, pt.size=0.01)
    p7 <- DimPlot(seur.obj, reduction ='tsne', group.by=cellclu, label.size = 6, label=F,raster =FALSE, pt.size=0.01)
    p8 <- DimPlot(seur.obj, reduction ='tsne', group.by='AGE', label.size = 6, label=F,raster =FALSE, pt.size=0.01)
    p9 <- DimPlot(seur.obj, reduction ='tsne', group.by='GW', label.size = 6, label=F,raster =FALSE, pt.size=0.01)
    
    pall<-grid.arrange(grobs = list(p0,p1,p2,p3,p6,p7,p8,p9),ncol = 4)
    ggsave( paste(header, DIMS, Res, reduction, "feature.density.pdf", sep='_'), pall, width = 36, height = 18) 
    pall
}
                     
plotumaptsne <-function(seur.obj, grouy.by, colormap=NULL, header='umap.tsne'){
    p1 <- DimPlot(seur.obj, reduction ='umap', group.by=grouy.by, label.size = 6, label=F,raster =FALSE, cols=colormap, pt.size=0.01)
    p2 <- DimPlot(seur.obj, reduction ='tsne', group.by=grouy.by, label.size = 6, label=F,raster =FALSE, cols=colormap, pt.size=0.01)
    pall<-grid.arrange(grobs = list(p1,p2), ncol = 2)
    ggsave( paste(header, grouy.by, "feature.density.pdf", sep='_'), pall, width = 14, height = 7) 
    pall
}

groupDimplot <- function(seur.obj, group.by='sampleid', out=NULL, reduction='umap', outpre=NULL, 
                         size=NULL, ncol=4, scale=5, widtherror = 0.2, outform='png', 
                         raster=FALSE, control=FALSE, ...){
    Idents(object = seur.obj) <- group.by
    Nam <- levels(seur.obj)
    Col <- scales::hue_pal()(length(Nam))

    plots <- lapply(
      X = seq(length(Nam)),
      FUN = function(x) {
        return(DimPlot(
          seur.obj,
          order =TRUE,
          reduction = reduction,
          pt.size = size,
          cols.highlight = Col[x],
          sizes.highlight = size,
          raster =raster,
          cells.highlight = CellsByIdentities(seur.obj, idents = Nam[x]) 
          #+theme(legend.position='hidden')
        ))
      }
    )

    Ctrl = DimPlot(seur.obj, order =FALSE, reduction = reduction, pt.size = size, raster=FALSE, 
                   label=F,label.size = 6, group.by=group.by)
    legend <- ggpubr::get_legend(Ctrl)
    
    if (control){
        Ctrl  = Ctrl + theme(legend.position='hidden')
        plots = c(list(Ctrl),plots)
    }
    
    A = length(Nam) + control
    C = ifelse(A - floor(A/ncol)*ncol ==1, ncol + 1, min(ncol, A))
    R = ceiling(A/C)

    plots <- grid.arrange( grobs =plots, ncol = C, limitsize = FALSE)
    if (is.null(out)){
       ggsave(paste(outpre, group.by, reduction, 'split.DimPlot', outform,  sep='.' ), plots,
              width = C*(scale+widtherror), height = R*scale)
    }else if(is.character(out)){
       ggsave(out, plots, width = C*scale, height = R*scale)
    }
    plots
}

splitDimplot <- function(seur.obj, group.by=NULL, split.by=NULL, reduction='umap', outpre=NULL, ...){
    MTRX = seur.obj@meta.data
    EMB = seur.obj[[reduction]]@cell.embeddings
    X = paste0(toupper(reduction),'_',1)
    Y = paste0(toupper(reduction),'_',2)
    colnames(EMB) = c(X,Y)
    MTRX[,X] = EMB[,X]
    MTRX[,Y] = EMB[,Y]

    header = ifelse(is.null(outpre), reduction, paste(outpre, reduction, sep='.'))
    splitMtxDimplot(MTRX, x=X, y=Y, group.by=group.by, split.by=split.by, header=header, ...)
}

get_only_legend <- function(plot) {
  plot_table <- ggplot_gtable(ggplot_build(plot))
  legend_plot <- which(sapply(plot_table$grobs, function(x) x$name) == "guide-box")
  legend <- plot_table$grobs[[legend_plot]]
  return(legend)
}

splitMtxDimplot <- function(DF, x='umap1', y='umap2', group.by=NULL, split.by=NULL, size=0.01,
                            bgsize =NULL, soft=TRUE, gncol=1,
                            header ='umap', outform='png', out=NULL, colormap=NULL, colorstack=TRUE, 
                            limitsize=FALSE,
                            rest='others', bgcolor='#DDDDDD', ncol=5, scale=5, control=TRUE){
    if(!is.factor(DF[,group.by]) & !is.null(group.by)) DF[,group.by]<- factor(DF[,group.by])
    if(!is.factor(DF[,split.by]) & !is.null(split.by)) DF[,split.by]<- factor(DF[,split.by])
    if (is.null(bgsize)) bgsize <- size
    Nam <- levels(DF[,group.by])
    if (is.null(colormap)){
        Col <- scales::hue_pal()(ifelse(!is.null(split.by), length(levels(DF[,split.by])), length(Nam)))
    }else{
        Col <- colormap
    }

    if (!is.null(split.by) & colorstack){
       plots <- lapply(seq(length(Nam)),
        function(i){
        ggplot(DF %>% filter(!!rlang::sym(group.by) != Nam[i])) +
              geom_point(aes_string(x = x, y = y), size=bgsize, color=bgcolor) +
              geom_point(data=DF %>% filter(!!rlang::sym(group.by) == Nam[i]), 
                               aes_string(x = x, y = y, color = split.by), size = size)+
              ggtitle(Nam[i]) +
              guides(color = guide_legend(override.aes = list(size = 3)), fill=guide_legend(ncol=gncol)) +
              scale_color_manual(values=Col)+
              theme(legend.position = "none", legend.text=element_text(size=15),
                    panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                    panel.background = element_rect(fill = "transparent", colour = "black"))
         })
    }else if(!is.null(split.by) & !colorstack){
       plots<- list()
       n <- 1
       for (i in seq(length(Nam))){
           r <- n
           e <- length(unique(DF[DF[,group.by]==Nam[i], split.by]))
           n <- r + e
           plots[[i]] <- ggplot(DF %>% filter(!!rlang::sym(group.by) != Nam[i])) +
                              geom_point(aes_string(x = x, y = y), size=bgsize, shape=20, color=bgcolor) +
                              geom_point(data=DF %>% filter(!!rlang::sym(group.by) == Nam[i]), 
                                               aes_string(x = x, y = y, color = split.by), shape=20,size = size)+
                              ggtitle(Nam[i]) +
                              guides(color = guide_legend(override.aes = list(size = 3)), fill=guide_legend(ncol=gncol)) +
                              scale_color_manual( values=Col[r:(n-1)])+
                              #scale_size_manual(values =c(1, 5))+
                              #scale_color_manual(breaks = c(Nam[i], rest),  values=c(Col[i],'grey'))+
                              theme(legend.position = "none", legend.text=element_text(size=15),
                                    panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                    panel.background = element_rect(fill = "transparent", colour = "black"))
       }
    }else{
       plots <- lapply(seq(length(Nam)),
        function(i){
        ggplot(DF %>% filter(!!rlang::sym(group.by) != Nam[i])) +
              geom_point(aes_string(x = x, y = y), size=bgsize, color=bgcolor) +
              geom_point(data=DF %>% filter(!!rlang::sym(group.by) == Nam[i]), 
                               aes_string(x = x, y = y), size = size, color = Col[i])+
              ggtitle(Nam[i]) +
              guides(color = guide_legend(override.aes = list(size = 3)), fill=guide_legend(ncol=gncol)) +
              theme(legend.position = "none", legend.text=element_text(size=15),
                                    panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                    panel.background = element_rect(fill = "transparent", colour = "black"))
         })  
    }
    
    gGRO = ifelse(!is.null(split.by), split.by, group.by)
    Ctrl = ggplot(DF, aes_string(x=x,y=y, color=gGRO)) + 
                        geom_point(size=size)+
                        ggtitle(gGRO) + 
                        guides(color = guide_legend(override.aes = list(size = 3), ncol=gncol), fill=guide_legend(ncol=gncol)) +
                        scale_color_manual( values=Col)
    legend <- ggpubr::get_legend(Ctrl)
    #legend <- get_only_legend(Ctrl)
    
    if (control){
        Ctrl  = Ctrl + theme(legend.position='hidden')
        plots = c(list(Ctrl),plots)
    }
    
    A = length(Nam) + control + 1
    R = colrows(A, ncols=ncol, soft=soft)[1]
    C = colrows(A, ncols=ncol, soft=soft)[2]

    #plots <- grid.arrange(grobs =plots, ncol = C, limitsize = FALSE)
    #plots <- grid.arrange(plots, legend,
    #                      ncol = 2, widths = c(C*scale, scale), limitsize = FALSE)

    plots <- c(plots, list(legend))
    #plots <- append(plots, list(legend))
    plots <- grid.arrange( grobs =plots, ncol = C, limitsize = FALSE)
    #plots <- ggpubr::ggarrange( plotlist =plots, ncol = C, limitsize = FALSE)
    
    if (is.null(out)){
       ggsave(paste(header, group.by, 'split', split.by, 'mtrx.DimPlot', outform, sep='.'), plots,
              limitsize=limitsize,
              width = (C)*scale, height = R*scale)
    }else if(is.character(out)){
       ggsave(out, plots, width = (C)*scale, height = R*scale, limitsize=limitsize)
    }
    return(plots)
}

Plotflow <-function(seur.obj, by1, by2, scaleby=NULL, order1 =NULL, order2= NULL, 
                    angle=45, fontsize=12, stat='counts', ncol=2, show_label=FALSE,
                    show_flow=TRUE,
                    label_size=8){
    library(ggalluvial)
    library(dplyr)
    library(tidyverse)
    
    GG <- seur.obj@meta.data
    if (!is.null(order1)){
        GG[, by1] <- factor(GG[, by1], levels=order1)
    }
    if (!is.null(order2)){
        GG[, by2] <- factor(GG[, by2], levels=order2)
    }
    
    if (is.null(scaleby)){ 
        GG$scaleby =GG[, by1]
    }else{
        GG$scaleby =GG[, scaleby]
    }
    scaleby = 'scaleby'
    GG <-GG[,c(scaleby, by1, by2)] %>%
            group_by_(scaleby, by1, by2) %>% 
            summarise(counts_e1= n()) %>%
            ungroup(!!rlang::sym(by2)) %>%
            mutate(counts_a1 = sum(counts_e1),
                   freq_e1 = counts_e1/counts_a1) %>% 
            data.frame() %>%
            arrange_(by1, by2) %>%
            group_by_(by1, by2) %>% 
            mutate(counts_e2 = sum(counts_e1),
                  ratio_e2 = sum(freq_e1)) %>%
            ungroup(!!rlang::sym(by2)) %>%
            mutate(counts_a2 = sum(counts_e1),
                   freq_e2 = counts_e2/counts_a2,
                   freq_e3 = ratio_e2/sum(freq_e1)) %>%
            data.frame() %>%
            dplyr::select(!!rlang::sym(by1), !!rlang::sym(by2), counts_e2, freq_e2, freq_e3) %>%
            distinct_all() %>%
            rename(freq_ = freq_e2, freq =freq_e3, counts=counts_e2) 
    print(GG)
    
    if (show_flow){
        KK <- GG %>% ggplot( aes_string(y = stat, x = by1, stratum = by2, alluvium = by2, fill = by2))+ 
                    geom_lode() + 
                    geom_flow(width = .4, color = "darkgrey", alpha=0.7) +
                    geom_stratum(width = .4,)
    }else{
        KK <- GG %>% ggplot( aes_string(y = stat, x = by1, stratum = by2, alluvium = by2, fill = by2))+ 
            geom_lode() + 
            geom_stratum(width = .4,)
    }

    if (show_label){
        KK <- KK + geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = label_size)
    }
                
    KK <- KK +  guides(fill=guide_legend(ncol=ncol)) + 
                 theme(text = element_text(size=fontsize),
                       axis.text.x = element_text(angle=angle, hjust=1))
    KK
}

Plotflow.df <-function(GGdf, by1, by2, scaleby=NULL, order1 =NULL, order2= NULL, 
                    angle=45, fontsize=12, stat='counts'){
    library(ggalluvial)
    library(dplyr)
    library(tidyverse)
    
    GG <- GGdf
    if (!is.null(order1)){
        GG[, by1] <- factor(GG[, by1], levels=order1)
    }
    if (!is.null(order2)){
        GG[, by2] <- factor(GG[, by2], levels=order2)
    }
    if (is.null(scaleby)){ 
        GG$scaleby =GG[, by1]
        scaleby = 'scaleby'
    }
    GG <- GG[,c(scaleby, by1, by2)] %>%
            group_by_(scaleby, by1, by2) %>% 
            summarise(counts_e1= n()) %>%
            ungroup(!!rlang::sym(by2)) %>%
            mutate(counts_a1 = sum(counts_e1),
                   freq_e1 = counts_e1/counts_a1) %>% 
            data.frame() %>%
            arrange_(by1, by2) %>%
            group_by_(by1, by2) %>% 
            mutate(counts_e2 = sum(counts_e1),
                  ratio_e2 = sum(freq_e1)) %>%
            ungroup(!!rlang::sym(by2)) %>%
            mutate(counts_a2 = sum(counts_e1),
                   freq_e2 = counts_e2/counts_a2,
                   freq_e3 = ratio_e2/sum(freq_e1)) %>%
            data.frame() %>%
            dplyr::select(!!rlang::sym(by1), !!rlang::sym(by2), counts_e2, freq_e2, freq_e3) %>%
            distinct_all() %>%
            rename(freq_ = freq_e2, freq =freq_e3, counts=counts_e2) 
    print(GG)
    GG %>% ggplot( aes_string(y = stat, x = by1, stratum = by2, alluvium = by2, fill = by2))+ 
                geom_lode() + 
                geom_flow(width = .4, color = "darkgrey", alpha=0.7) +
                geom_stratum(width = .4,) +
                geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 10) +
                theme(text = element_text(size=fontsize),
                      axis.text.x = element_text(angle=angle, hjust=1))
}
                              
Plotflow0 <-function(seur.obj, by1, by2, order1 =NULL, order2= NULL, angle=45, fontsize=12, stat='counts'){
    library(ggalluvial)
    library(tidyverse)
    GG <- seur.obj@meta.data
    if (!is.null(order1)){
        GG[, by1] <- factor(GG[, by1], levels=order1)
    }
    if (!is.null(order2)){
        GG[, by2] <- factor(GG[, by2], levels=order2)
    }
    GG <- GG %>%
                group_by_(by1, by2) %>%
                summarise(counts = n()) %>%
                mutate(freq = counts / sum(counts))
    GG %>%
            ggplot( aes_string(y = stat, x = by1,stratum = by2, alluvium = by2, fill = by2))+ 
            geom_lode() + 
            geom_flow(width = .4, color = "darkgrey", alpha=0.7) +
            geom_stratum(width = .4,) +
            theme(text = element_text(size=fontsize),
                  axis.text.x = element_text(angle=angle, hjust=1))
}

FeaturePlots <-function(seuobj, MARKERS, ncol=4, pt.size=0.1, legend.position = c(0.975, 0.5),
                        raster=FALSE, raster.dpi=c(1024, 1024), 
                       out=NULL, limitsize=FALSE, scale=6, error=0.1, dpi=300, soft=TRUE, ...){
    DefaultAssay(seuobj) <- 'RNA'
    kMARKERS <- intersect(MARKERS, rownames(seuobj@assays$RNA@data))
    A = length(kMARKERS)
    R = colrows(A, ncols=ncol, soft=soft)[1]
    C = colrows(A, ncols=ncol, soft=soft)[2]
    
    PP <- FeaturePlot(seuobj, features = kMARKERS, reduction='umap', ncol=C, 
                      #cols=c("lightgrey", 'yellow', 'red', 'darkred'),
                      raster=raster,
                      pt.size =pt.size,
                      raster.dpi=raster.dpi,
                      order=TRUE, ...) & 
                    NoAxes() & 
                    scale_colour_gradientn(colours = c("lightgrey", 'yellow', 'red', 'darkred')) &
                    theme(legend.position = legend.position,
                          panel.spacing = unit(x = 0, units = "lines"),
                          strip.background = element_blank())

    
    if (!is.null(out)){
       ggsave(out, PP, limitsize=limitsize, width = (C)*(scale+error), height = R*scale, dpi=dpi)
    }
    return(PP)
    #ggsave( 'umap.Scanorama.60.select.gene.pdf', PP, width = 20, height = 16, dpi=800) 
}
                      
COLORS <- c('#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', 
            '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8', 
            '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
            '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31')               
`%||%` <- function (x, y){
    if (is_null(x)) 
        y
    else x
}
sort(sapply(ls(), function(x) object.size(get(x))))