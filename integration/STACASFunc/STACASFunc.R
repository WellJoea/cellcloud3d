library(Seurat)
source("/home/gpfs/home/wulab17/JupyterCode/STACASFunc/integrateSTACAS.R")

STACAS <- function(seur.obj, group.by='sampleid', features='Guess', anchor.features='Guess', isnormal=TRUE,
                     normalization.method='LogNormalize', IDIMS=50, iNPCS=50, NPCS=100, headname='batch',
                     k.anchor=5,  k.filter = 200, k.score = 30, k.weight = 100, max.features=200,
                     dropMT=FALSE, dropRi=FALSE, dropHb=FALSE,
                     vars.to.regress=c("nCount_RNA","percent.mt"), ...){

        seur.list <- SplitObject(seur.obj, split.by = group.by)
        message("Normalize and Find Variable Features in each group...")
        seur.list <- lapply(X = seur.list, FUN = function(x) {
                if(isnormal){x <- NormalizeData(x)}
                features <- GuessNfeat(x, features)
                x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = features)
                x <- dropHGVSfeature(x, dropMT=dropMT, dropRi=dropRi, dropHb=dropHb)
                x
         })

        anchor.features <- GuessNfeat(seur.obj, anchor.features)
        features <- SelectIntegrationFeatures(object.list = seur.list, nfeatures = anchor.features)
        
        message("ScaleData and Run PCA in each group...")
        seur.list <- lapply(X = seur.list, FUN = function(x) {
            x <- ScaleData(x, features = features, verbose = FALSE)
            x <- RunPCA(x, features = features, npcs = iNPCS, nfeatures.print = 1, verbose = FALSE)
            x
        })  

        message("Find Integration Anchors...")
        seur.anchors <- FindIntegrationAnchors.STACAS(seur.list, 
                                                      normalization.method=normalization.method,
                                                      anchor.features = features,
                                                      k.filter = k.filter, 
                                                      k.score=k.score, 
                                                      k.anchor = k.anchor,
                                                      max.features=max.features,
                                                      reduction = "rpca", 
                                                      dims= 1:IDIMS,
                                                      verbose=TRUE, ...)
    
        names <- names(seur.list)
        names(seur.anchors@object.list) <- names

        anchors.tabel <- table(seur.anchors@anchors[,c("dataset1","dataset2")])
        rownames(anchors.tabel) <- names 
        colnames(anchors.tabel) <- names                                  
        print(anchors.tabel)

        options(repr.plot.height = 15, repr.plot.width = 22)
        plots <- PlotAnchors.STACAS(seur.anchors, obj.names=names)
        g.cols <- 3
        g.rows <- as.integer((length(plots)+2)/g.cols)
        g <- do.call("arrangeGrob", c(plots, ncol=g.cols, nrow=g.rows))
        plot(g)
        
        return(seur.anchors)
}

STACAS.filtered <- function(ref.anchors, rawobj, group.by='sampleid',
                             normalization.method='LogNormalize', IDIMS=50, NPCS=100, headname='batch',
                             dist.thr = NULL,
                             dist.pct = 0.8,
                             k.weight = 100,
                             vars.to.regress=c("nCount_RNA","percent.mt"), ...){
    ref.anchors.filtered <- FilterAnchors.STACAS(ref.anchors,  dist.thr = dist.thr, dist.pct = dist.pct)
    mySampleTree <- SampleTree.STACAS(ref.anchors.filtered)
    print(mySampleTree)
    anchors.state <- table(ref.anchors.filtered@anchors[,c("dataset1","dataset2")]) 
    print(anchors.state)

    all.genes <- row.names(ref.anchors@object.list[[1]])
    for (i in 2:length(ref.anchors@object.list)) {
       all.genes <- intersect(all.genes, row.names(ref.anchors@object.list[[i]]))
    }

    ref.integrated <- IntegrateData(anchorset=ref.anchors.filtered, 
                                    dims=1:IDIMS, 
                                    new.assay.name = "integrated",
                                    features.to.integrate=all.genes,
                                    sample.tree=mySampleTree, 
                                    k.weight=k.weight,
                                    preserve.order=T)
    
    ref.integrated@meta.data = rawobj@meta.data[colnames(ref.integrated),]

    
    DefaultAssay(ref.integrated) <- "integrated"
    options(future.globals.maxSize = 50 * 1024 ^ 10)
    plan("multiprocess", workers = 10)
    ref.integrated <- ScaleData(ref.integrated, vars.to.regress = vars.to.regress, verbose = FALSE)
    ref.integrated <- PCACLS(ref.integrated, NPCS=NPCS, group.by=group.by)
    p1 <- DimPlot(object = ref.integrated, reduction = "pca", pt.size = .1, raster =FALSE, group.by = group.by)
    p2 <- VlnPlot(object = ref.integrated, features = "PC_1", group.by = group.by, pt.size = .1)
    ph <- grid.arrange(grobs = list(p1, p2), ncol = 2)
    ggsave(paste(headname, group.by, 'STACAS.groups.DimPlot.pdf', sep='_'), ph, width = 18, height = 9)
    DefaultAssay(ref.integrated) ='integrated'
    return(ref.integrated)
}

FindIntegrationAnchors.STACAS<- function (
  object.list = NULL,
  assay = NULL,
  reference = NULL,
  anchor.features = 500,
  dims = 1:10,
  normalization.method = c("LogNormalize", "SCT"),
  k.anchor = 5,
  k.score = 30,
  reduction=c("cca", "rpca"),
  verbose = TRUE,
  ...
) {

  #perform rPCA to find anchors
  if (is.null(reference)) {
    ref.anchors <- FindIntegrationAnchors.wdist(object.list, dims = dims, k.anchor = k.anchor, 
                                                anchor.features=anchor.features,
                                                normalization.method = normalization.method,
                                                reduction=reduction, assay=assay, k.score=k.score, 
                                                verbose=verbose, ...)
  } else {
    ref.anchors <- FindIntegrationAnchors.wdist(object.list, reference=reference, dims = dims,
                                                k.anchor = k.anchor, anchor.features=anchor.features,
                                                normalization.method = normalization.method,
                                                reduction=reduction, assay=assay, k.score=k.score, 
                                                verbose=verbose, ...)
  }
  for (r in 1:dim(ref.anchors@anchors)[1]) {
    ref.anchors@anchors[r,"dist.mean"] <- mean(c(ref.anchors@anchors[r,"dist1.2"],ref.anchors@anchors[r,"dist2.1"]))
    ref.anchors@anchors[r,"dist.max"] <- max(c(ref.anchors@anchors[r,"dist1.2"],ref.anchors@anchors[r,"dist2.1"]))
    ref.anchors@anchors[r,"dist.min"] <- min(c(ref.anchors@anchors[r,"dist1.2"],ref.anchors@anchors[r,"dist2.1"]))
  }
  
  return(ref.anchors)
}
