library(velocyto.R)
library(pagoda2)

HE10_DRGpremrna = '/home/gpfs/home/wulab17/WorkSpace/11Project/02DRG/00DataBase/DRG_cellrangercount/HE10_DRGpremrna/velocyto/HE10_DRGpremrna.loom'
HE10_DRG = '/home/gpfs/home/wulab17/WorkSpace/11Project/02DRG/00DataBase/DRG_cellrangercount/HE10_DRG/velocyto/HE10_DRG.loom'

vlmpre = read.loom.matrices(HE10_DRGpremrna)
vlm = read.loom.matrices(HE10_DRG)

emat <- vlm$spliced
# this dataset has already been pre-filtered, but this is where one woudl do some filtering
emat <- emat[,colSums(emat)>=1e3]

getinfo <-function(){
    hist(log10(rowSums(emat)+1),col='red',xlab='log10[number of reads + 1]')
    dim(emat)  
    emat <- emat[as.numeric(rowSums(emat)) != 0,]
    dim(emat)
}

DropDup <-function(){
    emat <- ldat$spliced
    dim(emat)
    emat <- emat[,colSums(emat)>=1e3]
    dim(emat)
    emat <- emat[duplicated(rownames(emat))=="FALSE",]
    dim(emat)
}

DropSum <- function(sparmtx){
    mtx <- as.matrix(sparmtx)
    rct <- table(rownames(mtx))
    #names(rct[rct>1])
    #dim(mtx)
    #sum(mtx[rownames(mtx) %in% names(rct[rct>1]),])
    mtx <- rowsum(mtx, row.names(mtx), reorder = FALSE, na.rm = FALSE,)
    mtx <- Matrix(mtx, sparse = TRUE) 
    #dim(mtx)
    #sum(mtx[rownames(mtx) %in% names(rct[rct>1]),])
    #cc= Matrix(aa, sparse = TRUE) 
    #identical(cc, emat)
    return(mtx)
}
emat <- DropSum(emat)

r <- Pagoda2$new(emat,modelType='plain',trim=10,log.scale=T)

r$adjustVariance(plot=T,do.par=T,gam.k=10)

options(repr.plot.height = 10, repr.plot.width = 22)
r$calculatePcaReduction(nPcs=100,n.odgenes=3e3,maxit=300)

r$makeKnnGraph(k=30,type='PCA',center=T,distance='cosine');

r$getKnnClusters(method=multilevel.community,type='PCA',name='multilevel')
r$getEmbedding(type='PCA',embeddingType='tSNE',perplexity=50,verbose=T)

#r$getEmbedding(type='PCA',embeddingType='UMAP',perplexity=50,verbose=T)

par(mfrow=c(1,2))
r$plotEmbedding(type='PCA',embeddingType='tSNE',show.legend=F,mark.clusters=T,min.group.size=10,shuffle.colors=F,mark.cluster.cex=1,alpha=0.3,main='cell clusters')
r$plotEmbedding(type='PCA',embeddingType='tSNE',colors=r$depth,main='depth')  

emat <- vlm$spliced; nmat <- vlm$unspliced
emat <- emat[,colSums(emat)>=1e3]; nmat <- nmat[,colSums(nmat)>=1e3]
emat <- DropSum(vlm$spliced); nmat <- DropSum(vlm$unspliced)

emat <- emat[,rownames(r$counts)]; nmat <- nmat[,rownames(r$counts)]; # restrict to cells that passed p2 filter
# take cluster labels
cluster.label <- r$clusters$PCA[[1]]
cell.colors <- sccore::fac2col(cluster.label)
# take embedding
emb <- r$embeddings$PCA$tSNE

cell.dist <- as.dist(1-armaCor(t(r$reductions$PCA)))

emat <- filter.genes.by.cluster.expression(emat,cluster.label,min.max.cluster.average = 0.5)
nmat <- filter.genes.by.cluster.expression(nmat,cluster.label,min.max.cluster.average = 0.05)
length(intersect(rownames(emat),rownames(nmat)))

fit.quantile <- 0.02
rvel.cd <- gene.relative.velocity.estimates(emat,nmat,deltaT=1,kCells=20,cell.dist=cell.dist,fit.quantile=fit.quantile)

show.velocity.on.embedding.cor(emb,rvel.cd,n=300,scale='sqrt',cell.colors=ac(cell.colors,alpha=0.5),cex=0.8,arrow.scale=5,show.grid.flow=TRUE,min.grid.cell.mass=0.5,grid.n=40,arrow.lwd=1,do.par=F,cell.border.alpha = 0.1)

p <- parallel:::mcfork()
if (inherits(p, "masterProcess")) {
    cat("I'm a child! ", Sys.getpid(), "\n")
    parallel:::mcexit(,"I was a child")
}
cat("I'm the master\n")
unserialize(parallel:::readChildren(1.5))
