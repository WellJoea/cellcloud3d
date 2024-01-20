# Connect to AnnotationHub

geneDescript <- function(){
    library(AnnotationHub)
    library(ensembldb)
    
    ah <- AnnotationHub()
    # Access the Ensembl database for organism
    ahDb <- query(ah, 
                  pattern = c("Homo sapiens", "EnsDb"), 
                  ignore.case = TRUE)

    # Acquire the latest annotation files
    id <- ahDb %>%
            mcols() %>%
            rownames() %>%
            tail(n = 1)

    # Download the appropriate Ensembldb database
    edb <- ah[[id]]

    # Extract gene-level information from database
    annotations <- genes(edb, 
                         return.type = "data.frame")

    # Select annotations of interest
    annotations <- annotations %>%
            dplyr::select(gene_id, gene_name, seq_name, gene_biotype, description)
    annotations
}


FindIntegrationMatrix <- function(
  object,
  assay = NULL,
  integration.name = 'integrated',
  features.integrate = NULL,
  block.size = NULL,
  verbose = TRUE
) {
  library(future)
  library(future.apply)

  assay <- assay %||% DefaultAssay(object = object)
  neighbors <- GetIntegrationData(object = object, integration.name = integration.name, slot = 'neighbors')
  nn.cells1 <- neighbors$cells1
  nn.cells2 <- neighbors$cells2
  anchors <- GetIntegrationData(
    object = object,
    integration.name = integration.name,
    slot = 'anchors'
  )
  if (verbose) {
    message("Finding integration vectors")
  }
  features.integrate <- features.integrate %||% rownames(
    x = GetAssayData(object = object, assay = assay, slot = "data")
  )
  data.use1 <- t(x = GetAssayData(
    object = object,
    assay = assay,
    slot = "data")[features.integrate, nn.cells1]
  )
  data.use2 <- t(x = GetAssayData(
    object = object,
    assay = assay,
    slot = "data")[features.integrate, nn.cells2]
  )
  anchors1 <- nn.cells1[anchors[, "cell1"]]
  anchors2 <- nn.cells2[anchors[, "cell2"]]
  data.use1 <- data.use1[anchors1, ]
  data.use2 <- data.use2[anchors2, ]

  #integration.matrix <- data.use2 - data.use1
  print('use chunk...')
  dsize=max(nrow(data.use1), nrow(data.use2))
  chunk.points <- Seurat:::ChunkPoints(
    dsize = dsize,
    csize = block.size %||% ceiling(x = dsize / nbrOfWorkers())
  )
  integration.matrix <- future_lapply(
    X = 1:ncol(x = chunk.points),
    FUN = function(i) {
      block <- chunk.points[, i]
      data1 <- data.use1[block[1]:block[2], , drop = FALSE]
      data2 <- data.use2[block[1]:block[2], , drop = FALSE]
      return(data2 - data1)
    }
  )
  integration.matrix <- do.call(rbind, integration.matrix)
  print(dim(integration.matrix))
  object <- SetIntegrationData(
    object = object,
    integration.name = integration.name,
    slot = 'integration.matrix',
    new.data = integration.matrix
  )
  return(object)
}
#library(R.utils);
#reassignInPackage("FindIntegrationMatrix", pkgName="Seurat", FindIntegrationMatrix)
#options(future.globals.maxSize= 500*1024^3)

# RenameGenesSeurat  ------------------------------------------------------------------------------------
RenameGenesSeurat <- function(seuojb, newnames) {
  print("Run this before integration. It only changes obj@assays$RNA@counts, @data and @scale.data.")
  RNA <- seuojb@assays$RNA

  if (nrow(RNA) == length(newnames)) {
    if (length(RNA@counts)) RNA@counts@Dimnames[[1]]            <- newnames
    if (length(RNA@data)) RNA@data@Dimnames[[1]]                <- newnames
    if (length(RNA@scale.data)) RNA@scale.data <- new(Class = "matrix")
  } else {"Unequal gene sets: nrow(RNA) != nrow(newnames)"}
  seuojb@assays$RNA <- RNA
  return(seuojb)
}
# RenameGenesSeurat(obj = SeuratObj, newnames = HGNC.updated.genes)