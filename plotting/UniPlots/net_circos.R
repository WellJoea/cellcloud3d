library(circlize)
suppressPackageStartupMessages(library(dplyr))

scPalette <- function(n, colors=NULL) {
  colorSpace<- c('#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', 
                  '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8', 
                  '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
                 '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
                 '#B240CE','#B6B51F','#0780CF','#765005','#FA6D1D',
                 '#0E2C82','#DA1F18','#701866','#F47A75','#009DB2',
                 '#024B51','#0780CF','#765005','#6BEFFC','#3B45DD',
                 '#AD94EC','#00749D','#6ED0A7','#2F3EA8','#706C01',
                 '#9BE4FF','#D70000',
                 '#E41A1C','#377EB8','#4DAF4A','#984EA3','#F29403','#F781BF',
                 '#BC9DCC','#A65628','#54B0E4','#222F75','#1B9E77','#B2DF8A',
                 '#E3BE00','#FB9A99','#E7298A','#910241','#00CDD1','#A6CEE3',
                 '#CE1261','#5E4FA2','#8CA77B','#00441B','#DEDC00','#B3DE69',
                 '#8DD3C7','#999999')
    
  if(!is.null(colors)){
      colors = colors
  }else if (n <= length(colorSpace)) {
    colors <- colorSpace[1:n]
  } else {
    colors <- grDevices::colorRampPalette(colorSpace)(n)
  }
  return(colors[1:n])
}

net_mult_circos <- function(df, 
                            color.use = NULL,  cell.order = NULL,
                            group = NULL, use.group=TRUE,
                            sort.by='stlr',
                            start.degree = NA,
                            sources.use = NULL, targets.use = NULL,
                            lab.cex = 1, small.gap = 0.3, big.gap = 10, 
                            annotationTrackHeight = c(0.025),
                            target.prop.height = 0.025,
                            diffHeight = 0.05, 
                            remove.isolate = FALSE, link.visible = TRUE, scale = FALSE, directional = 1, 
                            link.target.prop = TRUE, reduce = -1,
                            show.title=TRUE,
                            transparency = 0.4, link.border = NA,
                            char_height_scale=0.02,
                            legend_fs=NA,
                            circle.margin=c(0.1, 0.1, 0.1, 0.1),
                            #gap.degree=0.5,
                            title.name = NULL, show.legend = TRUE, 
                            legend.pos.x = 0.95, legend.pos.y = 0.15, lengend_just=c("right", "center"),
                            legend_ncol=1, legend_arg=NULL,
                            thresh = 0,...){
  df <- df[,c('source', 'target', 'ligand', 'receptor', 'prob')]
  df <- df[(df$prob > thresh),]
  rec.levels <- unique(c(df$source, df$target))
  if (! is.factor(df$source)){
      df$source <- factor(df$source, levels=rec.levels)
  }
  if (! is.factor(df$target)){
      df$target <- factor(df$target, levels=rec.levels)
  }

  if (nrow(df) == 0) {
    stop("No signaling links are inferred! ")
  }

  if (!is.null(cell.order)){
    cell.levels <- cell.order
  }else{
    cell.levels <- tryCatch({levels(df$source)},
                            error = function(e){unique(c(df$source, df$target))}
                   ) %>% as.character()
  }

  if (is.null(sources.use)){
    sources.use <- tryCatch({levels(droplevels(df$source))},
                            error = function(e){unique(df$source)}
                   ) %>% as.character()
  }
  if (is.null(targets.use)){
    targets.use <- tryCatch({levels(droplevels(df$target))},
                            error = function(e){unique(df$target)}
                   ) %>% as.character()
  }

  df$id <- 1:nrow(df)
  # deal with duplicated sector names
  deprecated <-function(){
      ligand.uni <- unique(df$ligand)
      for (i in 1:length(ligand.uni)) {
        df.i <- df[df$ligand == ligand.uni[i], ]
        source.uni <- unique(as.character(df.i$source))
        for (j in 1:length(source.uni)) {
          df.i.j <- df.i[df.i$source == source.uni[j], ]
          df.i.j$ligand <- paste0(df.i.j$ligand, paste(rep(' ',j-1),collapse = ''))
          df$ligand[df$id %in% df.i.j$id] <- df.i.j$ligand
        }
      }

      receptor.uni <- unique(df$receptor)
      for (i in 1:length(receptor.uni)) {
        df.i <- df[df$receptor == receptor.uni[i], ]
        target.uni <- unique(as.character(df.i$target))
        for (j in 1:length(target.uni)) {
          df.i.j <- df.i[df.i$target == target.uni[j], ]
          df.i.j$receptor <- paste0(df.i.j$receptor, paste(rep(' ',j-1),collapse = ''))
          df$receptor[df$id %in% df.i.j$id] <- df.i.j$receptor
        }
      }
  }
  df$genel <- df$ligand
  df$gener <- df$receptor
  df <- df %>% group_by(genel)%>% mutate(probsuma=sum(prob))
  df <- df %>% group_by(gener)%>% mutate(probsumb=sum(prob))

  df$ligand <- paste(df$ligand, df$source, 'source', sep='@')
  df$receptor <- paste(df$receptor, df$target, 'target', sep='@')

  cell.order.sources <- cell.levels[cell.levels %in% sources.use]
  cell.order.targets <- cell.levels[cell.levels %in% targets.use]

  df$source <- factor(df$source, levels = cell.order.sources)
  df$target <- factor(df$target, levels = cell.order.targets)

  if (sort.by == 'lr'){
      df.ordered.source <- df[with(df, order(probsuma, desc(source), prob)), ]
      df.ordered.target <- df[with(df, order(-probsumb, target, -prob)), ]
  }else if(sort.by == 'st'){
      df.ordered.source <- df[with(df, order(desc(source), probsuma, prob)), ]
      df.ordered.target <- df[with(df, order(target, -probsumb, -prob)), ]
  }else{
      df.ordered.source <- df[with(df, order(desc(source), target, probsuma, prob)), ]
      df.ordered.target <- df[with(df, order(target, desc(source), -probsumb, -prob)), ]
  }
  order.source <- unique(df.ordered.source[ ,c('ligand','source')])
  order.target <- unique(df.ordered.target[ ,c('receptor','target')])

  # define sector order
  order.sector <- c(order.source$ligand, order.target$receptor)
  # define cell type color
  color.use = scPalette(length(cell.levels), colors=color.use)
  names(color.use) <- cell.levels

  # define edge color
  edge.color <- color.use[as.character(df.ordered.source$source)]
  names(edge.color) <- as.character(df.ordered.source$source)

  # define grid colors
  grid.col.ligand <- color.use[as.character(order.source$source)]
  names(grid.col.ligand) <- as.character(order.source$source)
  grid.col.receptor <- color.use[as.character(order.target$target)]
  names(grid.col.receptor) <- as.character(order.target$target)
  grid.col <- c(as.character(grid.col.ligand), as.character(grid.col.receptor))
  names(grid.col) <- order.sector

  if (is.null(group) & use.group){
     ga <- unique(order.source$ligand)
     gb <- unique(order.target$receptor)
     group <- c(rep('A', length(ga)), rep('B', length(gb)))
     names(group) <- c(ga, gb)
  }
  df.plot <- df.ordered.source[ ,c('ligand','receptor','prob')]

  if (directional == 2) {
    link.arr.type = "triangle"
  } else {
    link.arr.type = "big.arrow"
  }
  preAllocateTracks <- function(){
    if (char_height_scale>0){
       preAllocateTracks = list(track.height = max(strwidth(order.sector))*char_height_scale)
    }else{
       preAllocateTracks = NULL
    }
      return(preAllocateTracks)
  }
  if (is.na(start.degree)){
        start.degree <- ifelse(use.group, -(big.gap/2+small.gap), 0)
  }
  circos.clear()
  circos.par(circle.margin=circle.margin, start.degree = start.degree )
  chordDiagram(df.plot,
               order = order.sector,
               col = edge.color,
               grid.col = grid.col,
               transparency = transparency,
               link.border = link.border,
               directional = directional,
               direction.type = c("diffHeight","arrows"),
               diffHeight = diffHeight,
               target.prop.height = target.prop.height,
               link.arr.type = link.arr.type,
               annotationTrack = "grid",
               annotationTrackHeight = annotationTrackHeight,
               preAllocateTracks = preAllocateTracks(),
               small.gap = small.gap,
               big.gap = big.gap,
               link.visible = link.visible,
               scale = scale,
               group = group,
               link.target.prop = link.target.prop,
               reduce = reduce,
               ...)
  #abline(v = 0, lty = 2, col = "#00000080")
  if (show.title){
    circos.track(track.index = 1, panel.fun = function(x, y) {
        xlim = get.cell.meta.data("xlim")
        xplot = get.cell.meta.data("xplot")
        ylim = get.cell.meta.data("ylim")
        sector.name = get.cell.meta.data("sector.index")
        circos.text(mean(xlim), ylim[1], 
        unlist(lapply( strsplit(sector.name,split = '@'), function(x){x[1]})),
        #trimws(sector.name, which = "both" ), 
        facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5),cex = lab.cex)
    }, bg.border = NA)
  }
  # https://jokergoo.github.io/circlize_book/book/legends.html
  if (show.legend) {
    legend_arg <- c(list(at = names(color.use), type = "grid", 
                        labels_gp = grid::gpar(col = "black", fontsize = legend_fs),
                        #legend_height = grid::unit(1, "npc"),
                        #legend_width  = grid::unit(1, "npc"),
                        ncol = legend_ncol,
                        legend_gp = grid::gpar(fill = color.use)), 
                    legend_arg)
    lgd <- do.call(ComplexHeatmap::Legend, args=legend_arg)
    ComplexHeatmap::draw(lgd, 
                         x = grid::unit(legend.pos.x, "npc"), 
                         y = grid::unit(legend.pos.y, "npc"), 
                         #x = grid::unit(1, "npc")-grid::unit(legend.pos.x, "mm"),
                         #y = grid::unit(legend.pos.y, "npc"), 
                         just = lengend_just)

  }

  if(!is.null(title.name)){
    text(-0, 1.02, title.name, cex=1)
  }
  circos.clear()
  gg <- recordPlot()
  return(gg)
}

net_circos <- function(net, table.type='width', color.use = NULL, group = NULL, cell.order = NULL,
                      sources.use = NULL, targets.use = NULL,
                      lab.cex = 0.8,small.gap = 1, big.gap = 10, annotationTrackHeight = c(0.05),
                      diffHeight = 0.07, 
                      target.prop.height = 0.05,
                      remove.isolate = FALSE, link.visible = TRUE, scale = FALSE, directional = 1, 
                      link.target.prop = TRUE, reduce = -1,
                      show.title=FALSE,
                      transparency = 0.5, link.border = NA,
                      char_height_scale = 0.02,
                      title.name = NULL, show.legend = TRUE, legend.pos.x = -10, legend.pos.y = 50,...){
  if (table.type=='width') {
    cell.levels <- union(rownames(net), colnames(net))
    net <- reshape2::melt(net, value.name = "prob")
    colnames(net)[1:2] <- c("source","target")
  } else if (table.type=='longer') {
    if (all(c("source","target", "prob") %in% colnames(net)) == FALSE) {
      stop("The input data frame must contain three columns named as source, target, prob")
    }
    cell.levels <- as.character(union(net$source,net$target))
  }
    
  if (!is.null(cell.order)) {
    cell.levels <- cell.order
  }
  # define grid color
  color.use = scPalette(length(cell.levels), colors=color.use)
  names(color.use) <- cell.levels

  net$source <- as.character(net$source)
  net$target <- as.character(net$target)

  # keep the interactions associated with sources and targets of interest
  if (!is.null(sources.use)){
    if (is.numeric(sources.use)) {
      sources.use <- cell.levels[sources.use]
    }
    net <- subset(net, source %in% sources.use)
  }
  if (!is.null(targets.use)){
    if (is.numeric(targets.use)) {
      targets.use <- cell.levels[targets.use]
    }
    net <- subset(net, target %in% targets.use)
  }
  # remove the interactions with zero values
  #net[net$prob==0, 'prob'] = 0.000000001
  net <- subset(net, prob > 0)

  if(dim(net)[1]<=0){message("No interaction between those cells")}
  # create a fake data if keeping the cell types (i.e., sectors) without any interactions
  if (!remove.isolate) {
    cells.removed <- setdiff(cell.levels, as.character(union(net$source,net$target)))
    if (length(cells.removed) > 0) {
      net.fake <- data.frame(cells.removed, cells.removed, 1e-10*sample(length(cells.removed), length(cells.removed)))
      colnames(net.fake) <- colnames(net)
      net <- rbind(net, net.fake)
      link.visible <- net[, 1:2]
      link.visible$plot <- FALSE
      if(nrow(net) > nrow(net.fake)){
        link.visible$plot[1:(nrow(net) - nrow(net.fake))] <- TRUE
      }
      # directional <- net[, 1:2]
      # directional$plot <- 0
      # directional$plot[1:(nrow(net) - nrow(net.fake))] <- 1
      # link.arr.type = "big.arrow"
      # message("Set scale = TRUE when remove.isolate = FALSE")
      scale = TRUE
    }
  }

  df <- net
  cells.use <- union(df$source,df$target)
  # define grid order
  order.sector <- cell.levels[cell.levels %in% cells.use]
  #df$source <- factor(df$source, levels=order.sector)
  #df$target <- factor(df$target, levels=order.sector)
  #df <- df %>%dplyr::arrange(source, target, desc(prob))

  # define grid color 
  grid.col <- color.use[order.sector]
  names(grid.col) <- order.sector

  # set grouping information
  if (!is.null(group)) {
    group <- group[names(group) %in% order.sector]
  }
  # define edge color
  edge.color <- color.use[as.character(df$source)]

  if (directional == 0 | directional == 2) {
    link.arr.type = "triangle"
  } else {
    link.arr.type = "big.arrow"
  }

  preAllocateTracks <- function(){
    if (char_height_scale>0){
       preAllocateTracks = list(track.height = max(strwidth(order.sector))*char_height_scale)
    }else{
       preAllocateTracks = NULL
    }
      return(preAllocateTracks)
  }
  arr.col = data.frame(df$source, df$target, rep('black', nrow(df)))
  circos.clear()
  #circos.par(gap.after = c(rep(5, nrow(df)-1), 15, rep(5, ncol(df)-1), 15))
  chordDiagram(df,
               order = order.sector,
               col = edge.color,
               grid.col = grid.col,
               transparency = transparency,
               link.border = link.border,
               directional = 1,
               direction.type = c("diffHeight","arrows"), #'arrows',
               diffHeight = diffHeight,
               target.prop.height = target.prop.height,
               link.arr.type = link.arr.type, # link.border = "white",
               annotationTrack = "grid", #c("name", "grid"), #
               annotationTrackHeight = annotationTrackHeight,
               preAllocateTracks = preAllocateTracks(),
               small.gap = small.gap,
               big.gap = big.gap,
               link.visible = link.visible,
               scale = scale,
               group = group,
               link.target.prop = link.target.prop,
               reduce = reduce,
               ...)

  if (show.title){
      circos.track(track.index = 1, panel.fun = function(x, y) {
        xlim = get.cell.meta.data("xlim")
        xplot = get.cell.meta.data("xplot")
        ylim = get.cell.meta.data("ylim")
        sector.name = get.cell.meta.data("sector.index")
        circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5),cex = lab.cex)
      }, bg.border = NA)
  }
  # https://jokergoo.github.io/circlize_book/book/legends.html
  if (show.legend) {
    lgd <- ComplexHeatmap::Legend(at = names(grid.col), type = "grid", 
                                  legend_gp = grid::gpar(fill = grid.col), title = "Cell State")
    ComplexHeatmap::draw(lgd, x = grid::unit(1, "npc")-grid::unit(legend.pos.x, "mm"), 
                         y = grid::unit(legend.pos.y, "mm"), just = c("right", "bottom"))
  }

  if(!is.null(title.name)){
    # title(title.name, cex = 1)
    text(-0, 1.02, title.name, cex=1)
  }

  circos.clear()
  gg <- recordPlot()
  return(gg)
}