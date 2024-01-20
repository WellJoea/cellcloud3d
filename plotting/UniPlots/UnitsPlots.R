suppressPackageStartupMessages(library(ComplexHeatmap))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(hash))

split.dotplot <- function(
  object,
  feature_df, #group.by, features
  assay = NULL,
  cols = c("lightgrey", 'yellow', 'red', 'darkred'),
  col.min = NA,
  col.max = NA,
  dot.min = 0,
  dot.scale = 6,
  bar.width = 0.35,
  slot='data',
  idents = NULL,
  group.by = NULL,
  group.col = NULL,
  split.by = NULL,
  is.facet = FALSE,
  ncol=2,
  split.cmap = list(c('#FFFFFF',viridis::viridis(100,direction=-1)),
                    c('#FFFFFF',viridis::viridis(100, direction=-1, option='magma'))),
  split.scale = TRUE,
  use.count.mean = FALSE,
  cluster.idents = FALSE,
  scale.fun = 'radius',
  scale.by = 'zscore',
  rotate.axis =TRUE,
  scale.min = NA,
  scale.max = NA
) {
  assay <- assay %||% DefaultAssay(object = object)
  DefaultAssay(object = object) <- assay

  cells <- unlist(x = CellsByIdentities(object = object, idents = idents))
  
  Features <- unique(feature_df$features)
  feature_df$uni.features <-  make.unique(feature_df$features)
  feature_df$uni.features <- factor(feature_df$uni.features,   
                                    levels=rev(feature_df$uni.features))
  #data.features <- GetAssayData(object = object, slot = slot)[Features, cells, drop = FALSE]
  data.features <- FetchData(object = object, vars = Features, cells = cells, slot = slot)
  data.features$id <- if (is.null(x = group.by)) {
    Idents(object = object)[cells, drop = TRUE]
  } else {
    object[[group.by, drop = TRUE]][cells, drop = TRUE]
  }

  if (!is.factor(x = data.features$id)) {
    data.features$id <- factor(x = data.features$id)
  }
  id.levels <- levels(x = data.features$id)
  data.features$id <- as.vector(x = data.features$id)
  data.features$groups <- data.features$id
  data.features$splits <- 'nosplit'
  unique.splits <- c('nosplit')
    
  if (!is.null(x = split.by)) {
    splits <- object[[split.by, drop = TRUE]][cells, drop = TRUE]
    unique.splits <- if (!is.factor( splits)){
        unique(x = splits)
    }else{
        levels(x = splits)
    }
    if (is.null(cols)){
        cols = RColorBrewer::brewer.pal(max(3, length(unique.splits)), split.cmap)[1:length(unique.splits)]
    }
    #names(x = cols) <- unique.splits

    data.features$id <- paste(splits, data.features$id, sep = ':')
    data.features$splits <- splits
    id.levels <- paste0(rep(x = unique.splits, times = length(x = id.levels)),":", 
                        rep(x = id.levels, each = length(x = unique.splits)))
  }
  id.split <- data.features[,c('id','splits', 'groups')]
  id.split <- data.frame(id.split[!duplicated(id.split),], check.names=FALSE)

  data.plot <- lapply(id.levels, function(ident){
      data.use <- data.features[data.features$id == ident, Features, drop = FALSE]
      if (use.count.mean){
            avg.exp <- apply( data.use, MARGIN = 2, FUN = function(x) { return(mean(x = expm1(x = x)))})  
      }else{
            avg.exp <- apply( data.use, MARGIN = 2, mean)
      }
      pct.exp <- apply(X = data.use, MARGIN = 2, FUN = PercentAbove, threshold = 0)
      return(list(avg.exp = avg.exp, pct.exp = pct.exp))
  })
  names(x = data.plot) <- id.levels

  if (cluster.idents) {
    mat <- do.call(
      what = rbind,
      args = lapply(X = data.plot, FUN = unlist)
    )
    mat <- scale(x = mat)
    id.levels <- id.levels[hclust(d = dist(x = mat))$order]
  }
  data.plot <- lapply(names(data.plot), function(x) {
      data.use <- as.data.frame(x = data.plot[[x]])
      data.use$features.plot <- rownames(x = data.use)
      data.use$id <- x
      return(data.use)
   })
  data.plot <- do.call(what = 'rbind', args = data.plot)
  data.plot <- data.plot %>% dplyr::left_join(id.split, by='id')
  data.plot$id <- factor(x = data.plot$id, levels = id.levels)
  id.split$id <- factor(x = id.split$id, levels = rev(id.levels))
  rownames(id.split) <- NULL  #id.split$id
  ngroup <- length(id.levels)

  if (ngroup == 1) {
    scale.by <- 'None'
    warning( "Only one identity present, the expression values will be not scaled", 
            call. = FALSE, immediate. = TRUE)
  } else if (ngroup < 5 & scale.by=='zscore') {
    warning( "Scaling data with a low number of groups may produce misleading results",
              call. = FALSE, immediate. = TRUE)
  }
  avg.exp <- data.plot %>% 
                tidyr::pivot_wider(id_cols =c(id, splits, groups), 
                                   names_from = features.plot, 
                                   values_from = avg.exp,
                                   names_repair='check_unique') %>%
                data.frame(check.names=FALSE)

  if (is.na(col.min) | is.na(col.max)){
      if (scale.by=='zscore') { 
          col.min = -2.5
          col.max = 2.5
      }else if(scale.by=='var') {
          col.min = 0
          col.max = 1
      }else if(scale.by=='obs') {
          col.min = 0
          col.max = 1
      }else if(use.count.mean){
          col.min = 0
          col.max = 4
      }else{
          col.min = 0
          col.max = 4
      }
  }  
  scale.meth <- function(Exp, scale.by, use.count.mean=F){
      if (scale.by=='zscore') { 
          Exp.scaled <- scale(x = Exp)
          Exp.scaled <- MinMax(Exp.scaled, min = col.min, max = col.max)
      }else if(scale.by=='var') {
          maxc = apply(Exp,2,max)
          minc = apply(Exp,2,min)
          Exp.scaled <- t((t(Exp)-minc)/(maxc -minc))  
      }else if(scale.by=='obs') {
          maxc = apply(Exp,1,max)
          minc = apply(Exp,1,min)
          Exp.scaled <- (Exp-minc)/(maxc -minc)
      }else if(use.count.mean){
          Exp.scaled <- log1p(x = Exp)
      }else{
          Exp.scaled <- Exp
      }
      return(data.frame(Exp.scaled, check.names=FALSE))
  }

  if(split.scale & length(unique(avg.exp$splits))>1){
      avg.exp.scaled <- lapply(unique(avg.exp$splits), function(i.split){
          i.exp <- subset(avg.exp, splits==i.split)
          i.scaled <- scale.meth(i.exp[, Features], scale.by, use.count.mean=use.count.mean)
          i.scaled[, c('id', 'splits', 'groups')] <- i.exp[, c('id', 'splits', 'groups')]
          return(i.scaled)
      })
      avg.exp.scaled <- do.call(what = 'rbind', args = avg.exp.scaled)
  }else{
      #avg.exp <- avg.exp %>% column_to_rownames(var="id")
      avg.exp.scaled <- scale.meth(avg.exp[,Features], scale.by, use.count.mean=use.count.mean)
      avg.exp.scaled[, c('id', 'splits', 'groups')] <- avg.exp[, c('id', 'splits', 'groups')]
  }
  avg.exp.scaled <- avg.exp.scaled %>% 
                          tidyr::pivot_longer(Features, 
                                              names_to='features.plot',
                                              values_to='avg.exp.scaled')
  data.plot <- data.plot %>% dplyr::left_join(avg.exp.scaled, by=c('id', 'splits', 'groups', 'features.plot'))
  data.plot$pct.exp[data.plot$pct.exp < dot.min] <- NA
  data.plot$pct.exp <- data.plot$pct.exp * 100

  if (!is.na(x = scale.min)) {
    data.plot[data.plot$pct.exp < scale.min, 'pct.exp'] <- scale.min
  }
  if (!is.na(x = scale.max)) {
    data.plot[data.plot$pct.exp > scale.max, 'pct.exp'] <- scale.max
  }

  data.plot <- lapply(id.levels, function(ident){
        data.plot %>% 
            filter(id==ident) %>% 
            dplyr::right_join(feature_df[,c('features', 'uni.features')], by=c('features.plot'='features'))
  })
  data.plot <- do.call(what = 'rbind', args = data.plot)
  data.plot$uni.features <- factor(data.plot$uni.features, 
                                   levels=levels(feature_df$uni.features))
  data.plot$splits <- factor(data.plot$splits, levels=unique.splits)
  scale.func <- switch(
    EXPR = scale.fun,
    'size' = scale_size,
    'radius' = scale_radius,
    stop("'scale.fun' must be either 'size' or 'radius'")
  )

  color.by <- 'avg.exp.scaled'
  size.by  <- 'pct.exp'
  sp.level <- unique(data.plot$splits)
                      
  plt <- data.plot %>% ggplot(mapping = aes_string(x = 'uni.features', y = 'id'))
  for (ix in 1:length(unique.splits)){
      isp <- unique.splits[[ix]]
      idata <- data.plot
      idata[(idata$splits!=isp),size.by] =NA
      plt <- plt +
                geom_point( data = idata,
                           mapping = aes_string(size = size.by, color = color.by)) + 
                theme(axis.text.x = element_text(angle=90, hjust=1), 
                      panel.grid.major = element_blank(), 
                      panel.grid.minor = element_blank(),
                      panel.background = element_rect(fill = "transparent", colour = "black"),
                      axis.text.y = element_text(angle=0, hjust=1)) #+
                #guides(colour = guide_legend(title = isp))
      if (!is.null(split.by) & !is.null(split.cmap)){
          plt <- plt + 
              scale_colour_gradientn(colours = split.cmap[[ix]], 
                                     limits = c(col.min, col.max)) +
              ggnewscale::new_scale("color")
      }else{
          plt <- plt + 
              scale_colour_gradientn(colours = cols, 
                                     limits = c(col.min, col.max)) +
              ggnewscale::new_scale("color")
      } 
  }
 
  plt <- plt + 
            scale.func(range = c(0, dot.scale), limits = c(scale.min, scale.max)) +
            #geom_bar(data = id.split, aes(y = id, x=0.1, fill = groups), stat = "identity") +
            geom_bar(data=feature_df, aes(x = uni.features, y=bar.width, fill = group.by), stat = "identity") 

  if (!is.null(group.col)){
        plt <- plt + scale_fill_manual(values=group.col)
  }
  plt <- plt + 
            guides(size = guide_legend(title = group.by)) +
            labs( x = 'Features',
                  y = ifelse(test = is.null(x = split.by), yes = 'Identity', no = split.by))

  if (rotate.axis){
      plt <- plt + coord_flip()
  }
      
  if (is.facet){
       #plt <- plt  + facet_grid(facets = ~splits, scales = "free_x", 
       #                         space = "free_x", switch = "y")
       plt <- plt  + facet_wrap(vars(splits), 
                                scales = "free_x", 
                                ncol=ncol)
      
  }    
  print(head(data.plot))
  return(plt)

}

GeomSplitViolin <- ggproto(
      "GeomSplitViolin",
      GeomViolin,
      draw_group = function(self, data, ..., draw_quantiles = NULL) {
        data$xminv <- data$x - data$violinwidth * (data$x - data$xmin)
        data$xmaxv <- data$x + data$violinwidth * (data$xmax - data$x)
        grp <- data[1, 'group']
        if (grp %% 2 == 1) {
          data$x <- data$xminv
          data.order <- data$y
        } else {
          data$x <- data$xmaxv
          data.order <- -data$y
        }
        newdata <- data[order(data.order), , drop = FALSE]
        newdata <- rbind(
          newdata[1, ],
          newdata,
          newdata[nrow(x = newdata), ],
          newdata[1, ]
        )
        newdata[c(1, nrow(x = newdata) - 1, nrow(x = newdata)), 'x'] <- round(x = newdata[1, 'x'])
        grob <- if (length(x = draw_quantiles) > 0 & !zero_range(x = range(data$y))) {
          stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 1))
          quantiles <- QuantileSegments(data = data, draw.quantiles = draw_quantiles)
          aesthetics <- data[rep.int(x = 1, times = nrow(x = quantiles)), 
                             setdiff(x = names(x = data), y = c("x", "y")), drop = FALSE]
          aesthetics$alpha <- rep.int(x = 1, nrow(x = quantiles))
          both <- cbind(quantiles, aesthetics)
          quantile.grob <- GeomPath$draw_panel(both, ...)
          grobTree(GeomPolygon$draw_panel(newdata, ...), name = quantile.grob)
        }
        else {
          GeomPolygon$draw_panel(newdata, ...)
        }
        grob$name <- grobName(grob = grob, prefix = 'geom_split_violin')
        return(grob)
      }
)

geom_split_violin <- function(
  mapping = NULL,
  data = NULL,
  stat = 'ydensity',
  position = 'identity',
  ...,
  draw_quantiles = NULL,
  trim = TRUE,
  scale = 'area',
  na.rm = FALSE,
  show.legend = NA,
  inherit.aes = TRUE
) {
  return(layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomSplitViolin,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      draw_quantiles = draw_quantiles,
      na.rm = na.rm,
      ...
    )
  ))
}
      

ComHeatmap <-function(object, 
                      features = NULL, 
                       features.select = NULL,
                       features.split.name =NULL,
                       features.split = NULL,
                       col.split = NULL,
                        hcolors = c("white", "yellow", "red", "darkred"), 
                        col_fun=NULL,
                        cells = NULL, 
                        column_title='heatmap',
                        name= "Expression",
                        group.by = c("ident"), 
                        group.bar = TRUE, 
                        group.colors = NULL, 
                        show_column_names=FALSE,
                        show_row_names=TRUE,
                        show_left_label=TRUE,
                        show_top_label=TRUE,
                        row_gap=0.5,
                        column_gap=0.5,
                        disp.min = NULL, 
                        disp.max = NULL, 
                        slot = "data", 
                        assay = 'RNA',
                        label = TRUE, 
                        size = 5.5, 
                        bf_lty =2,
                        bf_col ='black',
                        hjust = 0, angle = 45,
                        slwd=1.5,
                        scol='black',
                        labels_fz=10,
                        raster = FALSE, 
                        raster_by_magick=TRUE,
                        draw.lines = TRUE, 
                        lines.width = NULL, 
                        scale_by='out_scale',
                        scale.split=NULL,
                        group.bar.height = 1, 
                        combine = TRUE, ...){
    library(grid)
    library(ComplexHeatmap)

    `%||%` <- Seurat:::`%||%`
    cells <- cells %||% colnames(x = object)
    if (is.numeric(x = cells)) {
        cells <- colnames(x = object)[cells]
    }

    assay <- assay %||% DefaultAssay(object = object)
    DefaultAssay(object = object) <- assay
    features <- features %||% VariableFeatures(object = object)
    #features <- rev(x = unique(x = features))
    disp.max <- disp.max %||% ifelse(test = slot == "scale.data",  yes = 2.5,  no = 6)
    disp.min <- disp.min %||% ifelse(test = slot == "scale.data",  yes = -2.5, no = 0)
    possible.features <- rownames(x = GetAssayData(object = object,  slot = slot))

    if (any(!features %in% possible.features)) {
        bad.features <- features[!features %in% possible.features]
        features <- features[features %in% possible.features]
        if (length(x = features) == 0) {
            stop("No requested features found in the ", slot, 
                " slot for the ", assay, " assay.")
        }
        warning("The following features were omitted as they were not found in the ", 
                slot, " slot for the ", assay, " assay: ", 
                paste(bad.features, collapse = ", "))
    }
    data <- GetAssayData(object = object, slot = slot)[features, cells, drop = FALSE]
    #object <- suppressMessages(expr = StashIdent(object = object, save.name = "ident"))
    group.by <- group.by %||% c("ident")
    features.split.name <- features.split.name %||% group.by[1]
    col.split <- col.split %||%  features.split.name

    if(col.split %in% group.by){
        groups.by <- group.by
    }else{
        groups.by <- c(col.split, group.by)
    }
    groups.use <- object[[groups.by]][cells, , drop = FALSE] %>% dplyr::arrange(!!!rlang::syms(groups.by))
    col.split  <- groups.use[,col.split]

    data <- data[, row.names(groups.use)]
    data <- as.matrix(data)
    disp.min <- disp.min %||% min(data)
    disp.max <- disp.max %||% max(data)
    data[data>disp.max] <- disp.max
    data[data<disp.min] <- disp.min
    if (scale_by=='var'){
        if (is.null(scale.split)){
            maxc = apply(data,1,max)
            minc = apply(data,1,min)
            data <- (data-minc)/(maxc -minc)
        }else{
            sdata <- lapply(unique(object@meta.data[, scale.split]), function(ig){
                icells <- rownames(object@meta.data)[object@meta.data[, scale.split]==ig]
                idata <-  data[, icells]
                maxc = apply(idata,1,max)
                minc = apply(idata,1,min)
                idata <- (idata-minc)/(maxc -minc)
                return(idata)
            })
            sdata <- do.call(cbind, sdata)
            data <- sdata[rownames(data), colnames(data)]
            rm(sdata)
        }
        disp.min <-0
        disp.max <-1
    }else if(scale_by=='obs'){
        maxc = apply(data,2,max)
        minc = apply(data,2,min)
        data <- t( (t(data)-minc)/(maxc -minc))
        disp.min <-0
        disp.max <-1
    }
    
    colours <- list()
    for (g in group.by){
        if(!is.factor(groups.use[,g])){
            groups.use[,g] <-factor(groups.use[,g])
        }
        colours[[g]] = group.colors[[g]][1:length(levels(groups.use[,g]))]
        names(colours[[g]]) = levels(groups.use[,g])
    }
    col_anno <- HeatmapAnnotation(df=dplyr::select(groups.use, all_of(group.by)), 
                                  name=group.by,
                                  which="col",
                                  col=colours, 
                                  show_annotation_name=show_top_label,
                                  annotation_width=unit(rep(group.bar.height, length(group.by)), "cm"), 
                                  gap=unit(1, "mm"))

    rc_anno <- anno_block(gp = gpar(fill = c("#c77cff","#FF9999"), col="white"),
                          height = unit(5, "mm"),
                          labels = c("TESA", "TESB"),
                          labels_gp = gpar(col = "white", fontsize = 8, fontface="bold"))
    rc_anno <-  HeatmapAnnotation(group=rc_anno)
        
    border_gp  = grid::gpar(col = bf_col, lty = 1, lwd =bf_lty)
    layer_fun = NULL

    if(!is.null(features.split)){
        if (length(setdiff(unique(features.split),levels(col.split)))==0){
            features.split = factor(features.split,
                                    levels=levels(col.split))
        }else if(!is.factor(features.split)){
             features.split = factor(features.split)
        }
        row_km <- length(levels(features.split))
        column_km <- length(levels(col.split))
        min_cycle <- min(row_km, column_km)
        border_gp  <- grid::gpar(col = NA, lty = 0, lwd =0)
        layer_fun <- function(j, i, x, y, width, height, fill, slice_r, slice_c){
                        v = pindex(data, i, j)
                        #grid.text(sprintf("%.1f", v), x, y, gp = gpar(fontsize = 10))
                        #if(slice_c==slice_r)
                        if(slice_c %% min_cycle ==slice_r %% min_cycle){
                            grid.rect(#x = x, y = y, width = width, height = height, 
                                      gp = grid::gpar(lwd = slwd, col=scol, fill = "transparent"))
                        }
        }
        rowleft = data.frame(groups=features.split, check.names=FALSE)
        colnames(rowleft) <- c(features.split.name)
        rowleft_anno <- HeatmapAnnotation(df=rowleft,
                              name=features.split.name,
                              labels=NULL,
                              show_annotation_name=show_left_label,
                              which="row",
                              show_legend=FALSE,
                              col=group.colors[features.split.name], 
                              annotation_width=unit(group.bar.height, "cm"), 
                              gap=unit(1, "mm"))
        
    }

    row_anno <- NULL
    if (!is.null(features.select)){
        features.select <- intersect(features.select, features)
        gene.pos <- match(features.select, features)
        show_row_names <- FALSE
        row_anno <- rowAnnotation( Feautrues = anno_mark(at = gene.pos,
                                                         which='row',
                                                         side='right',
                                                         labels = features.select,
                                                         lines_gp = grid::gpar(),
                                                         labels_gp = grid::gpar(fontsize = labels_fz),
                                                         padding = unit(1, "mm"),
                                                         link_width = unit(5, "mm"),
                                                         extend = unit(2, "mm"))
    )}

    if (is.null(col_fun)){ 
        #col_fun = scale_fill_gradientn(colors = c("white", "yellow", "red", "darkred"))
        col_fun <- circlize::colorRamp2(seq(disp.min, disp.max, length.out=length(hcolors)), hcolors)
    }

    Phmp <- ComplexHeatmap::Heatmap(data,
                column_title = column_title,
                name = name,
                col = col_fun,
                use_raster = raster,
                raster_by_magick = raster_by_magick,
                border_gp = border_gp,
                rect_gp = grid::gpar(col = NA, lwd =0),
                cluster_rows=FALSE, 
                cluster_columns=FALSE, 

                top_annotation = col_anno,
                right_annotation = row_anno,
                left_annotation = rowleft_anno,

                row_names_side = "left", 
                row_dend_side = "left",
                row_split = features.split,
                #row_title = NULL,
                show_row_names = show_row_names,
                row_gap = unit(row_gap, "mm"),

                column_names_side = "bottom", 
                show_column_names = show_column_names,
                column_dend_side = "bottom",
                column_split = col.split,
                column_gap = unit(column_gap, "mm"),
                
                layer_fun = layer_fun,
                ...)
    #draw(Phmp)
    return(Phmp)        
}

comb.compheat<- function(..., ncol=1, clip=TRUE){
    library(gridGraphics)
    library(grid)

    drawGridHeatmap <- function(hm){
      grid.echo(hm)
      grid.grab()
    }

    drawGridHeatmap  <- function(hm) {
        draw(hm)
        grid.grab()
    }
    gl <- lapply(list(...), drawGridHeatmap)
    plts <- gridExtra::grid.arrange(grobs=gl, ncol=ncol, clip=clip)
    return(plts)
}


ComHeatmaptest <- function(subobj){
    sidcol <-  c('#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', 
                      '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
          '#c49c94', '#f7b6d2', '#dbdb8d')
    group.colors <- c('#279e68', '#d62728')
    features.select <- c("NTRK3", "RUNX3", 'PDE4B', 'DPP10', 'LRP1B', 'ROBO1',
                         "NTRK1", "RUNX1", 'RBFOX1', 'SYN3', 'PID1', 'RUNX1', 'LDB2')

    height = 12
    width = 8
    disp.min=0
    disp.max=4

    options(repr.plot.height = height, repr.plot.width = width)
    Phmp <- ComHeatmap(subobj, 
                           features = marker.12$gene, 
                           features.split = marker.12$group,
                           features.select = features.select,
                           group.by =c('CellType', 'GW'),
                           group.colors=list(CellType=group.colors, GW=sidcol),
                           slot=slot, 
                           raster=FALSE,
                           disp.min= 0, 
                           disp.max =4,
                           height=height,
                           width=width,
                           row_names_gp = grid::gpar(fontsize = 8),
                           heatmap_legend_param = list(title = 'Expression',
                                                       border = "grey", 
                                                       lwd=1,
                                                       legend_height = unit(4, "cm")))
}   
