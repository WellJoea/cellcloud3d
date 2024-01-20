suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(dplyr))

`%||%`  <- function (lhs, rhs){
    if (!is.null(x = lhs)) {
        return(lhs)
    }
    else {
        return(rhs)
    }
}

CellState <-function(metedf, compare='CellType',groups='species', clusts='clusters'){
    library(dplyr)
    GG = metedf[c(clusts, groups, compare)] %>%
                group_by_( compare, clusts, groups) %>%
                summarise(countS = n()) %>%
                mutate(countC =sum(countS), maxC=max(countS)) %>%
                ungroup(!!rlang::sym(clusts)) %>%
                mutate(countL =sum(countS)) %>%
                ungroup(!!rlang::sym(compare)) %>%
                mutate(countA =sum(countS)) %>%
                mutate(freqP = countS/countC, 
                       freqS = countS/countL, 
                       freqC = countC/countL, 
                       freqL = countL/countA) %>%
                arrange(desc(freqL), desc(freqC), desc(countS))

    `%||%` <- Seurat:::`%||%`

    #df <- data.frame(lapply(df, as.factor), stringsAsFactors=TRUE)
    groups.od  <- levels(GG[, groups]) %||%  sort(unique(unlist(GG[, groups])))
    compare.od <- levels(GG[, compare]) %||%  sort(unique(unlist(GG[,compare])))

    KK <- GG %>% 
        group_by(!!rlang::sym(clusts)) %>% 
        slice_max(order_by = maxC, n=1) %>% data.frame()
    KK[, groups] <- factor(KK[, groups], levels=groups.od)
    KK[, compare] <- factor(KK[, compare], levels=compare.od)
    KK <- KK %>% arrange(!!rlang::sym(groups), !!rlang::sym(compare), -countL, -countC)
    
    return(list(GG, KK))
}

RiverPlot <- function(metedf, orderdf=NULL, nodeorder=NULL, 
                      compare='CellType', groups='species',clusts='clusters',
                      clust1='Human', clust2='Mouse', out=NULL, width=16, height=40,
                      label.cex=2, min.frac=0, river.yscale=8,
                      river.node_margin=2, ...){
    clust1 = metedf[metedf[,groups]==clust1,]
    cluster1 = droplevels(unlist(clust1[, compare]))
    names(cluster1)=rownames(clust1)

    clust2 = metedf[metedf[,groups]==clust2,]
    cluster2 = droplevels(unlist(clust2[, compare]))
    names(cluster2)=rownames(clust2)

    cluster_consensus = droplevels(unlist(metedf[, clusts]))
    names(cluster_consensus)=rownames(metedf)

    if (!is.null(nodeorder)){
        node.order=nodeorder
    }else if(!is.null(orderdf)){
        #order1 = unlist(lapply(unlist(unique(orderdf[,compare])), function(x) which(levels(cluster1)==x)))
        order1 = 1:length(levels(cluster1))
        #order2 = unlist(lapply(unlist(unique(orderdf[,compare])), function(x) which(levels(cluster2)==x)))
        order2 = 1:length(levels(cluster2))
        order0 = unlist(lapply(unlist(unique(orderdf[,clusts])), function(x) which(levels(cluster_consensus)==x)))
        node.order=list(order1, order0, order2)                   
    }else{
        node.order='auto'
    }
    if(!is.null(out)) pdf(file=out, width=width, height=height)
    doRiverplot(cluster1, cluster2, cluster_consensus,
                  node.order=node.order, label.cex=label.cex, 
                  min.frac=min.frac, river.yscale=river.yscale,
                  river.node_margin=river.node_margin, ...)
    if(!is.null(out)) dev.off()  
}                        

river_test <-function(){
    options(repr.plot.height = 24, repr.plot.width = 18)
    meta.df = seur.combined@meta.data

    header <- 'HumanMacaca'
    compare <- 'CellType'                     
    groups <- 'species'
    clust1 <- 'human'
    clust2 <- 'mouse'
    Clusters = 'integrated_snn_res.1'

    KK = CellState(meta.df, compare=compare, groups=groups, clusts=Clusters)[[2]]    
    ClustOrder = unlist(lapply(unlist(unique(KK[, Clusters])), 
                               function(x) which(levels(KK[, Clusters])==x)))
    RiverPlot(meta.df,
              orderdf=KK, 
              compare=compare,
              groups=groups,
              clusts=Clusters,
              clust1=clust1,
              clust2=clust2,

              out=paste(header, compare,groups,Clusters,'riverplot.pdf', sep='.'),
              min.cells = 10, 
              width=18,
              height=24) 
}                          

                           
make_split <- function(df, split.by, node.by, group.by = 'clusters'){
    `%||%` <- Seurat:::`%||%`
    df <- data.frame(df[,c(split.by, node.by, group.by)])
    #df <- data.frame(lapply(df, as.factor), stringsAsFactors=TRUE)
    idents <- levels(df[, split.by]) %||%  sort(unique(df[, split.by]))
    nodes  <- levels(df[, node.by])  %||%  sort(unique(df[, node.by]))
    groups <- levels(df[, group.by])  %||% sort(unique(df[, group.by]))
    ident.1 <- idents[1]
    ident.2 <- idents[2]

    suppressPackageStartupMessages(library(dplyr))
    suppressPackageStartupMessages(library(ggsankey))
    detach("package:dplyr", unload = TRUE)
    suppressPackageStartupMessages(library(dplyr))
    
    df1 <- df %>% filter(!!rlang::sym(split.by)==ident.1) %>%
                dplyr::group_by(!!rlang::sym(group.by)) %>%
                dplyr::mutate(numbering = row_number())
    df2 <- df %>% filter(!!rlang::sym(split.by)==ident.2) %>% 
                dplyr::group_by(!!rlang::sym(group.by)) %>%
                dplyr::mutate(numbering = row_number())
    
    DD <- merge(df1, df2, by=c(group.by, 'numbering'), all.x = T, all.y = T, sort=F)
    DD <- DD[c(sprintf('%s.x', node.by),  sprintf('%s.y', node.by), group.by)]
    colnames(DD) <- c(ident.1, ident.2, group.by)
    
    sangkdf <- DD %>% make_long(!!rlang::sym(ident.1), 
                                !!rlang::sym(group.by), 
                                !!rlang::sym(ident.2)) %>% data.frame()
    sangkdf <- sangkdf[!(is.na(sangkdf$node)), ] # remove blank
    rownames(sangkdf) <-NULL

    sangkdf$x <- factor(sangkdf$x, levels=c(ident.1, group.by, ident.2))
    sangkdf$next_x <- factor(sangkdf$next_x, levels=c(ident.1, group.by, ident.2))
    sangkdf$node <- factor(sangkdf$node, levels=c(nodes, groups))
    sangkdf$next_node <- factor(sangkdf$next_node, 
                                           levels=c(nodes, groups))
    return(sangkdf)
}

ggsankey.plot0<-function(sangkdf, show=FALSE, save=NULL,
                        cmap = NULL, label_size=5,  smooth=8, space=NULL, 
                        width=18,  height=24){
    #sangkdf$hjust <- ifelse(sangkdf$x=='human', 1.3, ifelse(sangkdf$x=='mouse', -0.5, 0.) )
    pp <- ggplot(sangkdf, aes(x = x, 
                   next_x = next_x, 
                   node = node, 
                   next_node = next_node,
                   label = node,
                   #hjust =hjust,
                   fill = node)
                  ) +
      geom_sankey(flow.alpha = 0.6, node.color = "gray30", show.legend=T, smooth=smooth, space=space) + 
      geom_sankey_label(size = label_size, color = "white", fill = "gray40", space=space) +
      #scale_fill_viridis_d(option = "viridis") +
      #scale_fill_manual(values = c('VEGFC' = "red",NRP2="red")
      theme_sankey(base_size = 16) +
      labs(x = NULL) +
      theme(legend.position = "none")
            #plot.title = element_text(hjust =0))

    if (show) pp
    if (!is.null(save)) ggsave(save, pp, width=width, height=height)
    return(pp)
}

ggsankey.plot<-function(sangkdf, show=FALSE, save=NULL,
                        colors =NULL, legend.position='none',
                        legend.size=3,
                        gncol=1,
                        show_label=TRUE, 
                        cmap = NULL, label_size=4,  smooth=8, space=NULL, 
                        width=18,  height=24){
    #sangkdf$hjust <- ifelse(sangkdf$x=='human', 1.3, ifelse(sangkdf$x=='mouse', -0.5, 0.) )
    
    if (is.null(colors)){
        node_len = length(na.omit(unique(c(as.character(df$node), as.character(df$next_node)))))
        colors = viridis::viridis(n=node_len)
    }
    pp <- ggplot(sangkdf, aes(x = x, 
                   next_x = next_x, 
                   node = node, 
                   next_node = next_node,
                   label = node,
                   #hjust =hjust,
                   fill = node)
                  ) +
      geom_sankey(flow.alpha = 0.6, node.color = "gray30",
                  show.legend=T, smooth=smooth, space=space)
     if (show_label){
         pp <- pp + geom_sankey_label(size = label_size, color = "white", 
                                      fill = "gray20", space=space, hjust = 'left')
     }
     pp <- pp + 
            #scale_fill_viridis_d(option = "viridis") +
              scale_fill_manual(values = colors) + 
              theme_sankey(base_size = 16) +
              labs(x = NULL) +
              guides(color = guide_legend(override.aes = list(size = legend.size)), 
                       fill=guide_legend(ncol=gncol)) +
              theme(legend.position = legend.position)
                #plot.title = element_text(hjust =0))

    if (show) pp
    if (!is.null(save)) ggsave(save, pp, width=width, height=height)
    return(pp)
}                           
                               
ggalluvial.plot <-function(sangkdf, show=FALSE, save=NULL,
                        width=18,  height=24){

    pp <- ggplot(sangkdf, aes(x = x, 
                   next_x = next_x, 
                   node = node, 
                   next_node = next_node,
                   label = node,
                   #hjust =hjust,
                   fill = node)
                  ) +
      geom_alluvial(flow.alpha = .6) +
      geom_alluvial_text(size = 3, color = "white") +

      #geom_sankey(flow.alpha = 0.6, node.color = "gray30") + 
      #geom_sankey_label( size = 4, color = "white", fill = "gray40") +
      scale_fill_viridis_d() +
      #theme_sankey(base_size = 10) +
      theme_alluvial(base_size = 18) +
      labs(x = NULL) +
      theme(legend.position = "right",
            plot.title = element_text(hjust =0))

    if (show) pp
    if (!is.null(save)) ggsave(save, pp, width=width, height=height)
    return(pp)
}

doRiverplot <- function (cluster1, cluster2, cluster_consensus, 
    min.frac = 0.05, min.cells = 10, river.yscale = 1, river.lty = 0, 
    river.node_margin = 0.1, label.cex = 1, label.col = "black",
    col.left =NULL, col.middle =NULL, col.rigth=NULL,
     plot_area = c(0.85, 0.95),
    lab.srt = 0, river.usr = NULL, node.order = "auto", ...) 
{
    cluster1 <- droplevels(cluster1)
    cluster2 <- droplevels(cluster2)

    if (length(intersect(levels(cluster1), levels(cluster2))) > 
        0 | length(intersect(levels(cluster1), levels(cluster_consensus))) > 
        0 | length(intersect(levels(cluster2), levels(cluster_consensus))) > 
        0) {
        message("Duplicate cluster names detected. Adding 1- and 2- to make unique names.")
        cluster1 <-plyr::mapvalues(cluster1, from = levels(cluster1), 
            to = paste("1", levels(cluster1), sep = "-"))
        cluster2 <- plyr::mapvalues(cluster2, from = levels(cluster2), 
            to = paste("2", levels(cluster2), sep = "-"))
    }

    cluster1 <- cluster1[intersect(names(cluster1), names(cluster_consensus))]
    cluster2 <- cluster2[intersect(names(cluster2), names(cluster_consensus))]

    if (identical(node.order, "auto")) {
        tab.1 <- table(cluster1, cluster_consensus[names(cluster1)])
        tab.1 <- sweep(tab.1, 1, rowSums(tab.1), "/")
        tab.2 <- table(cluster2, cluster_consensus[names(cluster2)])
        tab.2 <- sweep(tab.2, 1, rowSums(tab.2), "/")
        whichmax.1 <- apply(tab.1, 1, which.max)
        whichmax.2 <- apply(tab.2, 1, which.max)
        ord.1 <- order(whichmax.1)
        ord.2 <- order(whichmax.2)
        cluster1 <- factor(cluster1, levels = levels(cluster1)[ord.1])
        cluster2 <- factor(cluster2, levels = levels(cluster2)[ord.2])
    }
    else {
        if (is.list(node.order)) {
            cluster1 <- factor(cluster1, levels = levels(cluster1)[node.order[[1]]])
            cluster_consensus <- factor(cluster_consensus, levels = levels(cluster_consensus)[node.order[[2]]])
            cluster2 <- factor(cluster2, levels = levels(cluster2)[node.order[[3]]])
        }
    }
    cluster1 <- cluster1[!is.na(cluster1)]
    cluster2 <- cluster2[!is.na(cluster2)]
    nodes1 <- levels(cluster1)[table(cluster1) > 0]
    nodes2 <- levels(cluster2)[table(cluster2) > 0]
    nodes_middle <- levels(cluster_consensus)[table(cluster_consensus) > 0]
    node_Xs <- c(rep(1, length(nodes1)), rep(2, length(nodes_middle)), 
        rep(3, length(nodes2)))
    edge_list <- list()
    for (i in 1:length(nodes1)) {
        temp <- list()
        i_cells <- names(cluster1)[cluster1 == nodes1[i]]
        for (j in 1:length(nodes_middle)) {
            if (length(which(cluster_consensus[i_cells] == nodes_middle[j]))/length(i_cells) > 
                min.frac & length(which(cluster_consensus[i_cells] == 
                nodes_middle[j])) > min.cells) {
                temp[[nodes_middle[j]]] <- sum(cluster_consensus[i_cells] == 
                  nodes_middle[j])/length(cluster1)
            }
        }
        edge_list[[nodes1[i]]] <- temp
    }
    cluster3 <- cluster_consensus[names(cluster2)]
    for (i in 1:length(nodes_middle)) {
        temp <- list()
        i_cells <- names(cluster3)[cluster3 == nodes_middle[i]]
        for (j in 1:length(nodes2)) {
            j_cells <- names(cluster2)[cluster2 == nodes2[j]]
            if (length(which(cluster_consensus[j_cells] == nodes_middle[i]))/length(j_cells) > 
                min.frac & length(which(cluster_consensus[j_cells] == 
                nodes_middle[i])) > min.cells) {
                if (!is.na(sum(cluster2[i_cells] == nodes2[j]))) {
                  temp[[nodes2[j]]] <- sum(cluster2[i_cells] == 
                    nodes2[j])/length(cluster2)
                }
            }
        }
        edge_list[[nodes_middle[i]]] <- temp
    }
    node_cols <- list()
    ggplotColors <- function(g) {
        d <- 360/g
        h <- cumsum(c(15, rep(d, g - 1)))
        grDevices::hcl(h = h, c = 100, l = 65)
    }

    if (!is.null(col.left)){
        pal1 <- col.left[nodes1]
    }else{
        pal1 <- ggplotColors(length(nodes1))
    }
    for (i in 1:length(nodes1)) {
        node_cols[[nodes1[i]]] <- list(col = pal1[i], textcex = label.cex, 
            textcol = label.col, srt = lab.srt)
    }

    if (!is.null(col.middle)){
        pal2 <- col.middle[nodes_middle]
    }else{
        pal2 <- ggplotColors(length(nodes_middle))
    }
    for (i in 1:length(nodes_middle)) {
        node_cols[[nodes_middle[i]]] <- list(col = pal2[i], textcex = label.cex, 
            textcol = label.col, srt = lab.srt)
    }

    if (!is.null(col.rigth)){
        pal3 <- col.rigth[nodes2]
    }else{
        pal3 <- ggplotColors(length(nodes2))
    }
    for (i in 1:length(nodes2)) {
        node_cols[[nodes2[i]]] <- list(col = pal3[i], textcex = label.cex, 
            textcol = label.col, srt = lab.srt)
    }
    nodes <- list(nodes1, nodes_middle, nodes2)
    node.limit <- max(unlist(lapply(nodes, length)))
    node_Ys <- lapply(1:length(nodes), function(i) {
        seq(1, node.limit, by = node.limit/length(nodes[[i]]))
    })
    
    library(riverplot)
    rp <- makeRiver(c(nodes1, nodes_middle, nodes2), edge_list, 
        node_xpos = node_Xs, node_ypos = unlist(node_Ys), node_styles = node_cols)
    invisible(capture.output(riverplot(rp, yscale = river.yscale, lty = river.lty, fix.pdf = TRUE,
                                       node_margin = river.node_margin, usr = river.usr, plot_area = plot_area)))
}
