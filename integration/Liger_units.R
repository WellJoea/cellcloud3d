library(rliger)
library(riverplot)

CellState <-function(LigerObj, compare='CellType',groups='species', clusts='clusters'){
    LigerObj@cell.data[, clusts] = LigerObj@clusters
    GG = LigerObj@cell.data[c(clusts, groups, compare)] %>%
                group_by_( compare, clusts, groups) %>%
                summarise(countS = n()) %>%
                mutate(countC =sum(countS), maxC=max(countS)) %>%
                ungroup(!!rlang::sym(clusts)) %>%
                mutate(countL =sum(countS)) %>%
                ungroup(!!rlang::sym(compare)) %>%
                mutate(countA =sum(countS)) %>%
                mutate(freqP = countS/countC, freqS = countS/countL, freqC = countC/countL, freqL = countL/countA) %>%
                arrange(desc(freqL), desc(freqC), desc(countS))
    KK = GG %>% 
        group_by(!!rlang::sym(clusts)) %>% 
        slice_max(order_by = maxC, n=1) %>%
        arrange(!!rlang::sym(compare), countL, countC)
    
    return(list(GG, KK))
}

RiverPlot <- function(LigerObj, orderdf=NULL, nodeorder=NULL, 
                      compare='CellType', groups='species',clusts='clusters',
                      clust1='Human', clust2='Mouse', out=NULL, width=16, height=40,
                      label.cex=2, min.frac=0, river.yscale=8,
                      river.node_margin=2, ...){
    clust1 = LigerObj@cell.data[LigerObj@cell.data[,groups]==clust1,]
    cluster1 = droplevels(unlist(clust1[, compare]))
    names(cluster1)=rownames(clust1)
    
    clust2 = LigerObj@cell.data[LigerObj@cell.data[,groups]==clust2,]
    cluster2 = droplevels(unlist(clust2[, compare]))
    names(cluster2)=rownames(clust2)

    cluster_consensus=LigerObj@clusters
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
    makeRiverplotN(LigerObj, cluster1, cluster2, cluster_consensus=cluster_consensus,
                  node.order=node.order, label.cex=label.cex, 
                  min.frac=min.frac, river.yscale=river.yscale,
                  river.node_margin=river.node_margin, ...)
    if(!is.null(out)) dev.off()  
}                        

makeRiverplotN <- function (object, cluster1, cluster2, cluster_consensus = NULL, 
    min.frac = 0.05, min.cells = 10, river.yscale = 1, river.lty = 0, 
    river.node_margin = 0.1, label.cex = 1, label.col = "black",
     plot_area = c(0.85, 0.9),
    lab.srt = 0, river.usr = NULL, node.order = "auto", ...) 
{
    cluster1 <- droplevels(cluster1)
    cluster2 <- droplevels(cluster2)
    if (is.null(cluster_consensus)) {
        cluster_consensus <- droplevels(object@clusters)
    }
    if (length(intersect(levels(cluster1), levels(cluster2))) > 
        0 | length(intersect(levels(cluster1), levels(cluster_consensus))) > 
        0 | length(intersect(levels(cluster2), levels(cluster_consensus))) > 
        0) {
        message("Duplicate cluster names detected. Adding 1- and 2- to make unique names.")
        cluster1 <- mapvalues(cluster1, from = levels(cluster1), 
            to = paste("1", levels(cluster1), sep = "-"))
        cluster2 <- mapvalues(cluster2, from = levels(cluster2), 
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
    nodes_middle <- levels(cluster_consensus)[table(cluster_consensus) > 
        0]
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
    pal <- ggplotColors(length(nodes1))
    for (i in 1:length(nodes1)) {
        node_cols[[nodes1[i]]] <- list(col = pal[i], textcex = label.cex, 
            textcol = label.col, srt = lab.srt)
    }
    pal <- ggplotColors(length(nodes_middle))
    for (i in 1:length(nodes_middle)) {
        node_cols[[nodes_middle[i]]] <- list(col = pal[i], textcex = label.cex, 
            textcol = label.col, srt = lab.srt)
    }
    pal <- ggplotColors(length(nodes2))
    for (i in 1:length(nodes2)) {
        node_cols[[nodes2[i]]] <- list(col = pal[i], textcex = label.cex, 
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
    invisible(capture.output(riverplot(rp, yscale = river.yscale, lty = river.lty, 
        node_margin = river.node_margin, usr = river.usr, plot_area = plot_area)))
}