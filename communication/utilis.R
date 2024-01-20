net_computeCentrality <- function(object = NULL, slot.name = "netP", net = NULL, net.name = NULL, thresh = 0.05) {
  if (is.null(net)) {
    prob <- methods::slot(object, slot.name)$prob
    pval <- methods::slot(object, slot.name)$pval
    pval[prob == 0] <- 1
    prob[pval >= thresh] <- 0
    net = prob
  }

  if (is.null(net.name)) {
    net.name <- dimnames(net)[[3]]
  }

  if (length(dim(net)) == 3) {
    nrun <- dim(net)[3]
    my.sapply <- ifelse(
      test = future::nbrOfWorkers() == 1,
      yes = pbapply::pbsapply,
      no = future.apply::future_sapply
    )
    centr.all = my.sapply(
      X = 1:nrun,
      FUN = function(x) {
        net0 <- net[ , , x]
        return(CellChat:::computeCentralityLocal(net0))
      },
      simplify = FALSE
    )
  } else {
    centr.all <- as.list(CellChat:::computeCentralityLocal(net))
  }
  names(centr.all) <- net.name
  if (is.null(object)) {
    return(centr.all)
  } else {
    slot(object, slot.name)$centr <- centr.all
    return(object)
  }
}
