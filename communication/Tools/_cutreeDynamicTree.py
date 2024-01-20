
def cutreeHybrid(dendro, distM, cutHeight=None, minClusterSize=20, deepSplit=1,
                    maxCoreScatter=None, minGap=None, maxAbsCoreScatter=None,
                    minAbsGap=None, minSplitHeight=None, minAbsSplitHeight=None,
                    externalBranchSplitFnc=None, nExternalSplits=0, minExternalSplit=None,
                    externalSplitOptions=pd.DataFrame(), externalSplitFncNeedsDistance=None,
                    assumeSimpleExternalSpecification=True, pamStage=True,
                    pamRespectsDendro=True, useMedoids=False, maxPamDist=None,
                    respectSmallClusters=True):
    """
    Detect clusters in a dendorgram produced by the function hclust.

    :param dendro: a hierarchical clustering dendorgram such as one returned by hclust.
    :type dendro: ndarray
    :param distM: Distance matrix that was used as input to hclust.
    :type distM: pandas dataframe
    :param cutHeight: Maximum joining heights that will be considered. It defaults to 99of the range between the 5th percentile and the maximum of the joining heights on the dendrogram.
    :type cutHeight: int
    :param minClusterSize: Minimum cluster size. (default = 20)
    :type minClusterSize: int
    :param deepSplit: Either logical or integer in the range 0 to 4. Provides a rough control over sensitivity to cluster splitting. The higher the value, the more and smaller clusters will be produced. (default = 1)
    :type deepSplit: int or bool
    :param maxCoreScatter: Maximum scatter of the core for a branch to be a cluster, given as the fraction of cutHeight relative to the 5th percentile of joining heights.
    :type maxCoreScatter: int
    :param minGap: Minimum cluster gap given as the fraction of the difference between cutHeight and the 5th percentile of joining heights.
    :type minGap: int
    :param maxAbsCoreScatter: Maximum scatter of the core for a branch to be a cluster given as absolute heights. If given, overrides maxCoreScatter.
    :type maxAbsCoreScatter: int
    :param minAbsGap: Minimum cluster gap given as absolute height difference. If given, overrides minGap.
    :type minAbsGap: int
    :param minSplitHeight: Minimum split height given as the fraction of the difference between cutHeight and the 5th percentile of joining heights. Branches merging below this height will automatically be merged. Defaults to zero but is used only if minAbsSplitH
    :type minSplitHeight: int
    :param minAbsSplitHeight: Minimum split height given as an absolute height. Branches merging below this height will automatically be merged. If not given (default), will be determined from minSplitHeight above.
    :type minAbsSplitHeight: int
    :param externalBranchSplitFnc: Optional function to evaluate split (dissimilarity) between two branches. Either a single function or a list in which each component is a function.
    :param minExternalSplit: Thresholds to decide whether two branches should be merged. It should be a numeric list of the same length as the number of functions in externalBranchSplitFnc above.
    :type minExternalSplit: list
    :param externalSplitOptions: Further arguments to function externalBranchSplitFnc. If only one external function is specified in externalBranchSplitFnc above, externalSplitOptions can be a named list of arguments or a list with one component.
    :type externalSplitOptions: pandas dataframe
    :param externalSplitFncNeedsDistance: Optional specification of whether the external branch split functions need the distance matrix as one of their arguments. Either NULL or a logical list with one element per branch
    :type externalSplitFncNeedsDistance: pandas dataframe
    :param assumeSimpleExternalSpecification: when minExternalSplit above is a scalar (has length 1), should the function assume a simple specification of externalBranchSplitFnc and externalSplitOptions. (default = True)
    :type assumeSimpleExternalSpecification: bool
    :param pamStage: If TRUE, the second (PAM-like) stage will be performed. (default = True)
    :type pamStage: bool
    :param pamRespectsDendro: If TRUE, the PAM stage will respect the dendrogram in the sense an object can be PAM-assigned only to clusters that lie below it on the branch that the object is merged into. (default = True)
    :type pamRespectsDendro: bool
    :param useMedoids: if TRUE, the second stage will be use object to medoid distance; if FALSE, it will use average object to cluster distance. (default = False)
    :param maxPamDist: Maximum object distance to closest cluster that will result in the object assigned to that cluster. Defaults to cutHeight.
    :type maxPamDist: float
    :param respectSmallClusters: If TRUE, branches that failed to be clusters in stage 1 only because of insufficient size will be assigned together in stage 2. If FALSE, all objects will be assigned individually. (default = False)
    :type respectSmallClusters: bool

    :return: list detailing the deteced branch structure.
    :rtype: list
    """
    tmp = dendro[:, 0] > dendro.shape[0]
    dendro[tmp, 0] = dendro[tmp, 0] - dendro.shape[0]
    dendro[np.logical_not(tmp), 0] = -1 * (dendro[np.logical_not(tmp), 0] + 1)
    tmp = dendro[:, 1] > dendro.shape[0]
    dendro[tmp, 1] = dendro[tmp, 1] - dendro.shape[0]
    dendro[np.logical_not(tmp), 1] = -1 * (dendro[np.logical_not(tmp), 1] + 1)

    chunkSize = dendro.shape[0]

    if maxPamDist is None:
        maxPamDist = cutHeight

    nMerge = dendro.shape[0]
    if nMerge < 1:
        sys.exit("The given dendrogram is suspicious: number of merges is zero.")
    if distM is None:
        sys.exit("distM must be non-NULL")
    if distM.shape is None:
        sys.exit("distM must be a matrix.")
    if distM.shape[0] != nMerge + 1 or distM.shape[1] != nMerge + 1:
        sys.exit("distM has incorrect dimensions.")
    if pamRespectsDendro and not respectSmallClusters:
        print("cutreeHybrid Warning: parameters pamRespectsDendro (TRUE) "
                "and respectSmallClusters (FALSE) imply contradictory intent.\n"
                "Although the code will work, please check you really intented "
                "these settings for the two arguments.", flush=True)

    print(f"{OKCYAN}Going through the merge tree...{ENDC}")

    if any(np.diag(distM) != 0):
        np.fill_diagonal(distM, 0)
    refQuantile = 0.05
    refMerge = round(nMerge * refQuantile) - 1
    if refMerge < 0:
        refMerge = 0
    refHeight = dendro[refMerge, 2]
    if cutHeight is None:
        cutHeight = 0.99 * (np.max(dendro[:, 2]) - refHeight) + refHeight
        print("..cutHeight not given, setting it to", round(cutHeight, 3),
                " ===>  99% of the (truncated) height range in dendro.", flush=True)
    else:
        if cutHeight > np.max(dendro[:, 2]):
            cutHeight = np.max(dendro[:, 2])
    if maxPamDist is None:
        maxPamDist = cutHeight
    nMergeBelowCut = np.count_nonzero(dendro[:, 2] <= cutHeight)
    if nMergeBelowCut < minClusterSize:
        print("cutHeight set too low: no merges below the cut.", flush=True)
        return pd.DataFrame({'labels': np.repeat(0, nMerge + 1, axis=0)})

    if externalBranchSplitFnc is not None:
        nExternalSplits = len(externalBranchSplitFnc)
        if len(minExternalSplit) < 1:
            sys.exit("'minExternalBranchSplit' must be given.")
        if assumeSimpleExternalSpecification and nExternalSplits == 1:
            externalSplitOptions = pd.DataFrame(externalSplitOptions)
        # TODO: externalBranchSplitFnc = lapply(externalBranchSplitFnc, match.fun)
        for es in range(nExternalSplits):
            externalSplitOptions['tree'][es] = dendro
            if len(externalSplitFncNeedsDistance) == 0 or externalSplitFncNeedsDistance[es]:
                externalSplitOptions['dissimMat'][es] = distM

    MxBranches = nMergeBelowCut
    branch_isBasic = np.repeat(True, MxBranches, axis=0)
    branch_isTopBasic = np.repeat(True, MxBranches, axis=0)
    branch_failSize = np.repeat(False, MxBranches, axis=0)
    branch_rootHeight = np.repeat(np.nan, MxBranches, axis=0)
    branch_size = np.repeat(2, MxBranches, axis=0)
    branch_nMerge = np.repeat(1, MxBranches, axis=0)
    branch_nSingletons = np.repeat(2, MxBranches, axis=0)
    branch_nBasicClusters = np.repeat(0, MxBranches, axis=0)
    branch_mergedInto = np.repeat(0, MxBranches, axis=0)
    branch_attachHeight = np.repeat(np.nan, MxBranches, axis=0)
    branch_singletons = pd.DataFrame()
    branch_basicClusters = pd.DataFrame()
    branch_mergingHeights = pd.DataFrame()
    branch_singletonHeights = pd.DataFrame()
    nBranches = -1

    defMCS = [0.64, 0.73, 0.82, 0.91, 0.95]
    defMG = [(1.0 - defMC) * 3.0 / 4.0 for defMC in defMCS]
    nSplitDefaults = len(defMCS)
    if isinstance(deepSplit, bool):
        deepSplit = pd.to_numeric(deepSplit) * (nSplitDefaults - 2)
    if deepSplit < 0 or deepSplit > nSplitDefaults:
        msg = "Parameter deepSplit (value" + str(deepSplit) + \
                ") out of range: allowable range is 0 through", str(nSplitDefaults - 1)
        sys.exit(msg)

    if maxCoreScatter is None:
        maxCoreScatter = WGCNA.interpolate(defMCS, deepSplit)
    if minGap is None:
        minGap = WGCNA.interpolate(defMG, deepSplit)
    if maxAbsCoreScatter is None:
        maxAbsCoreScatter = refHeight + maxCoreScatter * (cutHeight - refHeight)
    if minAbsGap is None:
        minAbsGap = minGap * (cutHeight - refHeight)
    if minSplitHeight is None:
        minSplitHeight = 0
    if minAbsSplitHeight is None:
        minAbsSplitHeight = refHeight + minSplitHeight * (cutHeight - refHeight)
    nPoints = nMerge + 1
    IndMergeToBranch = np.repeat(-1, nMerge, axis=0)
    onBranch = np.repeat(0, nPoints, axis=0)
    RootBranch = 0

    mergeDiagnostics = pd.DataFrame({'smI': np.repeat(np.nan, nMerge, axis=0),
                                        'smSize': np.repeat(np.nan, nMerge, axis=0),
                                        'smCrSc': np.repeat(np.nan, nMerge, axis=0),
                                        'smGap': np.repeat(np.nan, nMerge, axis=0),
                                        'lgI': np.repeat(np.nan, nMerge, axis=0),
                                        'lgSize': np.repeat(np.nan, nMerge, axis=0),
                                        'lgCrSc': np.repeat(np.nan, nMerge, axis=0),
                                        'lgGap': np.repeat(np.nan, nMerge, axis=0),
                                        'merged': np.repeat(np.nan, nMerge, axis=0)})
    if externalBranchSplitFnc is not None:
        externalMergeDiags = pd.DataFrame(np.nan, index=list(range(nMerge)), columns=list(range(nExternalSplits)))

    extender = np.repeat(0, chunkSize, axis=0)

    for merge in range(nMerge):
        if dendro[merge, 2] <= cutHeight:
            if dendro[merge, 0] < 0 and dendro[merge, 1] < 0:
                nBranches = nBranches + 1
                branch_isBasic[nBranches] = True
                branch_isTopBasic[nBranches] = True
                branch_singletons.insert(nBranches, nBranches,
                                            np.concatenate((-1 * dendro[merge, 0:2], extender), axis=0))
                branch_basicClusters.insert(nBranches, nBranches, extender)
                branch_mergingHeights.insert(nBranches, nBranches,
                                                np.concatenate((np.repeat(dendro[merge, 2], 2), extender), axis=0))
                branch_singletonHeights.insert(nBranches, nBranches,
                                                np.concatenate((np.repeat(dendro[merge, 2], 2), extender), axis=0))
                IndMergeToBranch[merge] = nBranches
                RootBranch = nBranches
            elif np.sign(dendro[merge, 0]) * np.sign(dendro[merge, 1]) < 0:
                clust = IndMergeToBranch[int(np.max(dendro[merge, 0:2])) - 1]

                if clust == -1:
                    sys.exit("Internal error: a previous merge has no associated cluster. Sorry!")

                gene = -1 * int(np.min(dendro[merge, 0:2]))
                ns = branch_nSingletons[clust]
                nm = branch_nMerge[clust]

                if branch_isBasic[clust]:
                    branch_singletons.loc[ns, clust] = gene
                    branch_singletonHeights.loc[ns, clust] = dendro[merge, 2]
                else:
                    onBranch[int(gene)] = clust

                branch_mergingHeights.loc[nm, clust] = dendro[merge, 2]
                branch_size[clust] = branch_size[clust] + 1
                branch_nMerge[clust] = nm + 1
                branch_nSingletons[clust] = ns + 1
                IndMergeToBranch[merge] = clust
                RootBranch = clust
            else:
                clusts = IndMergeToBranch[dendro[merge, 0:2].astype(int) - 1]
                sizes = branch_size[clusts]
                rnk = np.argsort(sizes)
                small = clusts[rnk[0]]
                large = clusts[rnk[1]]
                sizes = sizes[rnk]

                if branch_isBasic[small]:
                    coresize = WGCNA.coreSizeFunc(branch_nSingletons[small], minClusterSize) - 1
                    Core = branch_singletons.loc[0:coresize, small] - 1
                    Core = Core.astype(int).tolist()
                    SmAveDist = np.mean(distM.iloc[Core, Core].sum() / coresize)
                else:
                    SmAveDist = 0

                if branch_isBasic[large]:
                    coresize = WGCNA.coreSizeFunc(branch_nSingletons[large], minClusterSize) - 1
                    Core = branch_singletons.loc[0:coresize, large] - 1
                    Core = Core.astype(int).tolist()
                    LgAveDist = np.mean(distM.iloc[Core, Core].sum() / coresize)
                else:
                    LgAveDist = 0

                mergeDiagnostics.loc[merge, :] = [small, branch_size[small], SmAveDist,
                                                    dendro[merge, 2] - SmAveDist,
                                                    large, branch_size[large], LgAveDist,
                                                    dendro[merge, 2] - LgAveDist,
                                                    None]
                SmallerScores = [branch_isBasic[small], branch_size[small] < minClusterSize,
                                    SmAveDist > maxAbsCoreScatter, dendro[merge, 2] - SmAveDist < minAbsGap,
                                    dendro[merge, 2] < minAbsSplitHeight]
                if SmallerScores[0] * np.count_nonzero(SmallerScores[1:]) > 0:
                    DoMerge = True
                    SmallerFailSize = not (SmallerScores[2] | SmallerScores[3])
                else:
                    LargerScores = [branch_isBasic[large],
                                    branch_size[large] < minClusterSize, LgAveDist > maxAbsCoreScatter,
                                    dendro[merge, 2] - LgAveDist < minAbsGap,
                                    dendro[merge, 2] < minAbsSplitHeight]
                    if LargerScores[0] * np.count_nonzero(LargerScores[1:]) > 0:
                        DoMerge = True
                        SmallerFailSize = not (LargerScores[2] | LargerScores[3])
                        x = small
                        small = large
                        large = x
                        sizes = np.flip(sizes)
                    else:
                        DoMerge = False

                if DoMerge:
                    mergeDiagnostics['merged'][merge] = 1

                if not DoMerge and nExternalSplits > 0 and branch_isBasic[small] and branch_isBasic[large]:
                    branch1 = branch_singletons[[large]][0:sizes[1]]
                    branch2 = branch_singletons[[small]][0:sizes[0]]
                    es = 0
                    while es < nExternalSplits and not DoMerge:
                        es = es + 1
                        args = pd.DataFrame({'externalSplitOptions': externalSplitOptions[[es]],
                                                'branch1': branch1, 'branch2': branch2})
                        # TODO: extSplit = do.call(externalBranchSplitFnc[[es]], args)
                        extSplit = None
                        DoMerge = extSplit < minExternalSplit[es]
                        externalMergeDiags[merge, es] = extSplit
                        mergeDiagnostics['merged'][merge] = 0
                        if DoMerge:
                            mergeDiagnostics['merged'][merge] = 2

                if DoMerge:
                    branch_failSize[[small]] = SmallerFailSize
                    branch_mergedInto[small] = large + 1
                    branch_attachHeight[small] = dendro[merge, 2]
                    branch_isTopBasic[small] = False
                    nss = branch_nSingletons[small] - 1
                    nsl = branch_nSingletons[large]
                    ns = nss + nsl
                    if branch_isBasic[large]:
                        branch_singletons.loc[nsl:ns, large] = branch_singletons.loc[0:nss, small].values
                        branch_singletonHeights.loc[nsl:ns, large] = branch_singletonHeights.loc[0:nss,
                                                                        small].values
                        branch_nSingletons[large] = ns + 1
                    else:
                        if not branch_isBasic[small]:
                            sys.exit("Internal error: merging two composite clusters. Sorry!")
                        tmp = branch_singletons[[small]].astype(int).values
                        tmp = tmp[tmp != 0]
                        tmp = tmp - 1
                        onBranch[tmp] = large + 1

                    nm = branch_nMerge[large]
                    branch_mergingHeights.loc[nm, large] = dendro[merge, 2]
                    branch_nMerge[large] = nm + 1
                    branch_size[large] = branch_size[small] + branch_size[large]
                    IndMergeToBranch[merge] = large
                    RootBranch = large
                else:
                    if branch_isBasic[large] and not branch_isBasic[small]:
                        x = large
                        large = small
                        small = x
                        sizes = np.flip(sizes)

                    if branch_isBasic[large] or (pamStage and pamRespectsDendro):
                        nBranches = nBranches + 1
                        branch_attachHeight[[large, small]] = dendro[merge, 2]
                        branch_mergedInto[[large, small]] = nBranches
                        if branch_isBasic[small]:
                            addBasicClusters = [small + 1]
                        else:
                            addBasicClusters = branch_basicClusters.loc[
                                (branch_basicClusters[[small]] != 0).all(axis=1), small]
                        if branch_isBasic[large]:
                            addBasicClusters = np.concatenate((addBasicClusters, [large + 1]), axis=0)
                        else:
                            addBasicClusters = np.concatenate((addBasicClusters,
                                                                branch_basicClusters.loc[(
                                                                                                branch_basicClusters[
                                                                                                    [
                                                                                                        large]] != 0).all(
                                                                    axis=1), large]),
                                                                axis=0)
                        branch_isBasic[nBranches] = False
                        branch_isTopBasic[nBranches] = False
                        branch_basicClusters.insert(nBranches, nBranches,
                                                    np.concatenate((addBasicClusters,
                                                                    np.repeat(0,
                                                                                chunkSize - len(addBasicClusters))),
                                                                    axis=0))
                        branch_singletons.insert(nBranches, nBranches, np.repeat(np.nan, chunkSize + 2))
                        branch_singletonHeights.insert(nBranches, nBranches, np.repeat(np.nan, chunkSize + 2))
                        branch_mergingHeights.insert(nBranches, nBranches,
                                                        np.concatenate((np.repeat(dendro[merge, 2], 2), extender),
                                                                    axis=0))
                        branch_nMerge[nBranches] = 2
                        branch_size[nBranches] = sum(sizes) + 2
                        branch_nBasicClusters[nBranches] = len(addBasicClusters)
                        IndMergeToBranch[merge] = nBranches
                        RootBranch = nBranches
                    else:
                        if branch_isBasic[small]:
                            addBasicClusters = [small + 1]
                        else:
                            addBasicClusters = branch_basicClusters.loc[
                                (branch_basicClusters[[small]] != 0).all(axis=1), small]

                        nbl = branch_nBasicClusters[large]
                        nb = branch_nBasicClusters[large] + len(addBasicClusters)
                        branch_basicClusters.iloc[nbl:nb, large] = addBasicClusters
                        branch_nBasicClusters[large] = nb
                        branch_size[large] = branch_size[large] + branch_size[small]
                        nm = branch_nMerge[large] + 1
                        branch_mergingHeights.loc[nm, large] = dendro[merge, 2]
                        branch_nMerge[large] = nm
                        branch_attachHeight[small] = dendro[merge, 2]
                        branch_mergedInto[small] = large + 1
                        IndMergeToBranch[merge] = large
                        RootBranch = large

    nBranches = nBranches + 1
    isCluster = np.repeat(False, nBranches)
    SmallLabels = np.repeat(0, nPoints)

    for clust in range(nBranches):
        if np.isnan(branch_attachHeight[clust]):
            branch_attachHeight[clust] = cutHeight
        if branch_isTopBasic[clust]:
            coresize = WGCNA.coreSizeFunc(branch_nSingletons[clust], minClusterSize)
            Core = branch_singletons.iloc[0:coresize, clust] - 1
            Core = Core.astype(int).tolist()
            CoreScatter = np.mean(distM.iloc[Core, Core].sum() / (coresize - 1))
            isCluster[clust] = (branch_isTopBasic[clust] and branch_size[clust] >= minClusterSize and
                                CoreScatter < maxAbsCoreScatter and branch_attachHeight[
                                    clust] - CoreScatter > minAbsGap)
        else:
            CoreScatter = 0
        if branch_failSize[clust]:
            SmallLabels[branch_singletons[[clust]].astype(int) - 1] = clust + 1

    if not respectSmallClusters:
        SmallLabels = np.repeat(0, nPoints)

    Colors = np.zeros((nPoints,))
    coreLabels = np.zeros((nPoints,))
    clusterBranches = np.where(isCluster)[0].tolist()
    branchLabels = np.zeros((nBranches,))
    color = 0

    for clust in clusterBranches:
        color = color + 1
        tmp = branch_singletons[[clust]].astype(int) - 1
        tmp = tmp[tmp != -1]
        tmp.dropna(inplace=True)
        tmp = tmp.iloc[:, 0].astype(int)
        Colors[tmp] = color
        SmallLabels[tmp] = 0
        coresize = WGCNA.coreSizeFunc(branch_nSingletons[clust], minClusterSize)
        Core = branch_singletons.loc[0:coresize, clust] - 1
        Core = Core.astype(int).tolist()
        coreLabels[Core] = color
        branchLabels[clust] = color

    Labeled = np.where(Colors != 0)[0].tolist()
    Unlabeled = np.where(Colors == 0)[0].tolist()
    nUnlabeled = len(Unlabeled)
    UnlabeledExist = nUnlabeled > 0

    if len(Labeled) > 0:
        LabelFac = pd.Categorical(Colors[Labeled])
        nProperLabels = len(LabelFac.categories)
    else:
        nProperLabels = 0

    if pamStage and UnlabeledExist and nProperLabels > 0:
        nPAMed = 0
        if useMedoids:
            Medoids = np.repeat(0, nProperLabels)
            ClusterRadii = np.repeat(0, nProperLabels)
            for cluster in range(nProperLabels):
                InCluster = np.where(Colors == cluster)[0].tolist()
                DistInCluster = distM.iloc[InCluster, InCluster]
                DistSums = DistInCluster.sum(axis=0)
                Medoids[cluster] = InCluster[DistSums.idxmin()]
                ClusterRadii[cluster] = np.max(DistInCluster[:, DistSums.idxmin()])

            if respectSmallClusters:
                FSmallLabels = pd.Categorical(SmallLabels)
                SmallLabLevs = pd.to_numeric(FSmallLabels.categories)
                nSmallClusters = len(FSmallLabels.categories) - (SmallLabLevs[1] == 0)

                if nSmallClusters > 0:
                    for sclust in SmallLabLevs[SmallLabLevs != 0]:
                        InCluster = np.where(SmallLabels == sclust)[0].tolist()
                        if pamRespectsDendro:
                            onBr = np.unique(onBranch[InCluster])
                            if len(onBr) > 1:
                                msg = "Internal error: objects in a small cluster are marked to belong\n " \
                                        "to several large branches:" + str(onBr)
                                sys.exit(msg)

                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[[onBr]]
                                labelsOnBranch = branchLabels[basicOnBranch]
                            else:
                                labelsOnBranch = None
                        else:
                            labelsOnBranch = list(range(nProperLabels))

                        DistInCluster = distM.iloc[InCluster, InCluster]

                        if len(labelsOnBranch) > 0:
                            if len(InCluster) > 1:
                                DistSums = DistInCluster.sum(axis=1)
                                smed = InCluster[DistSums.idxmin()]
                                DistToMeds = distM.iloc[Medoids[labelsOnBranch], smed]
                                closest = DistToMeds.idxmin()
                                DistToClosest = DistToMeds[closest]
                                closestLabel = labelsOnBranch[closest]
                                if DistToClosest < ClusterRadii[closestLabel] or DistToClosest < maxPamDist:
                                    Colors[InCluster] = closestLabel
                                    nPAMed = nPAMed + len(InCluster)
                            else:
                                Colors[InCluster] = -1
                        else:
                            Colors[InCluster] = -1

            Unlabeled = np.where(Colors == 0)[0].tolist()
            if len(Unlabeled > 0):
                for obj in Unlabeled:
                    if pamRespectsDendro:
                        onBr = onBranch[obj]
                        if onBr > 0:
                            basicOnBranch = branch_basicClusters[[onBr]]
                            labelsOnBranch = branchLabels[basicOnBranch]
                        else:
                            labelsOnBranch = None
                    else:
                        labelsOnBranch = list(range(nProperLabels))

                    if labelsOnBranch is not None:
                        UnassdToMedoidDist = distM.iloc[Medoids[labelsOnBranch], obj]
                        nearest = UnassdToMedoidDist.idxmin()
                        NearestCenterDist = UnassdToMedoidDist[nearest]
                        nearestMed = labelsOnBranch[nearest]
                        if NearestCenterDist < ClusterRadii[nearestMed] or NearestCenterDist < maxPamDist:
                            Colors[obj] = nearestMed
                            nPAMed = nPAMed + 1
                UnlabeledExist = (sum(Colors == 0) > 0)
        else:
            ClusterDiam = np.zeros((nProperLabels,))
            for cluster in range(nProperLabels):
                InCluster = np.where(Colors == (cluster + 1))[0].tolist()
                nInCluster = len(InCluster)
                DistInCluster = distM.iloc[InCluster, InCluster]
                if nInCluster > 1:
                    AveDistInClust = DistInCluster.sum(axis=1) / (nInCluster - 1)
                    AveDistInClust.reset_index(drop=True, inplace=True)
                    ClusterDiam[cluster] = AveDistInClust.max()
                else:
                    ClusterDiam[cluster] = 0

            ColorsX = Colors.copy()
            if respectSmallClusters:
                FSmallLabels = pd.Categorical(SmallLabels)
                SmallLabLevs = pd.to_numeric(FSmallLabels.categories)
                nSmallClusters = len(FSmallLabels.categories) - (SmallLabLevs[0] == 0)
                if nSmallClusters > 0:
                    if pamRespectsDendro:
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = list(range(nPoints))[SmallLabels == sclust]
                            onBr = pd.unique(onBranch[InCluster])
                            if len(onBr) > 1:
                                msg = "Internal error: objects in a small cluster are marked to belong\n" \
                                        "to several large branches:" + str(onBr)
                                sys.exit(msg)
                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[[onBr]]
                                labelsOnBranch = branchLabels[basicOnBranch]
                                useObjects = ColorsX in np.unique(labelsOnBranch)
                                DistSClustClust = distM.iloc[InCluster, useObjects]
                                MeanDist = DistSClustClust.mean(axis=0)
                                useColorsFac = pd.Categorical(ColorsX[useObjects])
                                # TODO
                                MeanMeanDist = MeanDist.groupby(
                                    'useColorsFac').mean()  # tapply(MeanDist, useColorsFac, mean)
                                nearest = MeanMeanDist.idxmin()
                                NearestDist = MeanMeanDist[nearest]
                                if np.logical_or(np.all(NearestDist < ClusterDiam[nearest]),
                                                    NearestDist < maxPamDist).tolist()[0]:
                                    Colors[InCluster] = nearest
                                    nPAMed = nPAMed + len(InCluster)
                                else:
                                    Colors[InCluster] = -1
                    else:
                        labelsOnBranch = list(range(nProperLabels))
                        useObjects = np.where(ColorsX != 0)[0].tolist()
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = np.where(SmallLabels == sclust)[0].tolist()
                            DistSClustClust = distM.iloc[InCluster, useObjects]
                            MeanDist = DistSClustClust.mean(axis=0)
                            useColorsFac = pd.Categorical(ColorsX[useObjects])
                            MeanDist = pd.DataFrame({'MeanDist': MeanDist, 'useColorsFac': useColorsFac})
                            MeanMeanDist = MeanDist.groupby(
                                'useColorsFac').mean()  # tapply(MeanDist, useColorsFac, mean)
                            nearest = MeanMeanDist[['MeanDist']].idxmin().astype(int) - 1
                            NearestDist = MeanMeanDist[['MeanDist']].min()
                            if np.logical_or(np.all(NearestDist < ClusterDiam[nearest]),
                                                NearestDist < maxPamDist).tolist()[0]:
                                Colors[InCluster] = nearest
                                nPAMed = nPAMed + len(InCluster)
                            else:
                                Colors[InCluster] = -1
            Unlabeled = np.where(Colors == 0)[0].tolist()
            if len(Unlabeled) > 0:
                if pamRespectsDendro:
                    unlabOnBranch = Unlabeled[onBranch[Unlabeled] > 0]
                    for obj in unlabOnBranch:
                        onBr = onBranch[obj]
                        basicOnBranch = branch_basicClusters[[onBr]]
                        labelsOnBranch = branchLabels[basicOnBranch]
                        useObjects = ColorsX in np.unique(labelsOnBranch)
                        useColorsFac = pd.Categorical(ColorsX[useObjects])
                        UnassdToClustDist = distM.iloc[useObjects, obj].groupby(
                            'useColorsFac').mean()  # tapply(distM[useObjects, obj], useColorsFac, mean)
                        nearest = UnassdToClustDist.idxmin()
                        NearestClusterDist = UnassdToClustDist[nearest]
                        nearestLabel = pd.to_numeric(useColorsFac.categories[nearest])
                        if np.logical_or(np.all(NearestClusterDist < ClusterDiam[nearest]),
                                            NearestClusterDist < maxPamDist).tolist()[0]:
                            Colors[obj] = nearest
                            nPAMed = nPAMed + 1
                else:
                    useObjects = np.where(ColorsX != 0)[0].tolist()
                    useColorsFac = pd.Categorical(ColorsX[useObjects])
                    tmp = pd.DataFrame(distM.iloc[useObjects, Unlabeled])
                    tmp['group'] = useColorsFac
                    UnassdToClustDist = tmp.groupby(
                        ['group']).mean()  # apply(distM[useObjects, Unlabeled], 2, tapply, useColorsFac, mean)
                    nearest = np.subtract(UnassdToClustDist.idxmin(axis=0),
                                            np.ones(UnassdToClustDist.shape[1])).astype(
                        int)  # apply(UnassdToClustDist, 2, which.min)
                    nearestDist = UnassdToClustDist.min(axis=0)  # apply(UnassdToClustDist, 2, min)
                    nearestLabel = nearest + 1
                    sumAssign = np.sum(np.logical_or(nearestDist < ClusterDiam[nearest], nearestDist < maxPamDist))
                    assign = np.where(np.logical_or(nearestDist < ClusterDiam[nearest], nearestDist < maxPamDist))[
                        0].tolist()
                    tmp = [Unlabeled[x] for x in assign]
                    Colors[tmp] = [nearestLabel.iloc[x] for x in assign]
                    nPAMed = nPAMed + sumAssign

    Colors[np.where(Colors < 0)[0].tolist()] = 0
    UnlabeledExist = (np.count_nonzero(Colors == 0) > 0)
    NumLabs = list(map(int, Colors.copy()))
    Sizes = pd.DataFrame(NumLabs).value_counts().sort_index()
    OrdNumLabs = pd.DataFrame({"Name": NumLabs, "Value": np.repeat(1, len(NumLabs))})

    if UnlabeledExist:
        if len(Sizes) > 1:
            SizeRank = np.insert(stats.rankdata(-1 * Sizes[1:len(Sizes)], method='ordinal') + 1, 0, 1)
        else:
            SizeRank = 1
        for i in range(len(NumLabs)):
            OrdNumLabs.Value[i] = SizeRank[NumLabs[i]]
    else:
        SizeRank = stats.rankdata(-1 * Sizes[0:len(Sizes)], method='ordinal')
        for i in range(len(NumLabs)):
            OrdNumLabs.Value[i] = SizeRank[NumLabs[i]]

    print("\tDone..\n")

    OrdNumLabs.Value = OrdNumLabs.Value - UnlabeledExist
    return OrdNumLabs
