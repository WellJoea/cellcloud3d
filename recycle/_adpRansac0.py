import numpy as np
from warnings import warn

def adpRansac(data, model_class, threshold, min_inliers=5, min_samples=5, P=0.99, seed=200504):
    rng = np.random.default_rng(seed)
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    iterations = np.inf
    count = 0

    inliers = []
    best_residual = np.inf
    model = model_class()

    while iterations > count:
        if 0 < min_inliers and min_inliers < 1:
            min_inliers = int(min_inliers * num_samples)

        spl_idxs = rng.choice(num_samples, min_samples, replace=False)
        samples = [d[spl_idxs] for d in data]

        try:
            model.estimate(*samples)
            residuals = np.abs(model.residuals(*data))
            inliers = residuals < threshold

            residuals_sum = residuals.dot(residuals)
            inliers_count = np.sum(inliers)

            if (inliers_count >= min_inliers):
                data_inliers = [d[inliers] for d in data]
                model.estimate(*data_inliers)
                residuals = np.abs(model.residuals(*data))
                if (residuals_sum < best_residual):
                    inliers = residuals < threshold
                    best_inliers = inliers
                    best_residual = residuals.dot(residuals)
                    #inliers_count = np.sum(inliers)

        except ZeroDivisionError as e:
            print(e)

        ratio = inliers_count/ num_samples
        if min_inliers / num_samples < ratio:
            best_residual = np.inf
            min_inliers = ratio
            iterations = int(np.log(1 - P) / np.log(1 - min_inliers ** min_samples)) + 1
            print(iterations)
        count += 1

    if any(best_inliers):
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
    else:
        model = None
        best_inliers = None
        warn("No inliers found. Model not fitted")

    return model, best_inliers