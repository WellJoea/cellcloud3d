import numpy as np
from warnings import warn

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    _EPSILON = np.spacing(1)
    if probability == 0:
        return 0
    if n_inliers == 0:
        return np.inf
    inlier_ratio = n_inliers / n_samples
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    return np.ceil(np.log(nom) / np.log(denom))

def adpRansac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, rng=None, initial_inliers=None):

    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = []
    validate_model = is_model_valid is not None
    validate_data = is_data_valid is not None

    rng = np.random.default_rng(rng)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples <= num_samples):
        raise ValueError(f"`min_samples` must be in range (0, {num_samples}]")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError(
            f"RANSAC received a vector of initial inliers (length "
            f"{len(initial_inliers)}) that didn't match the number of "
            f"samples ({num_samples}). The vector of initial inliers should "
            f"have the same length as the number of samples and contain only "
            f"True (this sample is an initial inlier) and False (this one "
            f"isn't) values.")

    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None
                else rng.choice(num_samples, min_samples,
                                         replace=False))

    # estimate model for current random sample set
    model = model_class()

    num_trials = 0
    # max_trials can be updated inside the loop, so this cannot be a for-loop
    while num_trials < max_trials:
        num_trials += 1

        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]

        # for next iteration choose random sample set and be sure that
        # no samples repeat
        spl_idxs = rng.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if validate_data and not is_data_valid(*samples):
            continue

        success = model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if validate_model and not is_model_valid(model, *samples):
            continue

        residuals = np.abs(model.residuals(*data))
        # consensus set / inliers
        inliers = residuals < residual_threshold
        residuals_sum = residuals.dot(residuals)

        # choose as new best model if number of inliers is maximal
        inliers_count = np.count_nonzero(inliers)
        if (
            # more inliers
            inliers_count > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (inliers_count == best_inlier_num
                and residuals_sum < best_inlier_residuals_sum)):
            best_inlier_num = inliers_count
            best_inlier_residuals_sum = residuals_sum
            best_inliers = inliers
            max_trials = min(max_trials,
                             _dynamic_max_trials(best_inlier_num,
                                                 num_samples,
                                                 min_samples,
                                                 stop_probability))
            if (best_inlier_num >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum):
                break

    # estimate final model using all inliers
    if any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        if validate_model and not is_model_valid(model, *data_inliers):
            warn("Estimated model is not valid. Try increasing max_trials.")
    else:
        model = None
        best_inliers = None
        warn("No inliers found. Model not fitted")

    return model, best_inliers
