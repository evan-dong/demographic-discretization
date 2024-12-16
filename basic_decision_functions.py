import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
from sklearn.svm import LinearSVC

from utils import exact_counts

# region Single-Batch Functions


# region Individual Decision Functions
def argmax_discretization(probs) -> np.ndarray:
    """The argmax discretization rule. Ties are broken in class order.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.


    Returns
    -------
    np.ndarray of shape (`n_samples`)
        the discrete argmax predictions for each row.
    """
    return np.argmax(probs, axis=1)


def thompson_discretization(probs, random_seed=None) -> np.ndarray:
    """The Thompson sampling discretization rule.

    Args:
        random_seed (optional): a seed for a numpy BitGenerator. Defaults to None.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    random_seed : optional
        A seed for a numpy BitGenerator, by default None.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The Thompson sampled discrete predictions for each row.
    """

    random_generator = np.random.default_rng(seed=random_seed)

    probs = np.array(probs)
    ## random.choice is slow; this is vectorized
    randchoice = random_generator.random(size=probs.shape[0])
    cumprobs = np.cumsum(probs, axis=1)
    answers = np.sum(randchoice[:, np.newaxis] > cumprobs, axis=1)
    return np.array(answers)


def top_k_discretization(probs, top_k: int, random_seed=None) -> np.ndarray:
    """The top-k sampling discretization rule; apply Thompson sampling the top-k most
       likely classes, with renormalized probabilities.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    top_k : int
        The number of classes to sample from.
    random_seed : optional
        A seed for a numpy BitGenerator, by default None.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The top-k sampled discrete predictions for each row.

    Raises
    ------
    ValueError
        Raised when top_k is not in the interval [2, n_classes-1].
    """

    probs = np.array(probs).copy()
    _, n_classes = probs.shape
    if (top_k < 2) or (n_classes - 1 < top_k):
        raise ValueError("top_k must be an integer on the interval [2, n_classes-1]")
    bottom_k = n_classes - top_k

    ## in case of ties, keep both tied classes
    boundary_vals = np.partition(probs, bottom_k)[:, bottom_k]
    dropped = probs < boundary_vals[:, np.newaxis]
    probs[dropped] = 0
    probs /= np.sum(probs, axis=1)[:, np.newaxis]

    return thompson_discretization(probs, random_seed=random_seed)


def partial_threshold_discretization(
    probs, thresholds, uncoded_val: int = None
) -> np.ndarray:
    """The thresholding rule, where only rows with probabilities above a particular
       threshold value (customizable by class) are discretized.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    thresholds : array-like broadcasting to shape (`n_samples`, `n_classes`)
        Threshold values are assumed to be in the interval (0, 1).
    uncoded_val : int, default = `n_classes`+1
        The discretized value for samples where no threshold is reached.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The thresholded discrete predictions for each row.
        Includes an "uncoded" value of discrete output when no threshold is reached.
    """

    if np.any(thresholds <= 0.5):
        print(
            "Warning: having thresholds at <= 0.5 may lead to ties. If muliple classes are over the threshold, the class furthest over the threshold will be selected. If multiple classes are equally over the threshold, ties are broken in class order."
        )

    diffs = np.array(probs) - np.array(thresholds)
    over_mask = np.any(diffs >= 0, axis=1)

    ## the default uncoded_val is n_classes + 1
    uncoded_val = diffs.shape[1] + 1 if uncoded_val is None else uncoded_val

    values = np.zeros(over_mask.shape, dtype=int)
    values[over_mask] = np.argmax(diffs[over_mask], axis=1)
    values[~over_mask] = uncoded_val
    return values


# endregion

# region Joint Decision Functions


def matching_discretization(probs, reference_distribution=None) -> np.ndarray:
    """The matching discretization rule; sets a target allotment of output class
       counts and optimizes accuracy given that exact constraint.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    reference_distribution : array-like of shape (`n_classes`), optional
        The reference probability distribution that matching targets.
        Will be normalized to sum to 1.
        By default the aggregate posterior of `probs`.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The matching discrete predictions for each row.

    Notes
    -------
    The exact, discrete distribution used minimizes the L1 distance to the
    reference distribution, as per the `exact_counts` function in utils.py.

    It may be worth noting that this implementation uses a more general bipartite
    matching solver that is not asymptotically optimal (due to most columns of
    `probs_full`) being identical duplicates). However, making a practically faster
    implementation in CPython is beyond the scope of our work here.
    """

    ## n_classes
    probs = np.array(probs)
    n_samples, n_classes = probs.shape
    assert np.isclose(np.sum(probs), n_samples)

    ## Use the aggregate posterior if no reference is provided.
    ## If a reference is provided, renormalize it to a probability distribution.
    reference_distribution = (
        np.mean(probs, axis=0)
        if reference_distribution is None
        else reference_distribution / np.sum(reference_distribution)
    )
    assert len(reference_distribution) == n_classes

    ## calculate the discrete reference distribution
    reference_counts = exact_counts(reference_distribution, n_samples)

    print("output class distribution", reference_counts)

    ## duplicate the probabilities to create a balanced bipartite graph
    probs_full = np.repeat(probs, reference_counts, axis=1)
    ## calculate and pull out optimal indices
    answer_full = scipy.optimize.linear_sum_assignment(probs_full, maximize=True)[1]
    answer_orig = np.digitize(answer_full, np.cumsum(reference_counts))

    assert len(answer_orig) == n_samples, (len(answer_orig), probs_full.shape)

    return answer_orig


# endregion
# endregion

# region Multiple-Batch Functions


def data_driven_threshold_discretization(
    training_probs,
    training_labels,
    testing_probs,
    testing_labels=None,
    model=None,
) -> np.ndarray:
    """The data-driven threshold discretization rule; given a discretization rule that
       produces the the provided `training_labels` from `training_probs`, trains a
       machine learning model to approximate the rule and applies it to
       `testing_probs`. Evaluates the accuracy of outputs to `testing_labels` if
       provided.

    Parameters
    ----------
    training_probs : array-like of shape (`n_training_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    training_labels : array-like of shape (`n_training_samples`)
        A set of discrete labels corresponding to the rows of training_probs, such as
        those output by other discretization functions.
    testing_probs : array-like of shape (`n_testing_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    testing_labels : array-like of shape (`n_testing_samples`), optional
        A set of discrete labels corresponding to the rows of testing_probs, such as
        those output by other discretization functions. Used to evaluate the accuracy
        of the trained model, by default None.
    model : inherits sklearn.base.BaseEstimator and MixinClassifier, optional
        Any sklearn model-like object that implements the methods model.fit(X, y),
        model.predict(X), and model.score().
        By default sklearn.svm.LinearSVC with regularization parameter C=100.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The machine learning-based discrete predictions for each row of
        `testing_probs`, based on the values of `training_probs` and
        `training_labels`.

    Notes
    _______
    To retrieve model predictions for the training data,
    """

    if model is None:
        model = LinearSVC(C=100)
    model = model.fit(training_probs, training_labels)
    print("training accuracy", model.score(training_probs, training_labels))
    if testing_labels is not None:
        print("testing accuracy", model.score(testing_probs, testing_labels))

    model_outputs = model.predict(testing_probs)
    return model_outputs


def batched_matching_discretization(
    batches: list,
    reference_dists: list | None = None,
    n_processes: int = 0,
) -> np.ndarray:
    """Applies the matching discretization rule to pre-specified batches of
       probabilities.

    Parameters
    ----------
    batches : list of array-likes of shape (`batch_size`, `n_classes`)
        List of batches of probability outputs.
        To generate batches, see `batch_dataset` in utils.py.
        Each row in each batch represents a probability distribution, summing to 1.
    reference_dists : list of array-likes of shape (`n_classes`) | None, optional
        List of reference distributions for each batch, by default None.
        If None, the aggregate posterior is calculated in `matching_discretization`
        as the reference.
    n_processes : int, optional
        Number of processes to use for parallel processing, by default 0. If 0 or less,
        will run half the number of parallel processes as there are available CPU
        cores. Note that each process instance will still make use of multiple cores.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The matching discrete predictions for each row, calculated in batches.
    """

    ## heuristic for optimal number of processes
    n_processes = os.cpu_count() // 2 if n_processes < 1 else n_processes

    # default is the aggregate posterior
    reference_dists = (
        [None] * len(batches) if reference_dists is None else reference_dists
    )
    assert len(batches) == len(reference_dists)

    args = list(zip(batches, reference_dists))
    if n_processes > 1:
        with Pool(n_processes) as p:
            matching_predictions = p.starmap(matching_discretization, args, chunksize=1)
    else:
        matching_predictions = [
            matching_discretization(batch, reference_distribution=reference_dist)
            for batch, reference_dist in args
        ]

    matching_predictions = np.concatenate(matching_predictions)

    return matching_predictions


# endregion
