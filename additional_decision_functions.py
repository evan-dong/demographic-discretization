import os
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import cvxpy as cp

from utils import exact_counts
from metrics import FidelityMetrics


# region Single-Batch Functions


# region Individual Decision Functions
def custom_threshold_discretization(
    probs, thresholds, tiebreak_function: Callable, uncoded_val=None
):
    """The thresholding rule with an input function for customized tiebreaking.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    thresholds : array-like broadcasting to shape (`n_samples`, `n_classes`)
        Threshold values are assumed to be in the interval (0, 1).
    tiebreak_function : Callable of the form `f(probs: np.ndarray) -> np.ndarray`
        The customized function that breaks ties where multiple classes are above their thresholds.
    uncoded_val : int, default = `n_classes`+1
        The discretized value for samples where no threshold is reached.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The thresholded discrete predictions for each row.
        Includes an "uncoded" value of discrete output when no threshold is reached.
    """
    # assert np.all(thresholds >= 0.5) ## could have
    if np.any(thresholds <= 0.5):
        print("Warning: thresholds at <= 0.5 may have ties.")
        if tiebreak_function is None:
            print(
                "No custom tie-breaking function provided. Default is to choose the class furthest over the threshold."
            )

    # probs = np.array(probs)
    diffs = np.array(probs) - np.array(
        thresholds
    )  ## (n_samples, n_classes) - (n_classes)

    ## the default uncoded_val is n_classes + 1
    uncoded_val = diffs.shape[1] + 1 if uncoded_val is None else uncoded_val

    n_hits = np.sum(diffs >= 0, axis=1)
    ties_mask = n_hits > 1
    exact_mask = n_hits == 1

    values = np.zeros(n_hits.shape)  # hits_mask - 1
    ## take a tiebreak function argument?
    values[exact_mask] = np.argmax(diffs[exact_mask], axis=1)
    values[ties_mask] = tiebreak_function(
        probs[ties_mask]
    )  # np.argmax(diffs[hits_mask], axis=1) ##
    values[n_hits == 0] = uncoded_val
    return values


# endregion

# region Joint Decision Functions


def exact_matching_discretization(
    probs,
    class_count_caps,
) -> np.ndarray:
    """The matching discretization rule; sets a target allotment of output class
       counts and optimizes accuracy given that exact constraint.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    class_count_caps : array-like of shape (`n_classes`)
        The maximum number of samples that can be assigned to each class.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The matching discrete predictions for each row.

    Notes
    -------
    This implementation allows for matching without an exact predetermined number of
    samples in each class, but rather an upper limit. This may prove useful when the
    target distribution is uncertain and upper confidence bounds are a more suitable
    constraint.

    It may be worth noting that this implementation uses a more general bipartite
    matching solver that is not asymptotically optimal (due to most columns of
    `probs_full`) being identical duplicates). However, making a practically faster
    implementation in CPython is beyond the scope of our work here.
    """

    ## duplicate the probabilities to create a bipartite graph.
    ## NOTE: not all class counts may be filled.
    probs_full = np.repeat(probs, class_count_caps, axis=1)
    ## calculate and pull out optimal indices
    answer_full = scipy.optimize.linear_sum_assignment(probs_full, maximize=True)[1]
    answer_orig = np.digitize(answer_full, np.cumsum(reference_counts))

    return answer_orig


def integer_program_discretization(
    probs,
    gamma: float,
    reference_distribution=None,
    fidelity_metric: FidelityMetrics = "L1",
    timeout: int = 30,
    solver=cp.GUROBI,
    verbose: bool = False,
) -> np.ndarray:
    """The integer program discretization rule; exactly calculates the expected
       accuracy and fidelity of a discretization, and finds the Pareto-optimal tradeoff
       given a gamma parameter.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    gamma : float
        a tuning parameter that balances the accuracy and fidelity objectives.
    reference_distribution : array-like of shape (`n_classes`), optional
        The reference probability distribution that is used to calculate fidelity.
        Will be normalized to sum to 1.
        By default the aggregate posterior of `probs`.
    fidelity_metric : {"L1", "L2", "KL"}, optional
        The desired distribution fidelity metric, by default "L1".
    timeout : int, optional
        Number of seconds the integer program solver is given before terminating,
        by default 30.
    solver : cp.solver, optional
        A cvxpy.solver that can solve Mixed Integer-Linear Programs,
        by default cp.GUROBI.
    verbose : bool, optional
        Controls whether solver output and optimal objectives are displayed,
        by default False.

    Returns
    -------
    np.ndarray of shape (`n_samples`)
        The integer-program optimal discrete predictions for each row.

    Raises
    ------
    ValueError
        Raised if metric is not one of the implemented fidelity metrics.
    """

    ## aggregate posterior if not otherwise specified
    reference_distribution = (
        np.mean(probs, axis=0)
        if reference_distribution is None
        else reference_distribution
    )

    training_size, n_classes = probs.shape

    # initialize variables
    class_assignment = cp.Variable(
        (training_size, n_classes), boolean=True
    )  ## binary assignment matrix

    ## expected accuracy (calculated from `probs`)
    acc_objective = cp.sum(cp.multiply(class_assignment, probs)) / training_size

    marginal_distribution = (
        cp.sum(class_assignment, axis=0) / training_size
    )  ## normalize to one
    match fidelity_metric:
        case "L1":
            ## cvxpy doesn't like adding in the normal order
            distribution_objective = cp.norm(
                -marginal_distribution + reference_distribution, 1
            )
        case "L2":
            distribution_objective = cp.norm(
                -marginal_distribution + reference_distribution, 2
            )
        case "KL":
            distribution_objective = cp.sum(
                cp.rel_entr(reference_distribution, marginal_distribution)
            )
        case _:
            raise ValueError(
                "Not an implemented marginal objective. Please use one of {'L1', 'L2', 'KL'}"
            )

    ## maximize accuracy (linear) and minimize {L1 distance, L2 distance, KL divergence} (convex)
    total_objective = gamma * acc_objective - (1 - gamma) * distribution_objective

    constraints = [
        cp.sum(class_assignment, axis=1) == 1,
    ]

    objective = cp.Maximize(total_objective)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver, verbose=verbose, TimeLimit=timeout)

    ## find the class where the assignment is 1 for each row
    predicted_labels = np.argmax(class_assignment.value, axis=1)
    if verbose:
        print("gamma: ", gamma)
        print("integer program expected accuracy: ", acc_objective.value)
        print(
            f"integer program {fidelity_metric} fidelity: ",
            distribution_objective.value,
        )
    return predicted_labels


# endregion
# endregion

# region Multiple-Batch Functions


def pareto_curve_sweep(
    probs,
    gammas: list[float],
    reference_distribution=None,
    fidelity_metric: FidelityMetrics = "L1",
    timeout: int = 10,
    n_processes: int = 0,
) -> list[np.ndarray]:
    """Applies the integer program discretization rule to a single set of probabilities
       for multiple values of gamma to sample the Pareto curve of optimal
       accuracy-fidelity tradeoffs.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    gammas : list[float]
        a list of tuning parameters that balance the accuracy and fidelity objectives.
    reference_distribution : array-like of shape (`n_classes`), optional
        The reference probability distribution that is used to calculate fidelity.
        Will be normalized to sum to 1.
        By default the aggregate posterior of `probs`.
    fidelity_metric : {"L1", "L2", "KL"}, optional
        The desired distribution fidelity metric, by default "L1".
    timeout : int, optional
        Number of seconds the integer program solver is given before terminating,
        by default 10.
    n_processes : int, optional
        Number of processes to use for parallel processing, by default 0. If 0 or less,
        will run as many parallel processes as there are available CPU cores.

    Returns
    -------
    list[np.ndarray] of shape (`n_gammas`, `n_samples`)
        The array of integer-program optimal discrete predictions for each gamma value.

    Notes
    -------
    The default heuristic for `n_processes` may not be optimal when using MILP solvers
    other than Gurobi, as Gurobi appears to derive no speedup from having multiple
    cores available per process.
    """
    n_processes = os.cpu_count() if n_processes < 1 else n_processes  ## interpret

    if n_processes > 1:
        mapped_args = [
            (probs, gamma, reference_distribution, fidelity_metric, timeout)
            for gamma in gammas
        ]
        with Pool(n_processes) as p:
            integer_program_preds = p.starmap(
                integer_program_discretization, mapped_args, chunksize=1
            )
    else:
        integer_program_preds = [
            integer_program_discretization(
                probs,
                gamma,
                reference_distribution=reference_distribution,
                fidelity_metric=fidelity_metric,
                timeout=timeout,
            )
            for gamma in gammas
        ]
    return integer_program_preds


def batched_integer_program_discretization(
    batches: list,
    gammas: list[float],
    reference_dists: list | None = None,
    fidelity_metric: FidelityMetrics = "L1",
    timeout: int = 10,
    n_processes: int = 0,
) -> list[pd.Series]:
    """Applies the integer program discretization rule to pre-specified batches of
       probabilities for multiple values of gamma to sample the Pareto curve of
       optimal accuracy-fidelity tradeoffs.

    Parameters
    ----------
    batches : list of array-likes of shape (`batch_size`, `n_classes`)
        List of batches of probability outputs.
        To generate batches, see `batch_dataset` in utils.py.
        Each row in each batch represents a probability distribution, summing to 1.
    gammas : list[float]
        a list of tuning parameters that balance the accuracy and fidelity objectives.
    reference_dists : list of array-likes of shape (`n_classes`) | None, optional
        List of reference distributions for each batch, by default None.
        If None, the aggregate posterior is calculated in
        `integer_program_discretization` as the reference.
    fidelity_metric : {"L1", "L2", "KL"}, optional
        The desired distribution fidelity metric, by default "L1".
    timeout : int, optional
        Number of seconds the integer program solver is given before terminating,
        by default 10.
    n_processes : int, optional
        Number of processes to use for parallel processing, by default 0. If 0 or less,
        will run as many parallel processes as there are available CPU cores.

    Returns
    -------
    list[np.ndarray] of shape (`n_gammas`, `n_samples`)
        The array of integer-program optimal discrete predictions,
        calculated in batches, for each gamma value.

    Notes
    -------
    The default heuristic for `n_processes` may not be optimal when using MILP solvers
    other than Gurobi, as Gurobi appears to derive no speedup from having multiple
    cores available per process.
    """
    n_processes = os.cpu_count() if n_processes < 1 else n_processes  ## interpret

    # default is the aggregate posterior
    reference_dists = (
        [None] * len(batches) if reference_dists is None else reference_dists
    )
    assert len(batches) == len(reference_dists)

    if n_processes > 1:
        ## create n_gammas, n_batches
        nested_args = [
            [
                (batch, gamma, ref, fidelity_metric, timeout)
                for batch, ref in zip(batches, reference_dists)
            ]
            for gamma in gammas
        ]  ##[[batch1gamma1, batch2gamma1, ...], [batch1gamma2, ...], ...]
        with Pool(n_processes) as p:
            integer_program_preds = [
                np.concatenate(
                    p.starmap(integer_program_discretization, gamma_chunk, chunksize=1)
                )
                for gamma_chunk in nested_args
            ]
    else:
        integer_program_preds = []
        for gamma in gammas:
            batched_predictions = [
                integer_program_discretization(
                    batch, gamma, ref, fidelity_metric=fidelity_metric, timeout=timeout
                )
                for batch, ref in zip(batches, reference_dists)
            ]
            integer_program_preds.append(np.concatenate(batched_predictions))

    return integer_program_preds


# endregion
