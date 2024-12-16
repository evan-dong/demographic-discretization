from typing import Literal

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score

from utils import exact_counts, preds_to_counts


FidelityMetrics = Literal["L1", "L2", "KL"]
"""The fidelity metrics available for comparing two distributions.
"L1" : The L1 distance between two distributions.
"L2" : The L2 distance between two distributions.
"KL" : The Kullback-Leibler divergence between two distributions. 
        Note that this is not a distance, and is not symmetric.
"""


def true_accuracy(true_labels, predictions) -> float:
    return accuracy_score(true_labels, predictions)


def mean_expected_accuracy(probs, predictions) -> float:
    """Calculates the expected accuracy of discrete predictions in expectation over the
       provided probabilities for each row.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1. This is assumed
        to be the true class probabilities for the purpose of calculating the expected
        accuracy.
    predictions : array-like of shape (`n_samples`)
        The array of discrete outputs corresponding to each row of `probs`.

    Returns
    -------
    float
        The mean expected accuracy along all rows.
    """
    return np.mean(np.array(probs)[np.arange(probs.shape[0]), predictions], axis=0)


def distribution_fidelity(
    reference, counts, fidelity_metric: FidelityMetrics = "L1"
) -> float:
    """Calculates the distribution fidelity between the reference and actualized
       discrete count distributions after normalizing both to sum to 1.

    Parameters
    ----------
    reference : array-like of shape (`n_classes`)
        The reference distribution.
    counts : array-like of shape (`n_classes`)
        The class distribution of discrete outputs.
    fidelity_metric : {"L1", "L2", "KL"}, optional
        The desired distribution fidelity metric, by default "L1".

    Returns
    -------
    float
        The distribution fidelity in either (negative) L1 distance, L2 distance, or
        KL divergence.

    Raises
    ------
    ValueError
        Raised if metric is not one of the implemented fidelity metrics.

    Notes
    ------
    `reference` and `counts` are not symmetric in the KL-divergence case.
    """

    ## normalize if not yet normalized
    reference = reference / np.sum(reference)
    counts = counts / np.sum(counts)
    match fidelity_metric:
        case "L1":
            return -np.linalg.norm(reference - counts, 1)
        case "L2":
            return -np.linalg.norm(reference - counts, 2)
        case "KL":
            return -np.mean(scipy.special.rel_entr(counts, reference))
        case _:
            raise ValueError(
                "Not an implemented marginal objective. Please use one of {'L2', 'L1', 'KL'}"
            )


def assess_predictions(
    preds: pd.DataFrame,
    probs=None,
    true_labels=None,
    prior=None,
    fidelity_metric: FidelityMetrics = "L1",
    uncoded_val: int | None = None,
):
    """Assess discretization methods' accuracy and fidelity, including methods that
       leave some predictions uncoded.

    Parameters
    ----------
    preds : pd.DataFrame of shape (`n_samples`, `n_methods`)
        DataFrame containing the predictions. Columns are discretization methods.
    probs : array-like of shape (`n_samples`, `n_classes`), optional
        Each row represents a probability distribution, summing to 1. This is assumed
        to be the true class probabilities for the purpose of calculating the expected
        accuracy. By default None.
    true_labels : array-like of shape (`n_samples`), optional
        True labels for the predictions, by default None.
    prior : array-like of shape (`n_classes`), optional
        Prior probabilities, or other fixed reference distribution, by default None.
    fidelity_metric : {"L1", "L2", "KL"}, optional
        The desired distribution fidelity metric, by default "L1".
    uncoded_val : int | None, optional
        The value for the uncoded class, by default `n_classes` + 1, as inferred
        from `probs`, `true_labels`, and `prior`.

    Returns
    -------
    pd.DataFrame of shape (`n_methods`, [0-5])
        DataFrame of accuracy and fidelity metric results.
        Given probabilities (`probs`), calculates:
            "Expected Accuracy", the mean expected accuracy
            "Aggregate Posterior Fidelity", the fidelity to the aggregate posterior
        Given ground truth labels (`true_labels`), calculates:
            "Accuracy", the accuracy with respect to the true labels
            "Ground Truth Fidelity", the fidelity to the ground truth labels' marginal
        Given a prior fixed reference distribution (`prior`), calculates:
            "Prior Distribution Fidelity", the fidelity to the prior reference distribution

    Raises
    -------
    ValueError
        If unable to infer `uncoded_val` due to lack of assessment data.
    ValueError
        If `probs`, `true_labels`, and `prior` do not imply the same number of classes.
    """

    cols = preds.columns
    preds = preds.astype(int)

    ## validate or infer unvoded_val
    if uncoded_val is None:
        probs_uncoded = None if probs is None else probs.shape[1] + 1
        labels_uncoded = None if true_labels is None else np.max(true_labels) + 1
        prior_uncoded = None if probs is None else prior.shape[0] + 1
        inferred_vals = [
            val
            for val in [probs_uncoded, labels_uncoded, prior_uncoded]
            if val is not None
        ]
        if len(inferred_vals) == 0:
            raise ValueError(
                "unable to infer `uncoded_val`; no assessment data provided"
            )

        if not (np.array(inferred_vals) == inferred_vals[0]).all():
            raise ValueError(
                "probs, labels, and prior must all imply the same number of classes"
            )
        else:
            uncoded_val = inferred_vals[0]

    discrete_dists = np.array(preds_to_counts(preds).drop(labels=uncoded_val)).T
    method_sums = np.sum(discrete_dists, axis=1)  ##
    preds = np.array(preds)  ## n_samples, n_methods

    assert (preds <= uncoded_val).all()  ## needs to be this way

    results = pd.DataFrame(index=cols)
    results["Dropped Fraction"] = 1 - (method_sums / preds.shape[0])

    if probs is not None:
        assert (
            uncoded_val > probs.shape[1] - 1
        )  ## uncoded should be outside the probs prob distribution, by definition.
        reshaped_probs = np.zeros(
            (probs.shape[0], uncoded_val + 1)
        )  ## + 1 so that we can index the uncoded rows of preds into zeros
        reshaped_probs[: probs.shape[0], : probs.shape[1]] = probs

        # the zeros from uncoded rows are canceled out by the smaller denominators
        results["Expected Accuracy"] = (
            np.sum(reshaped_probs[np.arange(probs.shape[0]), preds.T], axis=1)
            / method_sums
        )

        aggregate_posterior = np.array(np.sum(probs, axis=0))

        results["Aggregate Posterior Fidelity"] = [
            distribution_fidelity(
                aggregate_posterior,
                counts,
                fidelity_metric=fidelity_metric,
            )
            for counts in discrete_dists
        ]

    if true_labels is not None:
        results["Accuracy"] = (
            np.sum(preds == np.array(true_labels)[:, np.newaxis], axis=0) / method_sums
        )

        true_label_counts = np.bincount(true_labels)
        results["Ground Truth Fidelity"] = [
            distribution_fidelity(
                true_label_counts,
                counts,
                fidelity_metric=fidelity_metric,
            )
            for counts in discrete_dists
        ]

    if prior is not None:
        results["Prior Distribution Fidelity"] = [
            distribution_fidelity(
                prior,
                counts,
                fidelity_metric=fidelity_metric,
            )
            for counts in discrete_dists
        ]

    results.index.name = "Method"
    return results


def batched_vs_global_aggregate_posterior_fidelity(
    batches, fidelity_metric: FidelityMetrics = "L1"
) -> float:
    """Compares the difference in aggregate posterior fidelity when using the matching
       discretization rule with batches of probabilities (i.e., in `batched_matching`)
       instead of a singular, global matching problem with all probabilities (i.e., in
       `matching_discretization`). Useful to assess the loss in overall fidelity when
       optimizing for conditional fidelities.

    Parameters
    ----------
    batches : list of array-likes of shape (`batch_size`, `n_classes`)
        List of batches of probability outputs.
        To generate batches, see `batch_dataset` in utils.py.
        Each row in each batch represents a probability distribution, summing to 1.
    fidelity_metric : {"L1", "L2", "KL"}, optional
        The desired distribution fidelity metric, by default "L1".

    Returns
    -------
    float
        The fidelity between the global and batched output distributions.
    """

    batched_counts = [
        exact_counts(np.mean(batch, axis=0), batch.shape[0]) for batch in batches
    ]
    batched_marginal = np.sum(batched_counts, axis=0)

    ## should we do discrete or continous aggregate posterior?
    global_marginal = exact_counts(
        np.mean(np.concatenate(batches), axis=0), len(np.concatenate(batches))
    )

    return distribution_fidelity(
        global_marginal, batched_marginal, fidelity_metric=fidelity_metric
    )


def calculate_error_rates(
    preds: pd.DataFrame,
    true_label_col="true",
    uncoded_val: int | None = None,
    class_map=None,
) -> pd.DataFrame:
    """Calculates false positive and false negative rates for each class.

    Parameters
    ----------
    preds : pd.DataFrame of shape (`n_samples`, `n_methods`)
        DataFrame containing the predictions. Columns are discretization methods.
    true_label_col : str | None, optional
        The column of `df` that contains the true labels for calculating error rates,
        by default "true". Has `n_classes` unique values.
    uncoded_val : int | None, optional
        The value for the uncoded class, by default `n_classes` + 1, as inferred
        from the values in the `true_label_col` column of `preds`.
    class_map : dict | pd.Series | None, optional
        A dict-like object to rename class values, by default None.
        If `class_map` is a Series with a name, it will be used as the name of the
        first index level; otherwise, that level name will be "Class".

    Returns
    -------
    pd.DataFrame of shape (2*`n_classes`, `n_methods`)
        A dataframe with false positive and false negative rates, excluding uncoded
        predictions from all calculations. The first index level is the class values
        and the second is error types.
    """

    table = []

    classes = preds[true_label_col].unique()
    uncoded_val = np.max(classes) + 1 if uncoded_val is None else uncoded_val
    coded = preds != uncoded_val
    for value in classes:
        # predicted as class `value`
        subdf = preds == value
        ## conditional on true label being `value`, how often is the predicted label `value`?
        total_pos_neg = subdf.groupby(true_label_col).sum()

        ## conditional on true label being `value`, how often is the prediction `uncoded`?
        coded[true_label_col] = subdf[true_label_col]
        # adjust the mean to exclude uncoded samples in each method column
        uncoded_adjusted_rates = total_pos_neg / coded.groupby(true_label_col).sum()

        table.append(uncoded_adjusted_rates)

    ## use index name if available; else map
    if isinstance(class_map, pd.Series):
        class_index = class_map.loc[classes]
    else:
        class_index = pd.Series(
            [class_map[k] for k in classes] if class_map is not None else classes,
            name="Class",
        )
    ## table of true positives and false negatives
    error_rate_table = pd.concat(table, keys=class_index, axis="index")

    ## turn true positives to false positives
    error_rate_table = (
        error_rate_table.reset_index(level=1)
        .replace({True: "False Negative", False: "False Positive"})
        .rename(columns={true_label_col: "Error Type"})
    )
    is_pos = (error_rate_table["Error Type"] == "False Negative").astype(int)  ##
    error_rate_table = error_rate_table.set_index("Error Type", append=True)
    error_rate_table = (np.array(is_pos)[:, np.newaxis] - error_rate_table).abs()

    return error_rate_table
