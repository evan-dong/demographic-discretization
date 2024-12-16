import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

### MISC HELPERS


def preds_to_marginal(predictions, n_classes: int | None = None):
    return np.bincount(predictions, minlength=n_classes)


def posterior_to_aggregate(posterior):
    return np.sum(posterior, axis=0).astype(float)


def exact_counts(frequencies, size: int) -> np.ndarray:
    """Given a probability distribution and a size, scale the distribution vector to
       that size as discrete counts. This is implemented as the vector with the least
       L1 distance to `frequencies * size` that is also a vector of integers.

    Parameters
    ----------
    frequencies : array-like of shape (`n_classes`), optional
        A probability distribution vector that must sum to 1.
    size : int
        The total number of samples to apportion to classes;
        the sum of the returned vector.

    Returns
    -------
    np.ndarray
        The discrete counts of the classes that sum to `size` that has the lowest L1
        distance to the scaled probability distribution.

    Notes
    -----
        There exist many ways to apportion a set of fixed, discrete counts to classes
        of fractional ratios. Other methods of rounding fractions to discrete values in
        vectors exist, as seen on the literature in the mathmatics of political seat
        apportionment.

    """
    assert np.isclose(np.sum(frequencies), 1)

    ## scale the probability distribution to the size
    reference_dist = frequencies * size
    ## round down to the nearest integer
    floored = np.floor(reference_dist).astype(int)

    ## allocate the remaining values
    rounding_values = reference_dist - floored
    rounding = size - np.sum(floored)
    ## round up the remainders in order of size
    plus_one = (
        np.argsort(rounding_values)[
            -rounding.astype(int) :  ## ascending, so take the largest
        ]
        if rounding > 0
        else []
    )  ## need the if statement to avoid taking the whole array when `rounding`` is 0

    ## plays nicer if frequencies is a Series with non-numeric indices
    added = np.zeros(rounding_values.shape[0], dtype=int)
    added[plus_one] = 1
    reference_dist = floored + added

    return reference_dist


def batch_dataset(probs, batch_size: int, remainder: bool = False) -> list:
    """Split a dataset of probabilities into batches of an approximately fixed size.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    batch_size : int
        The target size of each batch. The output batches may not all exactly have
        size `batch_size`, but will be as close as possible.
    remainder : bool, optional
        A flag to partition all but one batch into exactly size `batch_size`, with the
        remainder batch being as small as needed, by default False. By default, batches
        will be as close in size to each other as possible, such as their average size
        is as close to `batch_size` as possible.

    Returns
    -------
    batches : list of array-like of shape (`[variable batch size]`, `n_classes`)
        The batched values of `probs`. Note that sizes of the batches may vary.
    """

    if remainder:
        ## make the batches an exact size and then make a remainder batch
        n_full_batches = probs.shape[0] // batch_size
        partial_size = probs.shape[0] % n_full_batches
        batches = np.split(probs[:-partial_size], n_full_batches)
        batches.append(probs[-partial_size:])
    else:
        ## divide as evenly as possible
        n_batches = max(
            np.round(probs.shape[0] / batch_size), 1
        )  ## whatever's closer to the intended size
        batches = np.array_split(probs, n_batches)

    return batches


def conditional_batching(
    df: pd.DataFrame,
    batch_size: int,
    prob_cols: list[str],
    condition_columns: list[str] | None = None,
    true_label_col: str | None = None,
    remainder=False,
) -> tuple[list, list]:
    """Splits a dataframe of probabilities into batches, possibly conditioned on other
       variable columns, and calculates the reference distribution for each batch.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe containing probabilities in `prob_cols`, any categorical
        values to condition on in `condition_columns`, and possible true labels in
        `true_label_col`, to be split into batches.
    batch_size : int
        The target size of each batch. The output batches may not all exactly have
        size `batch_size`, but will be as close as possible.
    prob_cols : list[str]
        The columns of `df` that contain the probabilities to be batched.
    condition_columns : list[str] | None, optional
        The columns of `df` that batching should subdivide; by default None. If None,
        all rows of `df` can be batched together.
    true_label_col : str | None, optional
        The column of `df` that contains the true labels for calculating the reference
        distribution based on the true population, by default None. If None, the
        aggregate posterior is calculated as the reference distribution instead.
    remainder : bool, optional
        A setting for the batching method, by default False.
        For details, see the documentation in `batch_dataset`.

    Returns
    -------
    batches : list of array-like of shape (`[variable batch size]`, `n_classes`)
        The batched values of the `prob_cols` columns in `df`.
        Note that sizes of the batches may vary.
    reference_dists : list[array-like of shape `n_classes`]
        The reference distributions corresponding to `batches`, calculated either
        with `true_label_col` or the aggregate posterior.
    """

    grouped = (
        [("all", df)] if condition_columns is None else df.groupby(condition_columns)
    )

    batches_by_condition = []
    references_by_condition = []
    for condition, subset in grouped:
        print("conditional value", condition)
        print("condition size", len(subset))

        probs = subset[prob_cols]
        batches = batch_dataset(probs, batch_size, remainder=remainder)

        if true_label_col is not None:  # Calculate the ground truth marginal
            reference_dist = (
                np.bincount(subset[true_label_col], minlength=len(prob_cols))
                / subset.shape[0]
            )
            references = [reference_dist for _ in batches]
        else:
            references = [np.mean(batch, axis=0) for batch in batches]

        batches_by_condition.append(batches)
        references_by_condition.append(references)

    ## flatten the lists
    all_batches = [batch for b in batches_by_condition for batch in b]
    all_references = [ref for r in references_by_condition for ref in r]

    return all_batches, all_references


def preds_to_counts(predictions_df: pd.DataFrame, class_map=None) -> pd.DataFrame:
    """Count the predictions of each class in each column of `predictions_df`, ignoring
       any uncoded values.

    Parameters
    ----------
    predictions_df : pd.DataFrame of shape (`n_samples`, `n_methods`)
        The dataframe of discrete predictions by different methods.
    class_map : callable | dict | pd.Series | None, optional
        An argument to pd.Index.map() to rename class values, by default None.
        If `class_map` is a Series with a name, it will be used; otherwise, the
        index name will be "Class".

    Returns
    -------
    counts : pd.DataFrame of shape (`n_classes`, `n_methods`)
        the counts of each class in each column of `predictions_df`.
    """

    predictions_df = predictions_df.astype(int)

    counts = predictions_df.apply(pd.Series.value_counts).fillna(0).astype(int)

    ## is there a way we can easily set how many values there must be?
    counts.index.name = "Class"
    if isinstance(class_map, pd.Series):
        counts.index = class_map.loc[counts.index]
    elif class_map is not None:
        counts.index = counts.index.map(class_map)
    return counts


def recalibrate_probs(probs, true_labels) -> np.ndarray:
    """Applies Platt scaling to input probabilities to recalibrate them.

    Parameters
    ----------
    probs : array-like of shape (`n_samples`, `n_classes`)
        Each row represents a probability distribution, summing to 1.
    true_labels : array-like of shape (`n_samples`)
        True labels for each sample.

    Returns
    -------
    np.ndarray
        The recalibrated probabilities.
    """

    ## logistic regression is equivalent to Platt scaling
    calibrated_model = LogisticRegression(solver="saga", penalty=None).fit(
        posterior, true_labels
    )
    calibrated_probs = calibrated_model.predict_proba(posterior)
    return calibrated_probs
