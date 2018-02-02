import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from .exceptions import NoPostiveInGroupError


def avg_rank_func(y, y_pred, groups=None, pos_label=1, method="min", exclude_loners=False):
    """
    For each group, rank the probabilities.
    Then pick the ranks of positive cases in each group, calculate their average.
    The smaller the average rank is, the better performance the classifier has.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([[0.2, 0.8],
    ...                    [0.1, 0.9],
    ...                    [0.3, 0.7],
    ...                    [0.4, 0.6]])
    >>> groups = np.array([0, 0, 1, 1])
    >>> avg_rank(y, y_pred, groups)
    1.5

    :param y: pd.Series of length n_samples, with original index from label column or array of shape (n_samples,).
              The ground-true labels.
    :param y_pred: array of shape (n_samples,) or (n_samples, n_classes).
                   The predicted probabilities
    :param groups: array of shape (n_samples,).
                   The group ID for each label.
                   If it does not match y's dimensions, subset by `y.index.values`.
                   (This is an ugly hack because I cannot get `groups[train_index]` or
                   `groups[test_index]` in cross validation)
    :param pos_label: int or str. The TRUE label in y
    :param method: The method argument controls how ranks are assigned to equal values.
                   Identical to the `method` parameter of `pandas.Series.rank`
                   See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    :param exclude_loners: boolean. Whether going to exclude groups of one case in calculating average ranks
    :return: average rank of positive cases
    """
    # ----- Part 1: Check Conditions ----- #
    if y_pred.ndim > 2:
        raise ValueError("y_pred should be an array of shape (n_samples,) "
                         "or (n_samples, n_classes), got a %d-D array" % y_pred.ndim)

    if groups is None:
        raise ValueError("The 'groups' parameter should not be None.")

    # It's possible that `groups` contains group IDs for all cases
    #   while `y` is only a sequence of labels of a fold
    if len(groups) > len(y):
        try:
            groups = groups[y.index.values]
        except AttributeError as ae:
            raise AttributeError("Subsetting groups from y's index values. However y has no index.") from ae

    if isinstance(y, pd.Series):
        y = y.values.ravel()

    # ----- Part 2: Pick Positive Dimensions ----- #
    labels = np.unique(y)  # returns SORTED unique values
    pos_dim = np.where(labels == pos_label)[0]

    # If y_pred is of shape (n_samples, n_classes), pick the probs of positive predictions
    if y_pred.shape[1] == 2:
        """
        E.g.

        y_pred = np.array([[0.2, 0.8],
                           [0.1, 0.9],
                           [0.3, 0.7],
                           [0.4, 0.6]])
        y_pred[:, 1] == y_pred[:, (False, True)] = array([[0.8],
                                                          [0.9],
                                                          [0.7],
                                                          [0.6]])
        y_pred[:, 1].ravel() == array([0.8, 0.9, 0.7, 0.6])
        """
        y_pred = y_pred[:, pos_dim].ravel()

    # ----- Part 3: Calculate Ranks Within Each Group ----- #
    ranks = np.zeros(len(y), dtype=int)

    unique_groups = np.unique(groups)
    for group_id in unique_groups:
        index_in_group = np.where(groups == group_id)[0]  # np.where returns a one-elemented tuple

        labels_in_group = y[index_in_group]
        if all(labels_in_group != 1):
            raise NoPostiveInGroupError("No positive case in group '{}'".format(group_id))

        probs_in_group = y_pred[index_in_group]
        # `scipy.stats.rankdata` cannot rank descendingly
        # ranks_in_group = rankdata(probs_in_group, method=method)
        ranks_in_group = pd.Series(probs_in_group).rank(method=method, ascending=False).values
        ranks[index_in_group] = ranks_in_group

    # ----- Part 4: Exclude Loners ----- #
    pos_flag = (y == pos_label)

    if exclude_loners:
        loner_flag = np.full(len(y), False, dtype=bool)
        for group_id in unique_groups:
            index_in_group = np.where(groups == group_id)[0]
            if len(index_in_group) == 1:
                loner_flag[index_in_group] = True

        pos_flag = np.logical_and(pos_flag, np.logical_not(loner_flag))

    # ----- Part 5: Calculate Average Ranks ----- #
    # pos_index = np.where(y == pos_label)[0]
    pos_ranks = ranks[pos_flag]
    return pos_ranks.mean()


def avg_rank_scorer(groups=None):
    return make_scorer(avg_rank_func, greater_is_better=False, needs_proba=True, groups=groups)

