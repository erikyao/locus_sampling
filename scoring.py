import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import make_scorer
from .exceptions import NoPostiveInGroupError


def avg_rank_func(y, y_pred, groups=None, pos_label=1, method="average", exclude_loners=False):
    """
    For each group, rank the probabilities.
    Then pick the ranks of positive cases in each group, calculate their average.
    The smaller the average rank is, the better performance the classifier has.

    Compatibility
    -------------
    This function is compatible with the formula (which should be revised) in the CERENKOV paper.
    This function is NOT compatible with Steve's R code. (See comment of `avg_rank_func2`)
    
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
    :param groups: pd.Series of length n_samples, with original index from label column or array of shape (n_samples,).
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
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
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
        if all(labels_in_group != pos_label):
            raise NoPostiveInGroupError("No positive case in group '{}'".format(group_id))

        probs_in_group = y_pred[index_in_group]
        # `scipy.stats.rankdata` cannot rank descendingly
        # ranks_in_group = rankdata(probs_in_group, method=method)
        """
        R code from Steve:

            g_rank_by_score_decreasing <- function(x) {
                length(x) - rank(x, ties.method="average") + 1
            }
        """
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


def avg_rank_scorer(groups=None, pos_label=1, method="average", exclude_loners=False):
    return make_scorer(avg_rank_func, greater_is_better=False, needs_proba=True, needs_threshold=False, 
                       groups=groups, pos_label=pos_label, method=method, exclude_loners=exclude_loners)


def avg_rank_func2(y, y_pred, groups=None, pos_label=1, method="average"):
    """
    For each rSNP, rank its probability among all the other cSNPs in its locus, 
        ignoring other rSNPs (if any) in the same locus. Then average all the rSNP ranks.
    
    E.g. suppose locus L has 2 rSNPs and 2 cSNPs, (r1,r2,c1,c2), with predicted probabilities (Pr1,Pr2,Pc1,Pc2).

    For r1, output the rank (denoted by Rr1) of Pr1 among (Pr1,Pc1,Pc2), ignoring Pr2.
    For r2, output the rank (denoted by Rr2) of Pr2 among (Pr2,Pc1,Pc2), ignoring Pr1.
    The average rank would be (Rr1+Rr2)/2
    
    The smaller the average rank is, the better performance the classifier has.

    Compatibility
    -------------
    This function is NOT compatible with the formula (which should be revised) in the CERENKOV paper.
    This function is compatible with Steve's R code::

        g_rank_by_score_decreasing <- function(x) {
            length(x) - rank(x, ties.method="average") + 1
        }

        g_calculate_avgrank <- if (g_par$flag_locus_sampling) {
            g_make_calculate_avgrank_within_groups(g_snp_locus_ids,
                                                g_locus_to_snp_ids_map_list,
                                                g_rank_by_score_decreasing)
        } else {
            NULL
        }

        g_make_calculate_avgrank_within_groups <- function(p_case_group_ids, p_group_to_case_ids_map_list, p_rank_by_score_decreasing) {
            stopifnot(is.list(p_group_to_case_ids_map_list))

            p_case_group_ids
            p_group_to_case_ids_map_list
            
            function(p_case_scores, p_labels, p_case_ids) {
                ## need the indices of the rSNPs in the full set of 15,331 SNPs
                inds_pos <- p_case_ids[which(p_labels==1)]

                inds_map <- match(1:length(p_case_group_ids), p_case_ids)
                
                ## for each rSNP....
                mean(sapply(inds_pos, function(p_ind_pos) {
                    ## get the group ID of the rSNP
                    group_id <- p_case_group_ids[p_ind_pos]

                    ## get the set of all SNPs in that group
                    group_case_inds <- p_group_to_case_ids_map_list[[group_id]]
                    stopifnot(! is.null(group_case_inds))

                    inds_cases_within_fold_set <- inds_map[group_case_inds]
                    stopifnot(! is.na(inds_cases_within_fold_set))
                    
                    ## use group_case_inds to index into 
                    scores_for_group_cases <- p_case_scores[inds_cases_within_fold_set]
                    stopifnot(! is.na(scores_for_group_cases))
                    
                    labels_for_group_cases <- p_labels[inds_cases_within_fold_set]
                    stopifnot(! is.na(labels_for_group_cases))

                    group_case_inds_analyze <- c(p_ind_pos, group_case_inds[which(labels_for_group_cases==0)])
                    stopifnot(! is.na(group_case_inds_analyze))
                    
                    scores_to_analyze <- p_case_scores[inds_map[group_case_inds_analyze]]
                    stopifnot(! is.na(scores_to_analyze))

                    p_rank_by_score_decreasing(scores_to_analyze)[1]
                }))
            }
        }

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
    :param groups: pd.Series of length n_samples, with original index from label column or array of shape (n_samples,).
                   The group ID for each label.
                   If it does not match y's dimensions, subset by `y.index.values`.
                   (This is an ugly hack because I cannot get `groups[train_index]` or
                   `groups[test_index]` in cross validation)
    :param pos_label: int or str. The TRUE label in y
    :param method: The method argument controls how ranks are assigned to equal values.
                   Identical to the `method` parameter of `pandas.Series.rank`
                   See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
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
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
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

    # ----- Part 3: Calculate the Rank For Each rSNP ----- #
    def _cal_rank():
        unique_groups = np.unique(groups)
        for group_id in unique_groups:
            index_in_group = np.where(groups == group_id)[0]  # np.where returns a one-elemented tuple

            labels_in_group = y[index_in_group]
            if all(labels_in_group != pos_label):
                raise NoPostiveInGroupError("No positive case in group '{}'".format(group_id))

            probs_in_group = y_pred[index_in_group]

            pos_index_in_group = np.where(labels_in_group == pos_label)[0]
            neg_index_in_group = np.where(labels_in_group != pos_label)[0]

            pos_probs_in_group = probs_in_group[pos_index_in_group]
            neg_probs_in_group = probs_in_group[neg_index_in_group]
            
            for pos_prob in pos_probs_in_group:
                engaged_probs = [pos_prob] + list(neg_probs_in_group)
                
                # `scipy.stats.rankdata` cannot rank descendingly, so we tweak it this way
                pos_rank = len(engaged_probs) - rankdata(engaged_probs, method=method)[0] + 1

                yield pos_rank

    # ----- Part 4: Calculate Average Ranks ----- #
    pos_ranks = np.array(list(_cal_rank()))
    return pos_ranks.mean()

def avg_rank_scorer2(groups=None, pos_label=1, method="average"):
    return make_scorer(avg_rank_func2, greater_is_better=False, needs_proba=True, needs_threshold=False, 
                       groups=groups, pos_label=pos_label, method=method)