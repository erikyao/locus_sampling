# In python 2.7, the default `/` operator is integer division if inputs are integers.
# If you want float division, use this special import
# from __future__ import division
import pandas as pd
import numpy as np
import numbers
import math
from numpy.random import RandomState
from sklearn.model_selection import BaseCrossValidator
from .exceptions import NoFoldAvailableError, NoPostiveInGroupError


class _FoldMeter:
    """
    A meter to monitor the volume of a fold.
    Fold volumn is measured in two aspects: number of all cases, and number of positive cases in this fold
    """

    def __init__(self, fold_id, n_case, n_pos, max_n_case, max_n_pos):
        """
        Constructor of `FoldMeter`

        :param fold_id: integer ID of this fold
        :param n_case: current number of all cases (positive and negative) in this fold
        :param n_pos: current number of positive cases in this folder
        :param max_n_case: maximum number of all cases
        :param max_n_pos: maximum number of positive cases
        """
        self.fold_id = fold_id
        self.n_case = n_case
        self.n_pos = n_pos
        self.max_n_case = max_n_case
        self.max_n_pos = max_n_pos

    def allow(self, extra_n_case, extra_n_pos):
        """
        Check if this fold can hold an extra volume

        :param extra_n_case: additive number of cases
        :param extra_n_pos: additive number of positive cases
        :return: True if feasible; False otherwise
        """
        n_case_allows = self.n_case + extra_n_case <= self.max_n_case
        n_pos_allows = self.n_pos + extra_n_pos <= self.max_n_pos

        return n_case_allows and n_pos_allows

    def add(self, extra_n_case, extra_n_pos):
        """
        Update this fold's volume

        :param extra_n_case:
        :param extra_n_pos:
        :return: void
        """
        self.n_case += extra_n_case
        self.n_pos += extra_n_pos

    def vacancy_rates(self):
        """
        Return the vacancy rates on all cases and positive cases
        :return: a tuple of floats; all-case vacancy rate comes first
        """
        return 1 - self.n_case / self.max_n_case, 1 - self.n_pos / self.max_n_pos


class _FoldMeterArray:
    """
    An array of `FolderMeter`s
    """

    def __init__(self, k, fold_max_n_case, fold_max_n_pos, random_state=None):
        """
        Constructor of `FoldMeterArray`

        :param k: number of `FoldMeter`s
        :param fold_max_n_case: maximum number of all cases for each `FoldMeter`
        :param fold_max_n_pos: maximum number of positive cases for each `FoldMeter`
        :param random_state: an RNG seed
        """
        self.fold_ids = np.arange(k)
        self._folds = [_FoldMeter(fold_id=i,
                                  n_case=0,
                                  n_pos=0,
                                  max_n_case=fold_max_n_case,
                                  max_n_pos=fold_max_n_pos) for i in self.fold_ids]
        self.rs = RandomState(random_state)

    def __getitem__(self, fold_id):
        return self._folds[fold_id]

    def __len__(self):
        return len(self._folds)

    def _available_fold_ids(self, extra_n_case, extra_n_pos):
        """
        Check if any fold can hold an extra volume. Return IDs of all such folds

        :param extra_n_case: additive number of cases
        :param extra_n_pos: additive number of positive cases
        :return: a list of fold IDs
        """
        return [f.fold_id for f in self if f.allow(extra_n_case, extra_n_pos)]

    def allocate(self, extra_n_case, extra_n_pos):
        """
        Find an specific fold to hold an extra volume, update its volume, and then return ID of this folds.
        If no fold is available, raise a `NoFoldAvailableError`

        :param extra_n_case: additive number of cases
        :param extra_n_pos: additive number of positive cases
        :raise `NoFoldAvailableError` if no fold available
        :return: the selected fold ID
        """
        available_fold_ids = self._available_fold_ids(extra_n_case, extra_n_pos)

        if len(available_fold_ids) > 1:
            # When multiple folds are available, sample a fold ID by fold vacancy rates
            probs = np.array([np.multiply(*self[i].vacancy_rates()) for i in available_fold_ids])
            norm_probs = probs / probs.sum()  # np.random.choice need probabilities summed up to 1.0
            selected = self.rs.choice(available_fold_ids, size=1, p=norm_probs)[0]
        elif len(available_fold_ids) == 1:
            selected = available_fold_ids[0]
        else:
            raise NoFoldAvailableError("extra_n_case == {}; extra_n_pos == {}".format(extra_n_case, extra_n_case))

        self[selected].add(extra_n_case, extra_n_pos)

        return selected


class BalancedGroupKFold(BaseCrossValidator):
    """K-fold iterator variant with non-overlapping groups.

    The same group will not appear in two different folds.

    The folds are approximately balanced in the sense that the number of
    positive cases is approximately the same in each fold.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds.
    slop_allowed: float, default=0.5
        Scale parameter of maximum fold volume.
        Average fold volume is the number of cases divided by the number of folds.
        Maximum fold volume is (1 + slop_allowed) times average fold volume

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 0, 1, 0])
    >>> groups = np.array([0, 0, 2, 2])
    >>> bgkf = BalancedGroupKFold(n_splits=2, slop_allowed=0.5)
    >>> bgkf.get_n_splits(X, y, groups)
    2
    >>> print(bgkf)
    BalancedGroupKFold(n_splits=2, random_state=None, slop_allowed=0.5)
    >>> for train_index, test_index in bgkf.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...     print(X_train, X_test, y_train, y_test)
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [1 0] [1 0]
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [1 0] [1 0]

    See also
    --------
    sklearn.model_selection.StratifiedKFold
        Takes group information into account to avoid building folds with
        imbalanced class distributions (for binary or multiclass
        classification tasks).

    sklearn.model_selection.GroupKFold: K-fold iterator variant with non-overlapping groups.
    """
    def __init__(self, n_splits=3, slop_allowed=0.5, random_state=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. %s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError("k-fold cross-validation requires at least one train/test split "
                             "by setting n_splits=2 or more, got n_splits={0}.".format(n_splits))

        self.n_splits = n_splits
        self.slop_allowed = slop_allowed
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # ----- Part 1: Check Conditions ----- #
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        if isinstance(y, pd.Series):
            y = y.values.ravel()

        unique_groups, inverse_index = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        for group_id in unique_groups:
            index_in_group = np.where(groups == group_id)[0]  # np.where returns a one-elemented tuple
            labels_in_group = y[index_in_group]
            if all(labels_in_group != 1):
                raise NoPostiveInGroupError("No positive case in group '{}'".format(group_id))

        # ----- Part 2: Prepare `FoldMeterArray` ----- #

        n_fold = self.n_splits

        X_n_case = X.shape[0]
        y_n_pos = sum(y == 1)

        fold_max_n_case = math.ceil((1 + self.slop_allowed) * X_n_case / n_fold)
        # Revised `fold_max_n_pos`
        fold_max_n_pos = math.ceil((1 + self.slop_allowed) * y_n_pos / n_fold)
        # Old `fold_max_n_pos`
        # fold_max_n_pos = math.ceil((1 + self.slop_allowed) * fold_max_n_case * y_n_pos / X_n_case)

        fma = _FoldMeterArray(n_fold, fold_max_n_case, fold_max_n_pos, random_state=self.random_state)

        # ----- Part 3: Allocate Each Group To A Fold ----- #

        # Weight groups by their number of occurrences
        n_case_per_group = np.bincount(inverse_index)

        # Distribute the biggest groups first
        desc_unique_group_index = np.argsort(n_case_per_group)[::-1]

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups), dtype=int)

        for group_index in desc_unique_group_index:
            group_id = unique_groups[group_index]  # biggest groups come first

            index_in_group = np.where(groups == group_id)[0]  # np.where returns a one-elemented tuple
            labels_in_group = y[index_in_group]

            group_n_case = len(index_in_group)
            group_n_pos = sum(labels_in_group == 1)

            try:
                fold_id = fma.allocate(group_n_case, group_n_pos)
            except NoFoldAvailableError as nfae:
                raise NoFoldAvailableError("Cannot allocate group {}".format(group_id)) from nfae

            group_to_fold[group_index] = fold_id

        # ----- Part 4: Yield Folds ----- #

        folds = group_to_fold[inverse_index]
        for f in range(self.n_splits):
            yield np.where(folds == f)[0]
