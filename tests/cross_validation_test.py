import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from locus_sampling.cross_validation import BalancedGroupKFold


class CrossValidationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        group_id_file = "group_id_osu18.tsv"
        group_id_dfm = pd.read_table(group_id_file, header=0).set_index("name")

        feat_file = "feature_matrix_osu18_2.tsv"
        label_dfm = pd.read_table(feat_file, header=0).loc[:, ["name", "label"]].set_index("name")

        cls.dfm = group_id_dfm.assign(label=label_dfm["label"])
        cls.dfm = cls.dfm.reset_index(level="name")

    def test_split(self):
        bgkf = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1337)

        for train_index, test_index in bgkf.split(self.dfm, self.dfm["label"], self.dfm["group_id"]):
            self.assertTrue(len(train_index) > 0)
            self.assertTrue(len(test_index) > 0)

            train_groups = self.dfm.loc[train_index, "group_id"].unique()
            test_groups = self.dfm.loc[test_index, "group_id"].unique()

            intx_groups = set(train_groups).intersection(set(test_groups))
            self.assertEqual(intx_groups, set())

    def test_random_state(self):
        # CV with the same random_state should yield the same partition
        cv1 = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1337)
        cv2 = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1337)
        for (train_index_1, test_index_1), (train_index_2, test_index_2) \
                in zip(cv1.split(self.dfm, self.dfm["label"], self.dfm["group_id"]),
                       cv2.split(self.dfm, self.dfm["label"], self.dfm["group_id"])):
            assert_array_equal(train_index_1, train_index_2)
            assert_array_equal(test_index_1, test_index_2)

        # CV with different random_states should yield different partition
        cv1 = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1337)
        cv2 = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1339)
        for (train_index_1, test_index_1), (train_index_2, test_index_2) \
                in zip(cv1.split(self.dfm, self.dfm["label"], self.dfm["group_id"]),
                       cv2.split(self.dfm, self.dfm["label"], self.dfm["group_id"])):
            if len(train_index_1) == len(train_index_2):
                self.assertTrue(any(np.not_equal(train_index_1, train_index_2)))
            if len(test_index_1) == len(test_index_2):
                self.assertTrue(any(np.not_equal(test_index_1, test_index_2)))

            # if index lengths are different, the partition cannot be the same

    def test_reset_random_state(self):
        # CV with different random_states should yield different partition
        cv1 = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1337)
        cv2 = clone(cv1, safe=False)
        cv2.random_state = 1339
        for (train_index_1, test_index_1), (train_index_2, test_index_2) \
                in zip(cv1.split(self.dfm, self.dfm["label"], self.dfm["group_id"]),
                       cv2.split(self.dfm, self.dfm["label"], self.dfm["group_id"])):
            if len(train_index_1) == len(train_index_2):
                self.assertTrue(any(np.not_equal(train_index_1, train_index_2)))
            if len(test_index_1) == len(test_index_2):
                self.assertTrue(any(np.not_equal(test_index_1, test_index_2)))

                # if index lengths are different, the partition cannot be the same


if __name__ == '__main__':
    unittest.main()
