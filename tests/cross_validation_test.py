import unittest
import pandas as pd
from locus_sampling.cross_validation import BalancedGroupKFold


class CrossValidationTestCase(unittest.TestCase):
    def test_split(self):
        group_id_file = "group_id_osu18.tsv"
        group_id_dfm = pd.read_table(group_id_file, header=0).set_index("name")

        feat_file = "feature_matrix_osu18_2.tsv"
        label_dfm = pd.read_table(feat_file, header=0).loc[:, ["name", "label"]].set_index("name")

        dfm = group_id_dfm.assign(label=label_dfm["label"])
        dfm = dfm.reset_index(level="name")

        bgkf = BalancedGroupKFold(n_splits=10, slop_allowed=0.5, random_state=1337)

        for train_index, test_index in bgkf.split(dfm, dfm["label"], dfm["group_id"]):
            self.assertTrue(len(train_index) > 0)
            self.assertTrue(len(test_index) > 0)

            train_groups = dfm.loc[train_index, "group_id"].unique()
            test_groups = dfm.loc[test_index, "group_id"].unique()

            intx_groups = set(train_groups).intersection(set(test_groups))
            self.assertEqual(intx_groups, set())


if __name__ == '__main__':
    unittest.main()
