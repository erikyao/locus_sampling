## Cross Validation Utilities

- `BalancedGroupKFold` is a K-fold iterator variant with non-overlapping groups.
    - The same group will not appear in two different folds.
    - The folds are approximately balanced in the sense that the number of positive cases is approximately the same in each fold.
- `FixedReplicatedKFold` simply outputs partitions recorded in a tsv file.

## AVGRANK

`avg_rank_func2` calculates the AVGRANK scores while `avg_rank_scorer2` returns a scorer object. Both functions are scikit-learn compatible.

## Reference

- [CERENKOV: Computational Elucidation of the Regulatory Noncoding Variome](https://dl.acm.org/citation.cfm?id=3107414)
