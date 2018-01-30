import pandas as pd


def __generate_group_id(snp_bed_dfm, cutoff_bp):
    """

    :param snp_bed_dfm: A pandas DataFrame from SNPs' BED file,
                        having 3 columns, "name", "chrom" and "chromStart" at least
    :param cutoff_bp: sequential SNPs that are `cutoff_bp` away (in base-pair)
                      are partitioned into 2 groups
    :return:
    """
    for chrom, sub_dfm in snp_bed_dfm.groupby("chrom"):
        sub_dfm = sub_dfm.sort_values("chromStart", ascending=True)

        coord_diff = sub_dfm["chromStart"] - sub_dfm["chromStart"].shift(1)

        group_id = (coord_diff > cutoff_bp).cumsum(axis=0)
        group_id = chrom + "_" + group_id.astype(str)

        yield group_id


def assign_group_id(snp_bed_dfm, cutoff_bp=50000, col_name="group_id"):
    """

    :param snp_bed_dfm: A pandas DataFrame from SNPs' BED file,
                        having 3 columns, "name", "chrom" and "chromStart" at least
    :param cutoff_bp: sequential SNPs that are `cutoff_bp` away (in base-pair)
                      are partitioned into 2 groups
    :param col_name: the column name for the group IDs in the returned DataFrame
    :return:
    """
    group_id_list = list(__generate_group_id(snp_bed_dfm, cutoff_bp))
    group_id_series = pd.concat(group_id_list, axis=0)

    # `DataFrame.assign` will match values by indices
    return snp_bed_dfm.assign(**{col_name: group_id_series})


# if __name__ == '__main__':
#     bed_file = "ground_truth_snps_with_names.bed"
#
#     bed_dfm = pd.read_table(bed_file,
#                             header=None,
#                             comment='#',
#                             names=['chrom', 'chromStart', 'chromEnd', 'name'])
#
#     ret_dfm = assign_group_id(bed_dfm)
#
#     ret_dfm.to_csv("group_id_osu18.tsv", sep="\t", header=True, index=False)
