# Uni-Dock Score QC

This is a score and pose-file sanity audit for Uni-Dock phase 1 outputs. It flags values that are unsuitable for enrichment ranking; it does not rerun docking or delete poses.

## Inputs

- score_source: `runs/mapk1_lit_pcba_full/tables/mapk1_lit_pcba_full_unidock_raw.csv`
- total_rows: 309230
- flagged_rows: 6936
- valid_for_ranking: 302294
- best_per_ligand_rows: 60960

## Score Ranges

- raw_min: -77.089
- raw_median: -5.933
- raw_max: 216.089
- valid_min: -29.288
- valid_median: -5.961
- valid_max: -0.001

## Flag Counts

- score_missing: 0
- score_nonfinite: 0
- score_positive: 6910
- score_extreme_low: 26
- score_extreme_high: 3730
- pose_file_missing: 0
- status_not_success: 0

## Receptor Summary

```text
pdb_id  total  valid  flagged  score_min  score_median  score_max  label_active  label_inactive
  4QTA  61846  60640     1206    -56.080        -6.968    183.894           302           61544
  4ZZN  61846  60457     1389    -52.634        -6.068    216.089           302           61544
  5WP1  61846  60247     1599    -42.291        -5.526    192.511           302           61544
  6SLG  61846  60518     1328    -77.089        -5.794    189.352           302           61544
  8AOJ  61846  60432     1414    -50.264        -5.657    204.293           302           61544
```

## Warnings

- Loaded raw context table runs/mapk1_lit_pcba_full/tables/mapk1_lit_pcba_full_unidock_raw.csv because runs/mapk1_lit_pcba_full/tables/mapk1_lit_pcba_full_unidock_scores.csv was missing columns: ['docking_id', 'ligand_pdbqt', 'receptor_pdbqt']

## Top 20 Best Valid Scores

```text
ligand_id activity_label pdb_id   score                                                       output_pose_file
  3717802       inactive   8AOJ -29.288  runs/mapk1_lit_pcba_full/poses/unidock/8aoj/0006129_3717802_out.pdbqt
  4244747       inactive   4QTA -27.956  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0052108_4244747_out.pdbqt
  7972657       inactive   4QTA -27.159  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0015680_7972657_out.pdbqt
 26750727       inactive   4QTA -25.820 runs/mapk1_lit_pcba_full/poses/unidock/4qta/0053821_26750727_out.pdbqt
 26752503       inactive   4QTA -24.827 runs/mapk1_lit_pcba_full/poses/unidock/4qta/0002600_26752503_out.pdbqt
  4250824       inactive   4QTA -24.707  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0057684_4250824_out.pdbqt
   858393       inactive   6SLG -24.346   runs/mapk1_lit_pcba_full/poses/unidock/6slg/0059902_858393_out.pdbqt
  3713621       inactive   4ZZN -23.357  runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0017773_3713621_out.pdbqt
   844852       inactive   8AOJ -23.135   runs/mapk1_lit_pcba_full/poses/unidock/8aoj/0031380_844852_out.pdbqt
   842662         active   4QTA -23.078   runs/mapk1_lit_pcba_full/poses/unidock/4qta/0011968_842662_out.pdbqt
  4255873       inactive   4QTA -22.870  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0029759_4255873_out.pdbqt
  3716592       inactive   4QTA -22.436  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0039787_3716592_out.pdbqt
  4250031       inactive   4ZZN -20.786  runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0030034_4250031_out.pdbqt
  3715898       inactive   4ZZN -19.914  runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0051417_3715898_out.pdbqt
   844003       inactive   4QTA -19.449   runs/mapk1_lit_pcba_full/poses/unidock/4qta/0019344_844003_out.pdbqt
  4246826       inactive   8AOJ -19.260  runs/mapk1_lit_pcba_full/poses/unidock/8aoj/0047019_4246826_out.pdbqt
   862405       inactive   6SLG -19.245   runs/mapk1_lit_pcba_full/poses/unidock/6slg/0019187_862405_out.pdbqt
 17389521       inactive   4QTA -19.187 runs/mapk1_lit_pcba_full/poses/unidock/4qta/0009799_17389521_out.pdbqt
  3717814       inactive   6SLG -19.036  runs/mapk1_lit_pcba_full/poses/unidock/6slg/0017404_3717814_out.pdbqt
 26749938       inactive   6SLG -18.836 runs/mapk1_lit_pcba_full/poses/unidock/6slg/0021814_26749938_out.pdbqt

```

## Top 20 Most Positive Or Extreme Scores

```text
ligand_id activity_label pdb_id   score                   qc_flag_reasons                                                       output_pose_file
  4255001       inactive   4ZZN 216.089 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0018149_4255001_out.pdbqt
  4255873       inactive   8AOJ 204.293 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/8aoj/0029759_4255873_out.pdbqt
  7966663       inactive   5WP1 192.511 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/5wp1/0054291_7966663_out.pdbqt
  4248296       inactive   5WP1 190.807 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/5wp1/0052804_4248296_out.pdbqt
   843410       inactive   6SLG 189.352 score_positive;score_extreme_high   runs/mapk1_lit_pcba_full/poses/unidock/6slg/0044146_843410_out.pdbqt
  7965124       inactive   4QTA 183.894 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0028728_7965124_out.pdbqt
 11112652       inactive   4ZZN 177.597 score_positive;score_extreme_high runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0012350_11112652_out.pdbqt
  3713287       inactive   4ZZN 174.121 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0057712_3713287_out.pdbqt
  4249804       inactive   4QTA 167.808 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0036088_4249804_out.pdbqt
  4250941       inactive   4QTA 166.815 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0025629_4250941_out.pdbqt
  7968265         active   6SLG 165.791 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/6slg/0054843_7968265_out.pdbqt
  7966957       inactive   4ZZN 163.665 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0056361_7966957_out.pdbqt
  7973763       inactive   4QTA 163.401 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/4qta/0059889_7973763_out.pdbqt
  4262474       inactive   5WP1 158.042 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/5wp1/0038484_4262474_out.pdbqt
  4263731       inactive   8AOJ 150.028 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/8aoj/0035655_4263731_out.pdbqt
  4247134       inactive   5WP1 148.094 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/5wp1/0052403_4247134_out.pdbqt
  7967126       inactive   8AOJ 146.857 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/8aoj/0026555_7967126_out.pdbqt
  3711353       inactive   5WP1 144.728 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/5wp1/0002959_3711353_out.pdbqt
  4259267       inactive   5WP1 144.412 score_positive;score_extreme_high  runs/mapk1_lit_pcba_full/poses/unidock/5wp1/0039696_4259267_out.pdbqt
   849819       inactive   4ZZN 143.515 score_positive;score_extreme_high   runs/mapk1_lit_pcba_full/poses/unidock/4zzn/0038655_849819_out.pdbqt

```

## Missing Pose Or Score Examples

```text
None

```
