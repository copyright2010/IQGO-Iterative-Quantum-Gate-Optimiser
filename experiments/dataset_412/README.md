# Dataset 412 experiments

This folder is the IQGO version and the experiments used for the dataset 412 results, obtained in the Tutorial - Neurocomputing.

`IQGO_module/` is the package that produced those runs. Run the scripts from this directory so Python imports this copy.

Gate combinations are chosen with training accuracy (`accuracies = accuracies_train` in `IQGO_train.fit`). After the circuit is selected, `predict_validate` fits on the full training split and scores the held-out validation split.

## Setup

Python 3.12.

```sh
pip install -r requirements.txt
```

## Scripts

- `iqgo_412.py` reads `412_fig6_r2_param_estimates.csv`, fits a 4-layer circuit, and writes `results_new_iqgo.csv`.
- `kfold_zz.py` runs the ZZ feature-map kernel on the same table.
- `SVC_43.py` and `svc_851.py` run the classical baselines on the same table.
- `p_value_statistics.py` computes the paired comparisons and writes `paired_balanced_accuracy_results.csv`, `iqgo_pairwise_statistics.csv`, and `iqgo_paper_statistics.csv`.

```sh
python iqgo_412.py
```

## Saved outputs

`results_new_iqgo.csv` is the IQGO run. `results.xlsx` and `results_newest.xlsx` are the saved workbooks. The three statistics CSVs are the tables from `p_value_statistics.py`.
