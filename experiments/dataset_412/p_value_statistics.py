import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon,
    shapiro,
    friedmanchisquare,
    t,
)
from statsmodels.stats.multitest import multipletests


# ============================================================
# SEEDS
# ============================================================

seeds = np.array([
    1155,
    626,
    5451,
    123,
    4332,
    234,
    4,
    48,
    1233,
    343,
])


# ============================================================
# BALANCED ACCURACY PER SEED
#
# IMPORTANT:
# All arrays below must be in exactly the same seed order.
#
# These are BALANCED ACCURACY values.
#
# For the classical experiments, use the values printed in
# the "acc_test" column because scor_val was calculated as:
#
# balanced_accuracy_score(val_labels, pred)
#
# Do NOT use the displayed "Recall" column, because that was
# calculated using average="weighted".
# ============================================================


# ============================================================
# IQGO
# ============================================================

iqgo = np.array([
    0.846083,  # seed 1155
    0.846676,  # seed 626
    0.848015,  # seed 5451
    0.837601,  # seed 123
    0.843537,  # seed 4332
    0.848308,  # seed 234
    0.814163,  # seed 4
    0.853672,  # seed 48
    0.805085,  # seed 1233
    0.856592,  # seed 343
])


# ============================================================
# TabPFN
# ============================================================

tabpfn = np.array([
    0.847732,
    0.852651,
    0.848167,
    0.871013,
    0.886531,
    0.874013,
    0.858544,
    0.851608,
    0.816804,
    0.822107,
])


# ============================================================
# CatBoost
# ============================================================

catboost = np.array([
    0.809483,
    0.816014,
    0.830671,
    0.859430,
    0.864236,
    0.861974,
    0.840557,
    0.812208,
    0.807522,
    0.820468,
])


# ============================================================
# SVC Polynomial
# ============================================================

svc_poly = np.array([
    0.835260,
    0.815628,
    0.828479,
    0.825283,
    0.864579,
    0.867261,
    0.829573,
    0.811837,
    0.823603,
    0.825628,
])


# ============================================================
# SVC RBF
# ============================================================

svc_rbf = np.array([
    0.836861,
    0.819172,
    0.839575,
    0.785816,
    0.819655,
    0.851109,
    0.798276,
    0.847856,
    0.815502,
    0.836022,
])


# ============================================================
# SVC Linear
# ============================================================

svc_linear = np.array([
    0.629957,
    0.643524,
    0.669482,
    0.611416,
    0.641360,
    0.635220,
    0.583448,
    0.642490,
    0.666094,
    0.629770,
])


# ============================================================
# MODELS
# ============================================================

models = {
    "TabPFN": tabpfn,
    "CatBoost": catboost,
    "SVC Polynomial": svc_poly,
    "SVC Linear": svc_linear,
    "SVC RBF": svc_rbf,
}


# ============================================================
# SANITY CHECKS
# ============================================================

expected_n = len(seeds)

assert len(iqgo) == expected_n

for name, values in models.items():
    assert len(values) == expected_n, (
        f"{name} has {len(values)} results, "
        f"but expected {expected_n}"
    )


# ============================================================
# OPTIONAL: DISPLAY PAIRED DATA
# ============================================================

paired_data = pd.DataFrame({
    "Seed": seeds,
    "IQGO": iqgo,
    **models,
})

print("=" * 90)
print("PAIRED BALANCED ACCURACY DATA")
print("=" * 90)

print(paired_data.to_string(index=False))


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("DESCRIPTIVE STATISTICS")
print("=" * 90)

print(
    f"IQGO: "
    f"{iqgo.mean() * 100:.2f}% "
    f"± {iqgo.std(ddof=1) * 100:.2f}%"
)

for name, values in models.items():
    print(
        f"{name}: "
        f"{values.mean() * 100:.2f}% "
        f"± {values.std(ddof=1) * 100:.2f}%"
    )


# ============================================================
# FRIEDMAN OMNIBUS TEST
#
# IQGO + 5 baselines = 6 algorithms
# ============================================================

friedman_stat, friedman_p = friedmanchisquare(
    iqgo,
    tabpfn,
    catboost,
    svc_poly,
    svc_linear,
    svc_rbf,
)

n = len(iqgo)
k = 1 + len(models)

kendalls_w = friedman_stat / (n * (k - 1))


print("\n" + "=" * 90)
print("FRIEDMAN OMNIBUS TEST")
print("=" * 90)

print(f"N runs            = {n}")
print(f"Algorithms        = {k}")
print(f"Chi-square        = {friedman_stat:.6f}")
print(f"df                = {k - 1}")
print(f"p                 = {friedman_p:.12g}")
print(f"Kendall's W       = {kendalls_w:.6f}")


# ============================================================
# PAIRWISE COMPARISONS AGAINST IQGO
#
# Difference definition:
#
#     baseline - IQGO
#
# Positive difference:
#     baseline performs better
#
# Negative difference:
#     IQGO performs better
# ============================================================

results = []

for name, values in models.items():

    diff = values - iqgo

    # --------------------------------------------------------
    # Paired t-test
    # --------------------------------------------------------

    t_stat, p_t = ttest_rel(
        values,
        iqgo,
    )

    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------

    w_stat, p_w = wilcoxon(
        values,
        iqgo,
        alternative="two-sided",
    )

    # --------------------------------------------------------
    # Shapiro-Wilk test on PAIRED DIFFERENCES
    # --------------------------------------------------------

    shapiro_stat, shapiro_p = shapiro(diff)

    # --------------------------------------------------------
    # Mean paired difference
    # --------------------------------------------------------

    mean_diff = diff.mean()

    # --------------------------------------------------------
    # 95% CI for paired mean difference
    # --------------------------------------------------------

    n_pair = len(diff)

    sd_diff = diff.std(ddof=1)
    se = sd_diff / np.sqrt(n_pair)

    critical_t = t.ppf(
        0.975,
        df=n_pair - 1,
    )

    ci_low = mean_diff - critical_t * se
    ci_high = mean_diff + critical_t * se

    # --------------------------------------------------------
    # Paired Cohen's dz
    # --------------------------------------------------------

    cohens_dz = mean_diff / sd_diff

    # --------------------------------------------------------
    # Number of seeds where baseline beats IQGO
    # --------------------------------------------------------

    baseline_wins = np.sum(diff > 0)
    iqgo_wins = np.sum(diff < 0)
    ties = np.sum(diff == 0)

    results.append({
        "Model": name,

        "Mean": values.mean(),
        "SD": values.std(ddof=1),

        "Mean_difference": mean_diff,
        "CI_low": ci_low,
        "CI_high": ci_high,

        "t": t_stat,
        "p_t": p_t,

        "Wilcoxon_W": w_stat,
        "p_wilcoxon": p_w,

        "Shapiro_W": shapiro_stat,
        "p_shapiro": shapiro_p,

        "Cohens_dz": cohens_dz,

        "Baseline_wins": baseline_wins,
        "IQGO_wins": iqgo_wins,
        "Ties": ties,
    })


results = pd.DataFrame(results)


# ============================================================
# HOLM MULTIPLE-COMPARISON CORRECTION
#
# There are 5 pairwise comparisons:
#
# TabPFN vs IQGO
# CatBoost vs IQGO
# Poly SVC vs IQGO
# Linear SVC vs IQGO
# RBF SVC vs IQGO
# ============================================================

results["p_t_Holm"] = multipletests(
    results["p_t"],
    method="holm",
)[1]

results["p_wilcoxon_Holm"] = multipletests(
    results["p_wilcoxon"],
    method="holm",
)[1]


# ============================================================
# CONVERT RESULTS TO PERCENT / PERCENTAGE POINTS
# ============================================================

results["Mean_percent"] = results["Mean"] * 100
results["SD_percent"] = results["SD"] * 100

results["Difference_pp"] = (
    results["Mean_difference"] * 100
)

results["CI_low_pp"] = (
    results["CI_low"] * 100
)

results["CI_high_pp"] = (
    results["CI_high"] * 100
)


# ============================================================
# PAPER-FRIENDLY OUTPUT
# ============================================================

columns = [
    "Model",

    "Mean_percent",
    "SD_percent",

    "Difference_pp",
    "CI_low_pp",
    "CI_high_pp",

    "t",
    "p_t",
    "p_t_Holm",

    "Wilcoxon_W",
    "p_wilcoxon",
    "p_wilcoxon_Holm",

    "Cohens_dz",

    "Baseline_wins",
    "IQGO_wins",
    "Ties",

    "p_shapiro",
]


print("\n" + "=" * 90)
print("PAIRWISE RESULTS AGAINST IQGO")
print("=" * 90)

print(
    results[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# COMPACT PAPER TABLE
# ============================================================

paper_table = results[[
    "Model",
    "Mean_percent",
    "SD_percent",
    "Difference_pp",
    "CI_low_pp",
    "CI_high_pp",
    "p_t_Holm",
    "p_wilcoxon_Holm",
    "Cohens_dz",
    "Baseline_wins",
]].copy()

paper_table.columns = [
    "Algorithm",
    "Balanced Accuracy (%)",
    "SD (%)",
    "Delta BA vs IQGO (pp)",
    "95% CI Low (pp)",
    "95% CI High (pp)",
    "Paired t Holm p",
    "Wilcoxon Holm p",
    "Cohen's dz",
    "Baseline Wins",
]


print("\n" + "=" * 90)
print("COMPACT PAPER TABLE")
print("=" * 90)

print(
    paper_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# OPTIONAL SAVE
# ============================================================

paired_data.to_csv(
    "paired_balanced_accuracy_results.csv",
    index=False
)

results.to_csv(
    "iqgo_pairwise_statistics.csv",
    index=False
)

paper_table.to_csv(
    "iqgo_paper_statistics.csv",
    index=False
)
