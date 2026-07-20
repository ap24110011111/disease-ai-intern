# Adaptive AI Models for Proactive Disease Detection

An 8-week research internship project exploring whether a machine learning model can adapt to new patient data over time, and whether it can be pushed toward flagging disease risk earlier rather than just classifying it after the fact.

Built and evaluated primarily on the **Pima Indians Diabetes Dataset**, with a secondary check on the **Chronic Kidney Disease (CKD) dataset** to see whether the approach generalizes.

**Mentor:** Dr. Ch. Anil Carie, SRM University–AP

**Team:**
- Venkata Ajay Odugu (AP24110011016)
- Mohanasritha Eerla (AP24110011024)
- Vijay Perla (AP24110011059)
- Neelima Bojanapu (AP24110011111)

## What this project actually does

Most disease-prediction models are trained once and never touched again — they don't learn from new patients as data comes in, and they're built to classify, not to catch risk early. This project asks two questions:

1. Can a model **adapt** — retraining itself as new batches of patient data arrive — without sacrificing performance?
2. Can a **"proactive"** layer on top of that catch more true cases earlier, even if it means trading away some precision?

The work moved through four stages: baseline classifiers (Logistic Regression, SVM, Random Forest, MLP) → an adaptive version of the MLP that updates on rolling windows of new data → a proactive detection layer tuned to flag risk earlier → a generalization check on a second, unrelated disease dataset (CKD).

## Datasets

| Dataset | Role | Size | Features | Target |
|---|---|---|---|---|
| Pima Indians Diabetes | Primary benchmark | 768 patients | 8 clinical features | `Outcome` (0 = non-diabetic, 1 = diabetic) |
| Chronic Kidney Disease (CKD) | Generalization check only | — | Clinical/lab features | CKD present/absent |

**Pima class balance:** 500 non-diabetic (65.1%), 268 diabetic (34.9%) — moderately imbalanced.

**Known data quality issue:** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contain physiologically-impossible zero values that likely represent missing data rather than true zeros. This is handled inconsistently across the codebase — see [Limitations](#limitations) below.

## Methodology by week

| Week | Focus | Key notebooks |
|---|---|---|
| 1 | Initial setup, exploratory data analysis | `eda.ipynb`, `hello.ipynb` |
| 2 | Data pipeline, linear/logistic regression baselines, feature correlation | `data_pipeline.ipynb`, `linear_regression_numpy.ipynb`, `logistic_regression_sklearn.ipynb`, `pima_features_heatmap.ipynb` |
| 3 | Tree-based baselines | `decision_tree.ipynb`, `random_forest.ipynb`, `svm.ipynb` |
| 4 | Neural network baselines and tuning | `mlp_comparison.ipynb`, `mlp_tensorflow.ipynb`, `mlp_tuning.ipynb` |
| 5 | Core contribution: adaptive retraining + proactive detection layer, ablation study | `adaptive_mlp.ipynb`, `proactive_detection.ipynb`, `ablation_study.ipynb` |
| 6 | Generalization check on CKD, multi-seed validation | `eda.ipynb`, `generalization.ipynb`, `multiseed_validation.ipynb`, `random_forest.ipynb` |

`z_code/` holds standalone `.py` versions of the core modeling scripts (baselines, pipeline, MLP variants) outside the weekly notebooks.

## Results

### Baseline models on Pima (week 2–4)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | 0.8230 |
| SVM | 0.7532 | 0.6600 | 0.6111 | 0.6346 | 0.7919 |
| Random Forest | 0.7597 | 0.6809 | 0.5926 | 0.6337 | 0.8118 |
| MLP | 0.7078 | 0.5957 | 0.5185 | 0.5545 | 0.7956 |

Random Forest has the strongest balance of accuracy, F1, and AUC among the baselines. Logistic Regression posts the highest AUC despite a lower F1, suggesting it separates the classes reasonably well but isn't the sharpest at the default decision threshold.

### Adaptive and proactive framework — ablation study (week 5)

| Configuration | Precision | Recall | F1 Score |
|---|---|---|---|
| Static MLP (baseline) | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7000 | 0.6100 | 0.6500 |
| Adaptive + Proactive | 0.7400 | 0.6600 | 0.6900 |

Every metric improves at each stage. That said, `md/limitations.md` flags a real caveat behind this trend: error analysis found that a disproportionate share of false negatives fell in the earliest evaluation windows, where the adaptive model had the least historical data to learn from. Part of the apparent gain may reflect the model simply having more data to work with later on, not just the adaptive/proactive mechanism itself — this hasn't been isolated with a formal significance test (e.g. McNemar's test), which is listed as future work.

### Generalization check — CKD (week 6)

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Pima Diabetes | Random Forest | 0.7597 | 0.6809 | 0.5926 | 0.6337 |
| CKD | Adaptive MLP | 0.975 | 0.967 | 0.967 | 0.967 |

The CKD result is much stronger, but this is largely a property of the dataset: several CKD features (specific gravity, hemoglobin, serum creatinine) are near-direct clinical diagnostic markers for the condition, making the two classes much easier to separate than in Pima. It's a useful sanity check that the pipeline generalizes mechanically to a second dataset, not evidence that the model is equally strong across disease types.

### Error analysis summary

- **Pima (adaptive + proactive system):** evaluated across 5 rolling windows; roughly two dozen false negatives overall. Many missed cases had clearly elevated glucose readings and unimputed zero-value Insulin readings — pointing to inconsistent preprocessing as a real contributor to errors, not just model limits.
- **CKD (validation check):** only 2 total errors (1 FP, 1 FN) out of 80 test samples — consistent with how separable this dataset is, not a claim of general superiority.

Full write-up: `md/error_analysis.md`.

## Limitations

Summarized from `md/limitations.md`:

1. **Dataset scope** — only two disease domains evaluated; CKD was a limited generalization check, not a full second benchmark.
2. **No formal significance testing** — the Static → Adaptive → Proactive gains are ablation metrics only, not confirmed with a statistical test.
3. **Adaptive gains may be confounded by history length** — more errors occurred in early evaluation windows, where less historical data was available.
4. **Inconsistent preprocessing** — zero-value imputation for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` isn't applied uniformly across notebooks/scripts.
5. **Batch-based, not streaming, adaptation** — updates happen on fixed sequential batches; proactive thresholds were also tuned on the same window used for reporting, which may overstate performance slightly.
6. **Future work** flagged: significance testing, unified preprocessing pipeline, independent threshold-selection set, more disease datasets, true online/streaming adaptation, and real clinical-data validation.

## Related work

`md/related_work.md` and `paper/` summarize three prior papers used to position this project:

- **`paper/paper1`** — AdaBoost-based breast cancer recurrence prediction (AUC 0.987) using routine blood tests + SHAP explainability; limited by small single-hospital sample and no genetic features.
- **`paper/paper2`** — Random Forest classification of Type 2 Diabetes subtypes on a Japanese cohort; limited by single-population data and no comparison against other algorithms.
- **`paper/paper3`** — notes on what to replicate (multi-model comparison, SHAP, thorough preprocessing) versus what this project tries to improve on (proactive rather than reactive detection, class imbalance handling).

The identified gap this project targets: adaptive learning strategies that keep improving with new patient data are relatively underexplored compared to one-shot classical/deep learning approaches.

## Repository structure

```
disease-ai-intern/
├── data/
│   ├── diabetes.csv       # Pima Indians Diabetes dataset
│   ├── ckd.csv            # Chronic Kidney Disease dataset
│   └── sample.csv         # small sample used for early exploration (z_code/explore.py)
├── figures/               # Key plots referenced in the paper/report
├── md/                    # Related work, contribution statement, EDA report, error analysis, limitations
├── notebooks/
│   ├── week 1/            # Setup, initial EDA
│   ├── week 2/            # Data pipeline, regression baselines, feature correlation
│   ├── week 3/            # Decision tree, random forest, SVM
│   ├── week 4/            # MLP baselines and tuning
│   ├── week 5/            # Adaptive MLP, proactive detection, ablation study
│   └── week 6/            # CKD generalization check, multi-seed validation
├── paper/
│   ├── paper1/            # Paper summary — breast cancer recurrence (AdaBoost)
│   ├── paper2/            # Gap analysis — diabetes subtype classification (Random Forest)
│   └── paper3/            # Replicate/improve notes
├── results/
│   ├── week 1/            # class_distribution.png
│   ├── week 2/            # results_log.csv — baseline metrics, heatmaps
│   ├── week 3/            # dt_vs_rf_comparison.csv, feature importance
│   ├── week 4/            # best_config.json, MLP training curves
│   ├── week5/             # ablation_results.csv, adaptive/proactive results, false_negatives.csv
│   ├── week6/             # CKD EDA/results, pima_vs_ckd_comparison.csv
├── z_code/                # Standalone .py scripts (baselines, pipeline, MLP variants)
└── .gitignore
```

## Running the code

The notebooks and `z_code/` scripts read data with relative paths such as `data/diabetes.csv` (some `z_code` scripts use `../data/diabetes.csv`), so run them from the repository root, or adjust the path if running from inside `z_code/`.

```bash
git clone https://github.com/ap24110011111/disease-ai-intern.git
cd disease-ai-intern

pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

# Example: run a baseline script directly
python z_code/logistic_regression_sklearn.py

# Or open any notebook
jupyter notebook "notebooks/week 5/adaptive_mlp.ipynb"
```

No `requirements.txt` is currently checked in — the dependencies above were collected from the imports used across `z_code/` and the notebooks (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`).

## Notes on this README

This file was rewritten after checking the full repository contents (data, notebooks, results CSVs, `md/` write-ups, and `paper/` summaries) — the reported metrics above were cross-checked against the corresponding `results/` CSVs (`results/week 2/results_log.csv`, `results/week5/ablation_results.csv`, `results/week6/ckd_rf_result.csv`, `results/week6/pima_vs_ckd_comparison.csv`) and match. One small inconsistency worth flagging: `md/error_analysis.md` reports 27 total false negatives for the Pima adaptive+proactive system, while `results/week5/false_negatives.csv` contains 28 data rows — worth a quick recount if that number is quoted elsewhere.