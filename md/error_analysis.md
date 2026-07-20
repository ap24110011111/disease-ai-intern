# Error Analysis

## Objective
This analysis examines the misclassifications produced by the proposed disease detection framework, covering both the primary Pima Diabetes system and the CKD validation check, to identify patterns in the errors and inform the project's Limitations discussion.

## Methodology
Predictions from the trained models were compared against ground-truth labels on their respective held-out test sets. Misclassified samples were separated into False Positives (FP) and False Negatives (FN) for each dataset.

## Pima Diabetes — Adaptive + Proactive Detection System
- Evaluated across 5 rolling windows (Windows 3–7)
- Total false negatives: 27
- Per-window accuracy and F1 scores available in `results/week5/proactive_results.csv`. A confusion matrix for this system is not currently saved — the false negative counts above come from row-level predictions extracted directly from the model's output.

**Observations:** Most false negatives were not simple borderline cases — several missed patients had clearly elevated glucose readings. A number of missed cases also had missing (zero-value) Insulin readings, which are not currently imputed in preprocessing. Errors were not evenly distributed across time windows; more occurred in earlier windows than later ones.

**Clinical significance:** False negatives are more costly than false positives in a screening context — a missed diabetic patient risks delayed treatment, while a false positive typically leads only to an extra test.

## CKD Validation Check
- Total test samples: 80
- False Positives: 1
- False Negatives: 1
- Accuracy: 97.5%
- Confusion matrix: `[[49, 1], [1, 29]]`

**Observations:** Very few errors occurred overall. This is consistent with the CKD dataset's known properties — several of its features (e.g., specific gravity, hemoglobin, serum creatinine) are themselves close to clinical diagnostic markers for the condition, making the dataset easier to separate than Pima Diabetes. Near-perfect performance here reflects the nature of the dataset rather than a claim of general model superiority.

## Discussion
Across both datasets, errors are more informative on Pima, where the larger number of misclassifications makes patterns identifiable. The CKD errors are too few to draw a pattern from, but the low count is expected and explainable given the dataset's characteristics rather than indicating leakage or an unusually strong model.

## Conclusion
The framework's main limitations are visible in the Pima results — a tendency to miss clearly elevated cases, likely worsened by unimputed missing values and by higher error rates when trained on limited history. The CKD check performed well, consistent with the dataset being inherently easier to classify. Future work should focus on improving imputation and window training history for the Pima system, since that is the primary contribution of this project.