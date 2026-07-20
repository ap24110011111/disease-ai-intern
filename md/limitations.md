# Limitations

Although the proposed adaptive disease detection framework demonstrated promising performance, several limitations should be considered.

## 1. Dataset Scope
The proposed framework was evaluated using the Pima Indians Diabetes dataset as the primary benchmark and the Chronic Kidney Disease (CKD) dataset for cross-dataset validation. Although these datasets are widely used in medical machine learning research, they represent only two disease domains, and CKD was evaluated only as a limited generalization check rather than a full second benchmark. Future work should evaluate the framework on additional public healthcare datasets to assess its generalizability across different diseases, patient demographics, and clinical settings.

---

## 2. Statistical Significance
A formal statistical significance test (e.g., McNemar's test) comparing the Adaptive MLP against the Random Forest baseline has not yet been conducted. The reported improvements from Static → Adaptive → Adaptive + Proactive are based on ablation metrics only. Future work should include statistical significance testing to verify whether the observed performance gains are reliable rather than due to random variation.

---

## 3. Adaptive Learning Performance
The ablation study compared Static MLP, Adaptive MLP, and Adaptive + Proactive configurations:

| Configuration | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| Static MLP | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7000 | 0.6100 | 0.6500 |
| Adaptive + Proactive | 0.7400 | 0.6600 | 0.6900 |

The Adaptive MLP achieved a modest improvement over the static baseline, while the Adaptive + Proactive framework produced the highest overall performance, particularly in recall. However, error analysis indicated that a significant proportion of false negatives occurred during the earliest evaluation window, where limited historical data were available for adaptation. This suggests that part of the observed improvement depends on the amount of accumulated historical information and should be further validated under longer-term deployment scenarios.

---

## 4. Preprocessing Consistency
Zero-value imputation for invalid placeholder values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI was not consistently applied across all notebooks in the repository. Error analysis showed that many false-negative predictions contained unimputed **Insulin = 0** values, indicating that inconsistent preprocessing may introduce unnecessary prediction errors. Applying a unified preprocessing pipeline across all experiments would improve reproducibility and model reliability.

---

## 5. Batch-Based Evaluation
The adaptive framework currently performs model updates using fixed sequential batches rather than continuous real-time data streams. Furthermore, proactive detection thresholds were selected by evaluating multiple threshold values on the same evaluation window used for reporting results, which may slightly overestimate performance. Future work should employ a dedicated validation set for threshold selection and extend the framework toward true online or streaming adaptive learning.

---

## 6. Future Work
Several directions can further improve the proposed framework:

- Perform formal statistical significance testing (e.g., McNemar's test) to validate performance improvements.
- Apply a standardized preprocessing pipeline with consistent missing-value and zero-value imputation across all datasets.
- Select proactive detection thresholds using an independent validation dataset to avoid evaluation bias.
- Evaluate the framework on additional publicly available medical datasets covering multiple diseases and patient populations.
- Extend the adaptive learning mechanism to support real-time online or streaming data updates.
- Validate the proposed framework using real-world clinical or hospital datasets in collaboration with healthcare professionals to assess its practical applicability and robustness in clinical environments.