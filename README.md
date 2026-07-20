```markdown
# Adaptive AI Models for Proactive Disease Detection

An 8-week research internship project exploring whether a machine learning model can adapt to new patient data over time, and whether it can be pushed toward flagging disease risk earlier rather than just classifying it after the fact.

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

Baseline classifiers (Logistic Regression, SVM, Random Forest, MLP) were built first, then an adaptive version of the MLP, then a proactive detection layer on top of that.

## Dataset

**Pima Indians Diabetes**: 768 patients, 8 clinical features, binary diabetes outcome.

## Results

### Baseline models (week 2)

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

Every metric improves at each stage — Adaptive MLP beats the static baseline, and Adaptive + Proactive beats both, gaining on precision, recall, and F1 simultaneously. 

## Repository structure

```
disease-ai-intern/
├── data/            # [describe: raw]
├── figures/          # Plots and visualizations
├── md/               # Related work, contribution statement, error analysis, limitations
├── notebooks/
│   ├── week 1/
│   ├── week 2/        # Baseline model training/evaluation
│   ├── week 3/
│   ├── week 4/
│   ├── week 5/        # Adaptive/proactive ablation study
│   └── week 6/
├── paper/
│   ├── paper1/
│   ├── paper2/
│   └── paper3/
├── results/
│   ├── week 1/
│   ├── week 2/        # results_log.csv — baseline metrics
│   ├── week 3/
│   ├── week 4/
│   ├── week5/          # ablation_results.csv — adaptive/proactive comparison
│   └── week6/
├── z_code/            
└── .gitignore
```


## Running this yourself

```bash
git clone https://github.com/ap24110011111/disease-ai-intern.git
cd disease-ai-intern
pip install -r requirements.txt   
jupyter notebook
```

