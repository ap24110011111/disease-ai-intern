import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
# Load Dataset

df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Grid Search
param_grid = {
    "max_depth": list(range(3, 11))
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Depth:", grid.best_params_)
# Predictions

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
# Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nMetrics")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", auc)
# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix_dt.png")
plt.close()
print("\nConfusion matrix saved.")
# Log Results

result = pd.DataFrame({
    "Model": ["DT"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [auc]
})
csv_file = "results/results_log.csv"
try:
    old = pd.read_csv(csv_file)
    old = old[old["Model"] != "DT"]
    updated = pd.concat(
        [old, result],
        ignore_index=True
    )
    updated.to_csv(csv_file, index=False)
except FileNotFoundError:
    result.to_csv(csv_file, index=False)
print("Results logged to results_log.csv")