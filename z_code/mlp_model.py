import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
df = pd.read_csv("../data/diabetes.csv")

print("Dataset Shape:", df.shape)
df.head()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def evaluate_model(name, model, X_train, X_test):
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return [
        name,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(roc_auc, 4)
    ]
    results = []
results.append(
    evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        X_train_scaled,
        X_test_scaled
    )
)

results.append(
    evaluate_model(
        "SVM",
        SVC(probability=True),
        X_train_scaled,
        X_test_scaled
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        X_train,
        X_test
    )
)

results.append(
    evaluate_model(
        "MLP",
        MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=42
        ),
        X_train_scaled,
        X_test_scaled
    )
)
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

results_df
results_df.to_csv(
    "../results/results_log.csv",
    index=False
)

print("results_log.csv saved successfully!")
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("Best Model based on F1 Score:")
print(best_model)
best_model_name = best_model["Model"]
best_f1 = best_model["F1 Score"]

print(best_model_name, best_f1)