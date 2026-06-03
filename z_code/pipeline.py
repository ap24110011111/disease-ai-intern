import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
RANDOM_SEED = 42
df = pd.read_csv("../data/diabetes.csv")
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
for col in zero_columns:
    median_value = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median_value)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training class distribution:")
print(y_train.value_counts(normalize=True))
print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))