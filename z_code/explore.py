import pandas as pd

df = pd.read_csv("data/sample.csv")

print("Shape of dataset:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())