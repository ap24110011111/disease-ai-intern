import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load Dataset
df = pd.read_csv("data/diabetes.csv")

print("Dataset Shape:", df.shape)

# Prepare Data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# MLP Tuning
batch_sizes = [32, 64, 128]

best_accuracy = 0
best_config = {}

for batch_size in batch_sizes:

    print("\n" + "=" * 50)
    print(f"Training with Batch Size = {batch_size}")
    print("=" * 50)

    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=1
    )

    predictions = (model.predict(X_test) > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy

        best_config = {
            "batch_size": batch_size,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "epochs": 30,
            "accuracy": float(accuracy)
        }
# Display Best Configuration
print("\nBest Configuration")
print(best_config)

# Save Best Configuration
with open("results/best_config.json", "w") as f:
    json.dump(best_config, f, indent=4)

print("\nbest_config.json saved successfully!")