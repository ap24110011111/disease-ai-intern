import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
print("X =", X)
print("y =", y)
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Training Data")
plt.show()
w = 0
b = 0
learning_rate = 0.08
epochs = 1000
m = len(X)
cost_history = []
for epoch in range(epochs):
    y_pred = w * X + b
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    cost_history.append(cost)
    dw = (1 / m) * np.sum((y_pred - y) * X)
    db = (1 / m) * np.sum(y_pred - y)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    print("Weight (w):", w)
print("Bias (b):", b)
plt.figure(figsize=(8,5))
plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost vs Epochs")
plt.grid(True)
plt.savefig("results/cost_vs_epochs.png")
plt.show()
plt.scatter(X, y, label="Actual Data")
plt.plot(X, w * X + b, label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression from Scratch")
plt.legend()
plt.show()
test_x = 6
prediction = w * test_x + b
print("Prediction for x = 6:", prediction)