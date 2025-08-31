import numpy as np

# Data
X = np.array([
    [1, 1],  # espresso (0)
    [5, 0],  # latte (1)
    [3, 1],  # latte (1)
    [2, 1]   # espresso (0)
])
y = np.array([[0],[1],[1],[0]])

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 3)   # 2 -> 3
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1)   # 3 -> 1
b2 = np.zeros((1, 1))

# Training loop
lr = 0.1  # learning rate
for epoch in range(5000):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # Loss (binary cross-entropy)
    loss = -np.mean(y*np.log(y_hat + 1e-9) + (1-y)*np.log(1-y_hat + 1e-9))

    # Backprop
    dz2 = y_hat - y
    dW2 = np.dot(a1.T, dz2) / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_deriv(z1)
    dW1 = np.dot(X.T, dz1) / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Update weights
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # Print every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Predictions
print("\nFinal predictions:")
print(y_hat.round())

def simple_nn(x):
    a1 = sigmoid(x@W1+b1)
    pred = sigmoid(a1@W2+b2)

    return pred.round()