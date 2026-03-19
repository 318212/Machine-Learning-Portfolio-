import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr

        # weights for neural
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    # activation
    # relu x2, sigmoid
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, y, y_hat):
        eps = 1e-9
        return -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

    def backward(self, X, y):
        m = X.shape[0]

        dZ2 = self.A2 - y
        dW2 = (1/m) * (self.A1.T @ dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)

        dW1 = (1/m) * (X.T @ dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=1000, verbose=True):
        for i in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)

            self.backward(X, y)

            if verbose and i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int)