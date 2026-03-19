import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from model import NeuralNetwork
from utility import plot_loss

data = load_iris()
X = data.data
y = data.target

X = X[y != 2]
y = y[y != 2]
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = NeuralNetwork(4, 10, 1, lr=0.1)

model.fit(X_train, y_train, epochs=1000)

preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)

print("Accuracy:", accuracy)

plot_loss(model.losses)