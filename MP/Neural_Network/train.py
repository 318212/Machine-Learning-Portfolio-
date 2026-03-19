import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model import NeuralNetwork

data = load_iris()
X = data.data
y = data.target

X = X[y != 2]
y = y[y != 2]
y = y.reshape(-1, 1)

#classic train, test, split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = NeuralNetwork(input_size=4, hidden_size=10, output_size=1, lr=0.1)

model.fit(X_train, y_train, epochs=1000)

preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)

print("Accuracy:", accuracy)