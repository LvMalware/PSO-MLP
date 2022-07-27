import warnings
import numpy as np

warnings.filterwarnings('ignore')

activation_functions = {
        'sigmoid' : lambda x: 1.0 / (1.0 + np.exp(-x)),
        'relu' : lambda x: np.maximum(0, x) #,
}

class Particle:
    def __init__(self, data, layers, activation, c1, c2, w, alpha):
        self.g = activation_functions[activation]
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.loss = 1
        self.data = data
        self.dspace = sum([layers[i] * layers[i+1] for i in range(len(layers)-1)]) + sum(layers)
        self.layers = layers
        self.weights = np.random.normal(*alpha, size=(self.dspace,))
        self.velocity = np.random.normal(*alpha, size=(self.dspace,))
        self.local_best = np.array(self.weights)

    def score(self, x, y):
        pred = self.predict(x)
        return sum([a == b for a, b in zip(y, pred)]) / len(y)

    def predict(self, x):
        l = 0
        for i in range(len(self.layers) - 1):
            r, c = self.layers[i], self.layers[i+1]
            size = r * c
            w = self.weights[l : l + size].reshape((r, c))
            l = l + size
            b = self.weights[l : l + c].reshape((c,))
            l = l + c
            x = self.g(x.dot(w) + b)

        return np.argmax(x, axis=1)

    def get_loss(self, recalculate=False):
        if recalculate or self.loss is None:
            self.loss = 1.0 - self.score(*self.data)
        return self.loss

    def __gt__(self, other):
        return self.get_loss() > other.get_loss()

    def __lt__(self, other):
        return self.get_loss() < other.get_loss()

    def __eq__(self, other):
        return self.get_loss() == other.get_loss()

    def update(self, gbest):
        r1, r2 = np.random.rand(2)
        self.velocity = self.velocity * self.w + self.c1 * r1 * (self.weights - self.local_best) + self.c2 * r2 * (self.weights - gbest.weights)
        self.weights += self.velocity
        loss = self.loss
        if self.get_loss(True) < loss:
            self.local_best = np.array(self.weights)
        return self 
