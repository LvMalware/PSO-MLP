import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def func(ax_b):
    return 1.0 / (1.0 + np.exp(-ax_b))

class Particle:
    def __init__(self, data, hlayers, inertia=0.72984, c1=2.05, c2=2.05, alpha=(-1.0, 1.0)):
        self.c1 = c1
        self.c2 = c2
        self.data = data
        self.bias = [np.array([0] * n) for n in hlayers[1:]]
        self.inertia = inertia
        self.weights = np.array([ np.random.uniform(*alpha, size=(hlayers[i], hlayers[i + 1])) for i in range(len(hlayers) - 1) ], dtype=object)
        self.velocity = np.array([ np.random.uniform(*alpha, size=(hlayers[i], hlayers[i + 1])) for i in range(len(hlayers) - 1) ], dtype=object)
        self.score = None
        self.pbest = np.array(self.weights, dtype=object)
        self.sbest = None

    def fitness(self, keep=True):
        if keep and self.score is not None:
            return self.score

        pred = self.classify()
        targ = self.data[1]
        self.score = sum(sum((pred - targ) ** 2))

        if self.sbest is None or self.score < self.sbest:
            self.sbest = self.score
            self.pbest = np.array(self.weights, dtype=object)

        return self.score

    def update(self, gbest):
        if self == gbest:
            return self.fitness()

        for i in range(len(self.weights)):
            self.weights[i] += self.velocity[i]

        self.velocity = [ v * self.inertia for v in self.velocity ] + self.c1 * np.random.rand() * (self.pbest - self.weights) + self.c2 * np.random.rand() * (gbest.weights - self.weights)
        return self.fitness(False)
		

    def classify(self, data=None):
        x, y = data if data is not None else self.data
        for bias, layer in zip(self.bias, self.weights):
            x = func(x.dot(layer) + bias)

        return x

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    def __str__(self):
        return f"Particle(fitness={self.fitness()})"

    def __gt__(self, other):
        return self.fitness() > other.fitness()

    def __eq__(self, other):
        return self.fitness() == other.fitness()

