import numpy as np
from PSOMLP import PSOMLP

n = 200
i = 5
# generate random dataset
x = np.random.normal(-1, 1, size=(n, i))
# the class is defined by a real function applyed to x
y = np.array([1 if sum(a) >= 1 else 0 for a in x])

pso = PSOMLP(hlayers=(10,))
mlp = pso.fit(x, y, iterations=50)
print("Accuracy for trainning data:", 100 * mlp.score(x, y))
