#!/usr/bin/env python3

# Author: Lucas V. Araujo <lucas.vieira.ar@disroot.org>

import numpy as np

from PSO import PSO

ninput = 3
noutput = 1
hlayers = (10,)

n = 10
X = np.random.normal(0.1, 1.0, size=(n, ninput))
Y = np.random.choice((0.0, 1.0), size=(n, noutput))

data = [X, Y]

pso = PSO(200, data, hlayers)
mlp = pso.start(200, 100, 0.001)

pred = np.round(mlp.classify(data))
accuracy = 100 * sum([1 if a == b else 0 for a, b in zip(pred, Y)])/ len(Y)
print(f"Accuracy for trainning data: {accuracy}%")
