import numpy as np
from tqdm import tqdm
from Particle import Particle

class PSOMLP:
    def __init__(self, hlayers=(100,), ativation='sigmoid', c1=2.05, c2=2.05, w=0.72984, alpha=(-1,1)):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.hlayers = hlayers
        self.ativation = ativation

    def fit(self, x, y, iterations=100, nparticles=100, early_stop=None, min_err=-np.inf, print_step=True):
        n_in = x.shape[1] if len(x.shape) > 1 else 1
        n_out = y.shape[1] if len(y.shape) > 1 else 1
        layers = [n_in, *self.hlayers, n_out]
        self.swarm = [ Particle([x, y], layers, self.ativation, self.c1, self.c2, self.w, self.alpha) for _ in range(nparticles)]
        self.gbest = min(self.swarm)
        loops = tqdm(range(iterations), desc="Trainning MLP") if print_step else range(iterations)
        nbest = self.gbest
        count = 0
        for _ in loops:
            for p in self.swarm:
                if p.update(self.gbest) < nbest:
                    nbest = p

            if nbest == self.gbest:
                count += 1
            else:
                count = 0
                self.gbest = nbest
            if early_stop is not None and count >= early_stop:
                break

        return self.gbest

