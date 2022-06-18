from Particle import Particle

class PSO:
    def __init__(self, nparticles, data, hlayers=(32,), inertia=0.8, c1=2.05, c2=2.05, alpha=(-1.0,1.0)):
        self.data = data
        self.size = nparticles
        layers = (data[0].shape[1], *hlayers, data[1].shape[1])
        self.swarm = [ Particle(data, layers, inertia, c1, c2, alpha) for _ in range(nparticles) ]

    def start(self, iterations, early_stop=None, min_err=None):
        print("Starting PSO process...")
        gbest = min(self.swarm)
        nbest = gbest
        count = 0
        for i in range(iterations):
            print(f"Iteration #{i + 1}")
            for p in self.swarm:
                if p.update(gbest) < nbest.fitness():
                    nbest = p

            if gbest == nbest:
                count += 1
            else:
                count = 0
            gbest = nbest
            print(f"Best: {gbest}")
            if min_err is not None and gbest.fitness() <= min_err:
                print(f"Early stop at iteration {i} because desired error was reached.")
                break
            if early_stop is not None and count >= early_stop:
                print(f"Early stop at iteration {i} because the fitness stopped improving.")
                break

        return gbest
