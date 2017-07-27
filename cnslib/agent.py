import random

import numpy as np
import torch
from torch.autograd import Variable

from cnslib.genome import ModelGenome


class Agent:
    def __init__(self, model):
        self.model = model
        self.genome = ModelGenome(model)
        self.genome_a = ModelGenome(model)
        self.genome_b = ModelGenome(model)

    def __call__(self, x):
        return self.model(x)

    def policy(self, state):
        state = state[None, ...].astype(np.float32)
        state = torch.from_numpy(state)
        inputv = Variable(state)
        ps = self.model(inputv).data.numpy()
        ps = np.argmax(ps)
        return ps

    def randomize(self, min_genes, max_genes, init_value_range):
        self.genome.randomize(min_genes, max_genes, init_value_range)

    def crossover(self, best_genomes):
        parents = random.sample(best_genomes, 2)
        best_genomes = [genome for genome, _ in best_genomes]
        self.genome_a.deserialize_genomes(best_genomes[0])
        self.genome_b.deserialize_genomes(best_genomes[1])
        self.genome.child(self.genome_a, self.genome_b)

    def mutate(self, value_range=(-1., 1.)):
        self.genome.mutate(value_range=value_range)

    def update_model(self):
        self.genome.decode(self.model)

    def load_genome(self, genome):
        self.genome.deserialize_genomes(genome)
