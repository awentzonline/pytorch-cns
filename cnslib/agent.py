import random

import numpy as np
import torch
from torch.autograd import Variable

from cnslib.genome import ModelGenome


class Agent:
    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        self.cuda = cuda
        self.genome = ModelGenome(model)
        self.genome_a = ModelGenome(model)
        self.genome_b = ModelGenome(model)

    def __call__(self, *args):
        return self.model(*args)

    def policy(self, state, *args):
        state = state[None, ...].astype(np.float32)
        state = torch.from_numpy(state)
        inputv = Variable(state)
        ps = self.model(inputv, *args).data.numpy()
        ps = np.argmax(ps)
        return ps

    def policy_rnn(self, state, *args):
        state = state[None, ...].astype(np.float32)
        state = torch.from_numpy(state)
        inputv = Variable(state)
        result = self.model(inputv, *args)
        action = np.argmax(result[0].data.numpy())
        return action, result[1]

    def randomize(self, gene_weight_ratio, freq_weight_ratio, init_value_range):
        self.genome.randomize(gene_weight_ratio, freq_weight_ratio, init_value_range)

    def crossover(self, best_genomes):
        parents = random.sample(best_genomes, 2)
        best_genomes = [genome for genome, _ in best_genomes]
        self.genome_a.deserialize_genomes(best_genomes[0])
        self.genome_b.deserialize_genomes(best_genomes[1])
        self.genome.child(self.genome_a, self.genome_b)

    def mutate(self, index_sigma=1., value_sigma=1.):
        self.genome.mutate(index_sigma=index_sigma, value_sigma=value_sigma)

    def update_model(self):
        self.genome.decode(self.model)
        if self.cuda:
            self.model.cuda()

    def load_genome(self, genome):
        self.genome.deserialize_genomes(genome)

    def summary(self):
        print(self.model)
        print(self.genome)
