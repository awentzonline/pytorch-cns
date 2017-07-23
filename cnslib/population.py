import operator
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class Gene:
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def clone(self):
        return Gene(self.index, self.value)


class Genome:
    '''Each gene is a tuple of the form (index, value) which represents
    a DCT coefficient.
    '''
    def __init__(self, max_index):
        self.max_index = max_index
        self.genes = []

    def randomize(self, min_genes, max_genes, sigma_value):
        self.genes = []
        num_genes = np.random.randint(min_genes, max_genes)
        for i in range(0, num_genes):
            gene = Gene(np.random.randint(0, self.max_index), np.random.normal(0, sigma_value))
            self.genes.append(gene)

    def decode(self, target):
        target.fill(0.)
        for gene in self.genes:
            target[gene.index] = gene.value
        idct(target, norm='ortho', overwrite_x=True)

    def mutate(self, p_index=0.1, p_value=0.8, sigma_value=1.0):
        for gene in self.genes:
            if np.random.uniform() < p_index:
                gene.index += np.random.randint(-1, 1)
                gene.index = np.clip(gene.index, 0, self.max_index)
            if np.random.uniform() < p_value:
                gene.value += np.random.normal(0., sigma_value)

    def split(self):
        p = np.random.randint(1, len(self.genes) - 1)
        left = [g.clone() for g in self.genes[:p]]
        right = [g.clone() for g in self.genes[p:]]
        return left, right

    def child(self, a, b):
        left_a, right_a = a.split()
        left_b, right_b = b.split()
        if np.random.uniform() > 0.5:
            left = left_a
        else:
            left = left_b
        if np.random.uniform() > 0.5:
            right = right_a
        else:
            right = right_b
        self.genes = left + right


class Population:
    def __init__(self, model_factory, num_models, cuda):
        self.model_factory = model_factory
        self.model = model_factory()
        if cuda:
            self.model.cuda()
        self.num_weights = np.sum(np.prod(p.size()) for p in self.model.parameters())
        self._tmp_weights = np.zeros(self.num_weights).astype(np.float32)
        print(self.num_weights)
        self.num_models = num_models
        self.genomes = [Genome(self.num_weights) for _ in range(num_models)]
        for genome in self.genomes:
            genome.randomize(15, 30, 10.)
        self.cuda = cuda

    def evaluate(self, x, y, f_loss):
        losses = []#np.zeros(self.num_models)
        for i, genome in enumerate(self.genomes):
            self.decode_genome(genome, self.model)
            y_pred = self.model(x)
            loss = f_loss(y_pred, y).data[0]
            #losses[i] = loss
            losses.append(loss)
        return losses

    def decode_genome(self, genome, model):
        genome.decode(self._tmp_weights)
        last_i = 0
        for parameter in model.parameters():
            p_size = parameter.size()
            this_size = np.prod(p_size)
            parameter.data = torch.from_numpy(
                self._tmp_weights[last_i:last_i + this_size].reshape(p_size)
            )
            last_i += this_size
        if self.cuda:
            model.cuda()

    def generation(self, x, y, f_loss):
        losses = self.evaluate(x, y, f_loss)
        ordered_losses = sorted([(loss, i) for i, loss in enumerate(losses)])
        num_best = len(ordered_losses) // 2
        ordered_genomes = [self.genomes[i] for _, i in ordered_losses]
        for genome in ordered_genomes[num_best:]:
            a, b = random.sample(ordered_genomes[:num_best], 2)
            genome.child(a, b)
            genome.mutate()
        return losses
