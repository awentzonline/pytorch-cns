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

    def randomize(self, min_genes, max_genes, max_index, sigma_value):
        self.genes = []
        num_genes = np.random.randint(min_genes, max_genes)
        for i in range(0, num_genes):
            gene = Gene(np.random.randint(0, max_index), np.random.uniform(-sigma_value, sigma_value))
            self.genes.append(gene)

    def decode(self, target):
        target.fill(0.)
        original_shape = target.shape
        target = target.flatten()
        for gene in self.genes:
            target[gene.index] = gene.value
        target = target.reshape(original_shape)
        len_shape = len(original_shape)
        kwargs = dict(norm='ortho', overwrite_x=True)
        if len_shape == 1:
            out = idct(target, **kwargs)
        elif len_shape == 2:
            out = idct(idct(target.T, **kwargs).T, **kwargs)
        elif len_shape >= 3:
            shape = (np.prod(original_shape[:-1]), original_shape[-1])
            target = target.reshape(shape)
            out = idct(idct(target.T, **kwargs).T, **kwargs)
            out = out.reshape(original_shape)
        return out

    def mutate(self, p_index=0.1, p_value=0.8, sigma_value=5.0):
        for gene in self.genes:
            if np.random.uniform() < p_index:
                gene.index += np.random.randint(-1, 2)
                gene.index = np.clip(gene.index, 0, self.max_index)
            if np.random.uniform() < p_value:
                gene.value += np.random.normal(0., sigma_value)

    def split(self):
        p = len(self.genes) // 2
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


class ModelGenome:
    def __init__(self, model):
        self.genomes = []
        self._tmp_storages = []
        for parameter in model.parameters():
            num_weights = np.prod(parameter.size())
            genome = Genome(num_weights)
            self.genomes.append(genome)
            self._tmp_storages.append(np.zeros(parameter.size(), dtype=np.float32))

    def randomize(self, min_genes, max_genes, sigma_value):
        for genome in self.genomes:
            genome.randomize(min_genes, max_genes, genome.max_index // 2, sigma_value)

    def decode(self, target_model):
        for parameter, genome, _tmp in zip(target_model.parameters(), self.genomes, self._tmp_storages):
            _tmp = genome.decode(_tmp)
            parameter.data = torch.from_numpy(_tmp)

    def mutate(self, p_index=0.1, p_value=0.8, sigma_value=5.0):
        for genome in self.genomes:
            genome.mutate(p_index=p_index, p_value=p_value, sigma_value=sigma_value)

    def split(self):
        lefts = []
        rights = []
        for genome in self.genomes:
            left, right = genome.split()
            lefts.append(left)
            rights.append(right)
        return lefts, rights

    def child(self, a, b):
        for genome, genome_a, genome_b in zip(self.genomes, a.genomes, b.genomes):
            left_a, right_a = genome_a.split()
            left_b, right_b = genome_b.split()
            if np.random.uniform() > 0.5:
                left = left_a
            else:
                left = left_b
            if np.random.uniform() > 0.5:
                right = right_a
            else:
                right = right_b
            genome.genes = left + right


class Population:
    def __init__(self, model_factory, num_models, cuda):
        self.model_factory = model_factory
        self.model = model_factory()
        if cuda:
            self.model.cuda()
        self.num_models = num_models
        self.genomes = [ModelGenome(self.model) for _ in range(num_models)]
        for genome in self.genomes:
            genome.randomize(10, 20, 1.)
        self.best_genome = self.genomes[0]
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
        genome.decode(model)
        if self.cuda:
            model.cuda()

    def generation(self, x, y, f_loss):
        losses = self.evaluate(x, y, f_loss)
        ordered_losses = sorted([(loss, i) for i, loss in enumerate(losses)])
        num_best = len(ordered_losses) // 2
        ordered_genomes = [self.genomes[i] for _, i in ordered_losses]
        self.best_genome = ordered_genomes[0]
        for genome in ordered_genomes[num_best:]:
            a, b = random.sample(ordered_genomes[:num_best], 2)
            genome.child(a, b)
            genome.mutate()
        return losses

    def best_model(self):
        self.decode_genome(self.best_genome, self.model)
        return self.model
