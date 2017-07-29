import base64
import itertools
import operator
import pickle
import random
import zlib

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

    def __str__(self):
        return '<Gene i={} v={}>'.format(self.index, self.value)


class Genome:
    '''Each gene is a tuple of the form (index, value) which represents
    a DCT coefficient.
    '''
    def __init__(self, num_weights, genes=None):
        self.num_weights = num_weights
        self.genes = genes or []

    def randomize(self, gene_weight_ratio, max_index, value_range):
        self.genes = []
        num_genes = max(2, int(gene_weight_ratio * self.num_weights))
        print(max_index, value_range)
        for i in range(num_genes):
            gene = Gene(np.random.randint(0, max_index), np.random.uniform(*value_range))
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

    def mutate(self, p_index=0.1, p_value=0.8, index_sigma=1., value_sigma=1.):
        for gene in self.genes:
            if np.random.uniform() < p_index:
                gene.index += int(np.random.normal(0., index_sigma))
                gene.index = np.clip(gene.index, 0, self.num_weights - 1)
            if np.random.uniform() < p_value:
                gene.value += np.random.normal(0., value_sigma)

    def split(self, random=True):
        if random:
            p = np.random.randint(len(self.genes))
        else:
            p = len(self.genes) // 2
        left = [g.clone() for g in self.genes[:p]]
        right = [g.clone() for g in self.genes[p:]]
        return left, right

    def cut(self, p_cut):
        num_genes = len(self.genes)
        if np.random.uniform() < p_cut * num_genes:
            pivot = np.random.randint(num_genes)
            left = [g.clone() for g in self.genes[:pivot]]
            right = [g.clone() for g in self.genes[pivot:]]
            return left, right
        return ([g.clone() for g in self.genes],)

    def splice(self, other, p_cut, p_splice):
        this_cuts = self.cut(p_cut)
        other_cuts = other.cut(p_cut)
        results = []
        current = []
        for chromosome in itertools.chain(this_cuts, other_cuts):
            if not current or np.random.uniform() < p_splice:
                current += chromosome
            elif current:
                results.append(current)
                current = []
        if current:
            results.append(current)
        return results

    def child(self, a, b):
        left_a, right_a = a.split()
        left_b, right_b = b.split()
        if np.random.uniform() < 0.5:
            left = left_a
            right = right_b
        else:
            left = left_b
            right = right_a
        self.genes = left + right

    def __str__(self):
        return '<Genome w={} {}>'.format(
            self.num_weights, ', '.join(str(g) for g in self.genes))


class ModelGenome:
    def __init__(self, model):
        self.genomes = []
        self._tmp_storages = []
        for parameter in model.parameters():
            num_weights = np.prod(parameter.size())
            genome = Genome(num_weights)
            self.genomes.append(genome)
            self._tmp_storages.append(np.zeros(parameter.size(), dtype=np.float32))

    def randomize(self, gene_weight_ratio, freq_weight_ratio, value_range):
        for genome in self.genomes:
            genome.randomize(
                gene_weight_ratio, max(1, int(genome.num_weights * freq_weight_ratio)), value_range)

    def decode(self, target_model):
        for parameter, genome, _tmp in zip(target_model.parameters(), self.genomes, self._tmp_storages):
            _tmp = genome.decode(_tmp)
            parameter.data = torch.from_numpy(_tmp)

    def mutate(self, p_index=0.1, p_value=0.8, index_sigma=1., value_sigma=1.):
        for genome in self.genomes:
            genome.mutate(p_index=p_index, p_value=p_value, value_sigma=value_sigma)

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
                right = right_b
            else:
                left = left_b
                right = right_a
            genome.genes = left + right

    def serialize_genomes(self):
        d = pickle.dumps(self.genomes)
        d = zlib.compress(d)
        d = base64.b64encode(d)
        return d

    def deserialize_genomes(self, data):
        g = base64.b64decode(data)
        g = zlib.decompress(g)
        self.genomes = pickle.loads(g)

    def __str__(self):
        return '<ModelGenome {}>'.format(', '.join(str(g) for g in self.genomes))
