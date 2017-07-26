import operator
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from .genome import Genome, ModelGenome


class Population:
    def __init__(self, model_factory, num_models, cuda):
        self.model_factory = model_factory
        self.model = model_factory()
        if cuda:
            self.model.cuda()
        self.num_models = num_models
        self.genomes = [ModelGenome(self.model) for _ in range(num_models)]
        for genome in self.genomes:
            genome.randomize(10, 20, (-1., 1.))
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
