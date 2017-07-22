import operator

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class Population:
    def __init__(self, model_factory, num_models, cuda):
        self.models = [model_factory() for _ in range(num_models)]
        self.num_models = num_models
        self.parent_code = self.encode_model(self.models[0])  # initial code
        self.cuda = cuda

    def evaluate(self, x, y, f_loss):
        losses = np.zeros(self.num_models)
        for i, model in enumerate(self.models):
            y_pred = model(x)
            loss = -f_loss(y_pred, y).data[0]
            losses[i] = loss
        return losses

    def encode_model(self, model):
        code = []
        for param in model.parameters():
            subcode = dct(param.data.view(-1).numpy(), norm='ortho')
            code.append(subcode)
        return code

    def decode_model(self, target_model, code):
        for subcode, param in zip(code, target_model.parameters()):
            decoded = idct(subcode, norm='ortho')
            decoded = decoded.reshape(param.size())
            param.data = torch.from_numpy(decoded)
        if self.cuda:
            target_model.cuda()
        return target_model

    def perturb_children(self, perturbations):
        for model_i, model in enumerate(self.models):
            target_model = self.models[model_i]
            for subcode, perturbation, param in zip(self.parent_code, perturbations, target_model.parameters()):
                decoded = idct(subcode + perturbation[model_i], norm='ortho')
                decoded = decoded.reshape(param.size())
                param.data = torch.from_numpy(decoded)
        if self.cuda:
            self.cuda_all()

    def generation(self, x, y, f_loss, learning_rate=0.001, sigma=0.1):
        perturbations = self.make_perturbations()
        self.perturb_children(perturbations)
        losses = self.evaluate(x, y, f_loss)
        combined = self.combine_perturbations(perturbations, losses)
        self.perturb_code(self.parent_code, combined, learning_rate, sigma)
        return losses

    def make_perturbations(self):
        ps = []
        for subcode in self.parent_code:
            p = np.random.randn(self.num_models, subcode.shape[0]).astype(np.float32)
            ps.append(p)
        return ps

    def combine_perturbations(self, perturbations, weights):
        out = []
        weights = (weights - np.mean(weights)) / np.std(weights)
        for p in perturbations:
            combined = np.dot(p.T, weights)
            out.append(combined)
        return out

    def perturb_code(self, code, perturbations, learning_rate, sigma):
        k = learning_rate / (self.num_models * sigma)
        for subcode, perturbation in zip(code, perturbations):
            dsc = k * perturbation
            subcode += dsc

    def parent_model(self):
        model = self.models[0]  # hacktown
        self.decode_model(model, self.parent_code)
        return model

    def cuda_all(self):
        for model in self.models:
            model.cuda()
