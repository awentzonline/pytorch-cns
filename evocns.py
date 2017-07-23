import operator

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class ModelCodec:
    def encode(self, model):
        code = []
        for param in model.parameters():
            subcode = param.data.view(-1).numpy()
            subcode = self.encode_subcode(subcode)
            code.append(subcode)
        return code

    def decode(self, codes, model):
        for subcode, param in zip(codes, model.parameters()):
            decoded = self.decode_subcode(subcode)
            decoded = decoded.reshape(param.size())
            param.data = torch.from_numpy(decoded)
        return model

    def encode_subcode(self, decoded):
        raise 'Override encode_subcode'

    def decode_subcode(self, encoded):
        raise 'Override decode_subcode'

    def decode_perturbed(self, codes, model, perturbations, model_i):
        # TODO: rethink this to get rid of model_i param
        for subcode, perturbation, param in zip(codes, perturbations, model.parameters()):
            decoded = self.decode_subcode(subcode + perturbation[model_i])
            decoded = decoded.reshape(param.size())
            param.data = torch.from_numpy(decoded)
        return model


class IdentityCodec(ModelCodec):
    def encode_subcode(self, decoded):
        return decoded

    def decode_subcode(self, encoded):
        return encoded


class DCTCodec(ModelCodec):
    def encode_subcode(self, decoded):
        return dct(decoded, norm='ortho')

    def decode_subcode(self, encoded):
        return idct(encoded, norm='ortho')


class Population:
    def __init__(self, model_factory, num_models, codec, cuda):
        self.models = [model_factory() for _ in range(num_models)]
        self.num_models = num_models
        self.codec = codec
        self.parent_code = self.codec.encode(self.models[0])  # initial code
        self.cuda = cuda

    def evaluate(self, x, y, f_loss):
        losses = np.zeros(self.num_models)
        for i, model in enumerate(self.models):
            y_pred = model(x)
            loss = -f_loss(y_pred, y).data[0]
            losses[i] = loss
        return losses

    def perturb_children(self, perturbations):
        for model_i, model in enumerate(self.models):
            target_model = self.models[model_i]
            self.codec.decode_perturbed(self.parent_code, target_model, perturbations, model_i)
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
        self.codec.decode(self.parent_code, model)
        if self.cuda:
            model.cuda()
        return model

    def cuda_all(self):
        for model in self.models:
            model.cuda()
