import numpy as np
import torch
from scipy.fftpack import dct, idct


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
