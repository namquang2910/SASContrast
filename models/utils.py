"""
Utility functions to be used by neural networks.
"""
from models.net.CNNEncoder import CNNEncoder
from models.net.sas_encoder import SASEncoder, StemEncoder


def get_base_encoder(name, args):
    if name == 'cnn':
        return CNNEncoder(**args)
    elif name == 'moe':
        return SASEncoder(**args)
    elif name == "sas_stem":
        return StemEncoder(**args)

