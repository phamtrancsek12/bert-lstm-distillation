"""
Utils functions
"""
import torch
import random
import numpy as np
from config import SEED


def set_seed():
    """ Set random seed to all """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)