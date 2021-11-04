"""randomness setting code

Fix all random seed

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""

import random
import torch
import numpy as np


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
