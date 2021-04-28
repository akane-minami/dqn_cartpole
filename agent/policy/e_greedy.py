import math
import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

class E_greedy():
    def __init__(self,steps_done,device):
        self.steps_done=steps_done
        self.device=device

    def e_greedy(self,state,eps_start,eps_end,eps_decay,policy_net,n_actions):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            policy_net.eval()
            with torch.no_grad():
            # t.max(1) は、テンソルの行方向の最大値に対応する列を返します。
            # max(1)[1]は、最大値に対応する列のインデックスを示しています。
            # つまり、このコードは報酬の期待値が最大となる行動を選択していることを意味します。
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)