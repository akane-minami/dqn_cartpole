import random
from collections import namedtuple
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """状態遷移の保存"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """batchサイズ分だけランダムに取り出す"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)