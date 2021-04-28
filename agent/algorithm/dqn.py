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
from agent.policy.e_greedy import E_greedy

class DQN():
    def __init__(self,steps_done,device):
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.steps_done=steps_done
        self.device=device
        self.policy=E_greedy(self.steps_done,self.device)

    def select_action(self,state,eps_start,eps_end,eps_decay,policy_net,n_actions):
        """sample = random.random()
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
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)"""
        action = self.policy.e_greedy(state,eps_start,eps_end,eps_decay,policy_net,n_actions)
        return action


    def optimize_model(self,batch_size,memory,policy_net,target_net,gamma,optimizer):
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        # バッチを転置します (詳細はhttps://stackoverflow.com/a/19343/3343043 を参照してください)
        # この処理では、Transitionsがバッチ配列になっているオブジェクトを、
        # バッチ配列がTransitionになっているオブジェクトに変換します
        #(state, action, state_next, reward)×BATCH_SIZE→(state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)
        batch = self.Transition(*zip(*transitions))

        # 最終状態「以外」を取り出すmaskを適用した後、バッチの要素を連結します。
        # (最終状態になるのは、シミュレーションが終了した後です)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))#!!!!!

        policy_net.eval()#!!
        # Q(s_t, a)を算出します。
        # 具体的には、作成したモデルでQ(s_t)を算出し、そこから各行動を示すカラムに対応する値を取得します
        # 値を取得するために使ったカラムは、バッチ内の各状態に対して、policy_netに従って取られた行動となっています
        state_action_values = policy_net(state_batch).gather(1, action_batch)#推論(近似関数)

        # 全ての遷移先の状態について、V(s_{t+1})を計算します
        # non_final_next_statesにおける行動の結果もたらされる利得の期待値は、「更新前の」target_netに基づいて計算されます
        # target_netのmax(1)[0]では、報酬の最大値を選択しています
        # この報酬は、最終状態の場合は0となりますが、それ以外の場合には、遷移先の状態における利得の期待値となります
        next_state_values = Variable(torch.zeros(batch_size, device=self.device).type(torch.FloatTensor))#!!
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # 遷移先の状態におけるQの期待値を計算します
        expected_state_action_values = (next_state_values * gamma) + reward_batch#教師信号作成!!!!!上2行はtargetに必要なものの作成
        policy_net.train()#!!
        # フーバー損失を計算します。
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))#損失計算(これは関数変えられる)2!!!!!

        # モデルを最適化します。ここは全部使う？
        optimizer.zero_grad()
        loss.backward()#逆誤差伝搬3
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()#update