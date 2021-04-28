import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from agent.network.dqn_network import DQN_Network
from agent.algorithm.dqn import DQN
from agent.replay_memory import ReplayMemory

class Simulator(object):
    def __init__(self,env,num_episodes,num_simulations,batch_size,gamma,eps_start,eps_end,eps_decay,target_update,network_alpha,replay_memory):
        self.env=env
        self.num_episodes=num_episodes
        self.num_simulations = num_simulations
        self.batch_size=batch_size
        self.gamma=gamma
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay=eps_decay
        self.target_update=target_update
        #self.network_alpha=network_alpha
        self.alpha_target=1 - network_alpha
        self.alpha_policy=network_alpha
        self.replay_memory=replay_memory
        self.episode_durations = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        self.screen_height=0.0
        self.screen_width=0.0

    def get_cart_location(self,screen_width):
        """画面上のカートの位置を取得"""
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0) # カートの真ん中

    def get_screen(self):
    # gymｄ使用されるスクリーンのサイズは400x600x3ですが、
    # 実際のスクリーンがそれよりも大きい場合もあります。（ 例えば800x1200x3など）
    # ここでは、スクリーンの次元の順序を、Pytorch標準の次元の順序(CHW/色縦横)に並び替え
        resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC),T.ToTensor()])
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        #self.env.close()
    # カートはスクリーンの下半分にあるので、上下をトリミングします
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    #左右の処理
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,cart_location + view_width // 2)
    # 縁をトリミングし、カートの中央に四角い画像がくるようにします
        screen = screen[:, :, slice_range]
    # floatに変換し、再拡大して、Pytorchテンソルに変換します
    # (この操作では変数のコピーは不要です。)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
    # リサイズをして、バッチ数の次元を追加します。(BCHW/サンプル(バッチ)数色縦横)という形になります。
        return resize(screen).unsqueeze(0).to(self.device)
    
    def screen_processing(self):
        #plt.ion()
        #環境初期化
        self.env.reset()
        #画像の表示
        #plt.figure()
        #画像の前処理の準備
        #plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
        #plt.title('Example extracted screen')
        #plt.show()

        # カート・ポールの画面サイズを取得し、gymから返される形状に基づき正しくレイヤーを初期化できるようにします
        # ここで取得している画面のサイズは、3x40x90程度となっています。
        # この3x40x90というサイズは、get_screen() 内でトリミングや縮小が施されたたrender buffer（描画されるカート・ポールの画面）のサイズです
        init_screen = self.get_screen()
        _, _, self.screen_height, self.screen_width = init_screen.shape



    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # 100エピソードの平均を取り、プロットします。
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # プロットを更新するために一時停止します
        """if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())"""
    def plot(self,results,label,dir_path):
        for i in range(len(label)):
            y=np.mean(results[i],axis=0)#平均
            num=5#移動平均の個数
            b=np.ones(num)/num
            y2=np.convolve(y, b, mode='same')#移動平均
            #plt.plot(np.linspace(1, self.num_episodes, num=self.num_episodes),y , label=label) #平均して出力
            plt.plot(np.linspace(1, self.num_episodes, num=self.num_episodes),y2 , label=label[i]+"_move_average") #平均して出力

        plt.legend(loc="best")
        plt.xlabel("episodes")
        plt.ylabel("rewards")
        plt.savefig(dir_path + "rewards", bbox_inches='tight', pad_inches=0)
        plt.show()

    def run(self,network,algo,soft_update_flag):
        label = algo
        rewards = np.zeros((len(algo),self.num_simulations,self.num_episodes),dtype = int)
        for i in range(len(algo)):
            self.screen_processing()
            rewards[i] = self.run_sim(network[i],soft_update_flag[i])
            print('Complete')
        #self.env.render()
        #self.env.close()
        time_now = datetime.datetime.now()
        dir_path = 'results/{0:%Y%m%d%H%M}/'.format(time_now)
        os.makedirs(dir_path, exist_ok=True)
        #label = algo[0].__name__
        self.plot(rewards,label,dir_path)
    
    def run_sim(self,net,soft_update):
        # Open AI Gymのアクションスペースから、選択できる行動の数を取得します
        # （日本語訳注：ここでは、「右に動かす」と「左に動かす」の２つの行動が選択できるので、n_actionsの値は2となります）
        n_actions = self.env.action_space.n
        sum_rewards = np.zeros((self.num_simulations,self.num_episodes),dtype = int)
        for sim in range(self.num_simulations):
            #方策用のDQNとtargetのDQNを生成
            policy_net = net(self.screen_height, self.screen_width, n_actions).to(self.device)
            target_net = net(self.screen_height, self.screen_width, n_actions).to(self.device)
            #policy_net = algo(n_actions).to(self.device)
            #target_net = algo(n_actions).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())#モデルの呼び出し
            target_net.eval()#評価のためにドロップアウトetcをオフ
            #optimizer = optim.RMSprop(policy_net.parameters())#RMSで更新、ここはSGDにする、
            optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)  # 元々学習率はdefaultの1e-2、日本語版は変更
            memory = ReplayMemory(self.replay_memory)
            steps_done=0
            agent = DQN(steps_done,self.device)#Agent初期化


            #訓練開始
            for i_episode in range(self.num_episodes):
                # 環境と状態を初期化します
                self.env.reset()
                #state = self.env.reset()
                #state = torch.from_numpy(state)
                last_screen = self.get_screen()
                current_screen = self.get_screen()
                state = current_screen - last_screen
                for t in count():
                # アクションを選択して実行します
                    action = agent.select_action(state,self.eps_start,self.eps_end,self.eps_decay,policy_net,n_actions)#ここ確認
                    obs, reward, done, _ = self.env.step(action.item())# observation（行動後の状態）,reward（報酬）,done（終了判定）,info（デバッグなどの情報）が返り値
                    #observation=[カートの位置　カートの速度　ポールの角度　ポールの回転速度]が入っているので、これを状態として使うのはあり？差分でもなくそのまま
                    reward = torch.tensor([reward], device=self.device)

                # 新しい状態を観察します
                    last_screen = current_screen
                    current_screen = self.get_screen()
                    if not done:
                        next_state = current_screen - last_screen
                        #next_state = obs
                    else:
                        next_state = None
            
                # 遷移をメモリに保存します
                    memory.push(state, action, next_state, reward)

                # 次の状態に遷移
                    state = next_state
                    #state = torch.from_numpy(state)

                # 最適化のステップを(ターゲットネットワーク上で)実行します
                    agent.optimize_model(self.batch_size,memory,policy_net,target_net,self.gamma,optimizer)
                    if done:
                        self.episode_durations.append(t + 1)
                        print("SIMULATION:{} | EPISODE:{} | TotalStep:{}".format(sim,i_episode,t))  # 日本語版追記
                        # plot_durations()  # 日本語版コメントアウト
                        sum_rewards[sim,i_episode]=t
                        break

                if soft_update == True:
                    for target_param, value_param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(self.alpha_target * target_param.data + self.alpha_policy * value_param.data)
                elif soft_update == False:
                    # DQNの重みとバイアスをすべてコピーし、ターゲット・ネットワークを更新します
                    if i_episode % self.target_update == 0:#hard_update
                        target_net.load_state_dict(policy_net.state_dict())
        
        return sum_rewards