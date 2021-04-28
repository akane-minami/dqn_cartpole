import gym
from environment.sim import Simulator
from agent.network.dqn_network import DQN_Network

def main():

    BATCH_SIZE = 32#128
    GAMMA = 0.99#0.999
    EPS_START = 1.0
    EPS_END = 0.01#0.01
    EPS_DECAY = 200#30000stepでの減衰、今はエピソードでの減衰
    TARGET_UPDATE = 10# 元々10でしたが、日本語版は変更
    #NETWORK_ALPHA = 0.001
    NETWORK_ALPHA = 0.1#0.1以下だとうまく学習しない
    NUM_EPISODES = 1000  # 1000
    NUM_SIMULATIONS = 50#50
    REPLAY_MEMORY=10000

    #ゲームの環境生成
    env = gym.make('CartPole-v0').unwrapped

    sim = Simulator(env,NUM_EPISODES,NUM_SIMULATIONS,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE,NETWORK_ALPHA,REPLAY_MEMORY)
    ALGO_NAME = ['DQN','DQN_softupdate']
    NETWORK = [DQN_Network,DQN_Network]
    SOFT_UPDATE_FLAG = [False,True]
    sim.run(NETWORK,ALGO_NAME,SOFT_UPDATE_FLAG)

main()