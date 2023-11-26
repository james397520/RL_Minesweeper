# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np

from collections import deque
import random
# from dqn import DQN
import sys
from game import Game
from time import sleep

# 超參數設定
learning_rate = 0.001
max_episodes = 1000
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
memory = deque(maxlen=10000)

# 模型與優化器
# model = DQN(state_size, action_size)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)




# # 經驗回放
# def replay(memory, batch_size):
#     minibatch = random.sample(memory, batch_size)
#     for state, action, flag, reward, next_state, done in minibatch:
#         target = reward
#         if not done:
#             target = reward + gamma * torch.max(model(torch.tensor(next_state, dtype=torch.float))).item()
#         expected = model(torch.tensor(state, dtype=torch.float))[action]
#         loss = nn.MSELoss()(expected, torch.tensor(target))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

total_reward = 0
# 訓練循環
for episode in range(max_episodes):
    game = Game([8,8], 0.06)
    state = game.state
    done = False
    while not done:
        # if np.random.rand() <= epsilon:
        #     action = env.action_space.sample()  # 探索
        # else:
        #     action = torch.argmax(model(torch.tensor(state, dtype=torch.float))).item()  # 利用
        can_be_choose = np.argwhere(state == -1)
        # print("CCCCVVVBBBB",can_be_choose.shape)
        # print(can_be_choose)
        if can_be_choose.shape[0] == 0:
            can_be_choose = np.argwhere(state == -2)
            # print(can_be_choose)
            print("ONLY FLAG")
            choose = random.randint(0, can_be_choose.shape[0]-1)
            flag = True
            print(choose, flag)
            index = can_be_choose[choose]
            # self.board.handleClick(self.board.getPiece(index), flag)
            print("GoGO")
        else:
            choose = random.randint(0, can_be_choose.shape[0]-1)
            index = can_be_choose[choose]
            # print("be choose", index)
            flag = bool(random.randint(0, 1))
        # position, flag = model(torch.tensor(state, dtype=torch.float))
        next_state, reward, done = game.run(index, flag)
        # next_state, reward, done, _ = game.step(action)
        # memory.append((state, position, flag, reward, next_state, done))
        state = next_state
        print("done: ",done)
        total_reward += reward

    #     if len(memory) > batch_size:
    #         replay(memory, batch_size)

    # epsilon = max(epsilon * epsilon_decay, min_epsilon)
    print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
    sleep(1)

# env.close()
