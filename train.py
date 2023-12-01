import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque
import random
from dqn import DQN
import sys
from game import Game
from time import sleep
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_position(num, m, n):
    """
    返回给定数字在 m*n 二维数组中的位置。
    
    :param num: 需要定位的数字（假设从0开始计数）。
    :param m: 数组的行数。
    :param n: 数组的列数。
    :return: 一个元组，表示数字在数组中的行和列。
    """
    row = num // n
    col = num % n

    # 检查数字是否在数组的范围内
    if row >= m or col >= n:
        return "数字超出了数组的范围"
    else:
        return (row, col)

def convert_to_onehotcode(state, inreplay=False):
    # print("OGSTATE",state)
    # if not inreplay:
    # state+=5
    state_tensor = torch.tensor(state, dtype=torch.int64).to(device)
    # print("one_hot",state_tensor)
    if (state_tensor < 0).any() or (state_tensor >= 15).any():
        raise ValueError("State contains out-of-bound values for one-hot encoding")
    state_tensor = F.one_hot(state_tensor, num_classes=15)
    # print(state_tensor.size())
    state_tensor = state_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return state_tensor.float()


def update_target_model(target_model, model, update_frequency):
    """ 每隔一定時間更新目標模型的權重 """
    if update_frequency % 10 == 0:  # 每10次迭代更新一次
        target_model.load_state_dict(model.state_dict())


# # 經驗回放
def replay(memory, target_model, model, optimizer, batch_size):
    gamma = 0.99  # 折扣因子
    
    minibatch = random.sample(memory, batch_size)

    for state, action, flag, reward, next_state, done in minibatch:
        # 转换为张量并移到正确的设备上
        # print("replay_1")
        # print(state)
        onehotstate = state+5
        state_tensor = convert_to_onehotcode(onehotstate, inreplay=True).float().to(device)
        # print("replay_2")
        # next_state_tensor = state_tensor = convert_to_onehotcode(onehotstate, inreplay=True).float().to(device)
        # next_state_tensor = convert_to_onehotcode(next_onehotstate, inreplay=True).float().to(device)
        next_onehotstate = next_state + 5
        next_state_tensor = convert_to_onehotcode(next_onehotstate, inreplay=True).float().to(device)
        
        reward_tensor = torch.tensor([reward], device=device)
        # done = torch.tensor([done], device=device)

        # 计算目标 Q 值
        with torch.no_grad():  # 在计算目标值时不需要梯度
            max_position, max_act = target_model(next_state_tensor)
            # print("max_position: ", max_position.size(),",", torch.max(max_position).size())
            # print("max_act: ", max_act.size(),",", torch.max(max_act).size())
            # position_target = (reward if done else reward + gamma * torch.max(max_position)).unsqueeze(0)
            # act_target = (reward if done else reward + gamma * torch.max(max_act)).unsqueeze(0)
            position_target = reward_tensor if done else reward_tensor + gamma * torch.max(max_position)
            act_target = reward_tensor if done else reward_tensor + gamma * torch.max(max_act)

        # 从当前模型获取预测值
        position, act = model(state_tensor)

        # reward = torch.tensor([reward], device=device).float()
        # done = torch.tensor([done], device=device).float()

        # # 用於計算損失的張量也應該是浮點型
        # position_target = torch.max(max_position)
        # act_target = act_target.float()
        # position_target = torch.max(max_position)
        # act_target = act_target.float()

        # 確保模型的輸出也是浮點型
        position, act = model(state_tensor)
        # print("QQQQQposition: ", position)

        qposition = position.gather(1, action)#.float()
        # print("position_gather: ", qposition)
        act = act.float()

        # 计算损失
        # position_loss = nn.MSELoss()(position, position_target)
        # act_loss = nn.MSELoss()(act, act_target)
        print("position: ", position.size())
        print("position_target: ", position_target.size())
        print("act: ", act.size())
        print("act_target: ", act_target.size())
        position_loss = nn.MSELoss()(position, position_target.unsqueeze(0))
        act_loss = nn.MSELoss()(act, act_target.unsqueeze(0))

        # 反向传播和优化
        total_loss = position_loss + act_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def train(state_size, prob):
    # 超參數設定
    learning_rate = 0.001
    max_episodes = 1000
    # gamma = 0.99  # 折扣因子
    epsilon = 1.0  # 探索率
    epsilon_decay = 0.995
    min_epsilon = 0.01
    batch_size = 64
    memory = deque(maxlen=10000)
    update_frequency = 0
    # 模型與優化器
    model = DQN(state_size).cuda()
    target_model = DQN(state_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    total_reward = 0
    # 訓練循環
    game = Game(state_size, prob)
    for episode in range(max_episodes):
        game.__init__(state_size, prob)
        state = np.zeros(state_size)-1
        done = False
        
        while not done:
            if np.random.rand() <= epsilon:
                # 探索
                can_be_choose = np.argwhere(state == -1)
                # print("CCCCVVVBBBB",can_be_choose.shape)
                # print(can_be_choose)
                if can_be_choose.shape[0] == 0:
                    can_be_choose = np.argwhere(state == -2)
                    # print(can_be_choose)
                    # print("ONLY FLAG")
                    choose = random.randint(0, can_be_choose.shape[0]-1)
                    act = True
                    # print(choose, act)
                    index_2d = can_be_choose[choose]
                    index_1d = index_2d[0]*state.shape[1]+index_2d[1]

                    # print("GoGO")

                else:
                    choose = random.randint(0, can_be_choose.shape[0]-1)
                    index_2d = can_be_choose[choose]
                    index_1d = index_2d[0]*state.shape[1]+index_2d[1]
                    # print("be choose", index)
                    act = bool(random.randint(0, 1))

            else:
                # print("AI")
                # 利用
                onehotstate = state+5
                state_tensor = convert_to_onehotcode(onehotstate).float()
                
                position, act = model(state_tensor)
                index_1d = np.argmax(position.detach().cpu().numpy())
                act = bool(np.argmax(act.detach().cpu().numpy()))
                # print(state_size[0]," , ",index)
                (row, col) = get_position(index_1d, state_size[0],state_size[1])
                index_2d = np.array([row, col])
                # print("AAAAAAAAAAIIIIIIII")
            print("IAIAIAIAIA",index_1d, "QQQQQQDDD",act)
            print("index_2d:", index_2d)
            next_state, reward, done = game.run(index_2d, act)

            memory.append((state, index_1d, act, reward, next_state, done))
            state = next_state
            # print("done: ",done)
            total_reward += reward

            if len(memory) > batch_size:
                replay(memory, target_model, model, optimizer, batch_size)

        update_target_model(target_model, model, update_frequency)
        update_frequency += 1

        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
        # sleep(1)

    # env.close()

if __name__ == '__main__':
    size = int(sys.argv[1]), int(sys.argv[2])
    prob = float(sys.argv[3])
    train(size, prob)