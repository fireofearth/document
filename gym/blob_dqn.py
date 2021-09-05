import os
import math
import random
import numbers
from datetime import datetime
from collections import namedtuple, deque

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utility as util


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        """
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8).
        """
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=None, y=None):

        # If no value for x, move randomly
        if x is None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if y is None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv(object):
    SIZE          = 10
    RETURN_IMAGES = True
    MOVE_PENALTY  = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD   = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N   = 2  # food key in dict
    ENEMY_N  = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        # resizing so we can see our agent in all its glory.
        img = img.resize((300, 300), resample=Image.BOX)
        cv2.imshow("", np.array(img))  # show it!
        cv2.waitKey(50)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


config = util.AttrDict(
    episodes=20_000,
    map_channels=3,
    patch_size=[10, 10],
    action_space_size=9,
    replay_memory_maxlen=50_000,
    replay_memory_minlen=1_000,
    minibatch_size=64,
    learning_rate=0.001,
    discount=0.99,
    update_target_interval=5,
    eps_start=0.5,
    eps_end=0.001,
    eps_decay=0.99975,
    show_preview=False,
    aggregate_stats_interval=50
)

env = BlobEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
        ('curr_state', 'action', 'next_state', 'reward', 'is_done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.map_channels = config.map_channels
        self.patch_size = config.patch_size
        self.action_space_size = config.action_space_size
        self.convs = nn.Sequential(*[
                nn.Conv2d(self.map_channels, 256, 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Dropout(0.2),
                nn.Conv2d(256, 256, 3),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Dropout(0.2),
                nn.Flatten()])
        dummy = torch.ones((self.map_channels, *self.patch_size)).unsqueeze(0) \
                * torch.tensor(float('nan'))
        dummy = self.convs(dummy)
        self.fc_inputs = dummy.numel()
        self.fcs = nn.Sequential(*[
                nn.Linear(self.fc_inputs, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_space_size)])
    
    def forward(self, x, is_training=True):
        if is_training:
            self.train()
        else:
            self.eval()
        x = self.convs(x)
        return self.fcs(x)


class DQNAgent(object):
    MODEL_DIR = 'models/blob_dqn'

    def update_epsilon(self):
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.epsilon, self.eps_end)

    def __update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.terminal_state_counter = 0

    def __init__(self, config, action_space_size=9):
        self.action_space_size = action_space_size
        self.minibatch_size = config.minibatch_size
        self.replay_memory_maxlen = config.replay_memory_maxlen
        self.replay_memory_minlen = config.replay_memory_minlen
        self.update_target_interval = config.update_target_interval
        self.discount = config.discount
        self.epsilon = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.policy_net = DQNet(config).to(device)
        self.target_net = DQNet(config).to(device)
        self.target_net.eval()
        self.__update_target_model()
        self.replay_memory = ReplayMemory(config.replay_memory_maxlen)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        datestr = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.log_dir = f"{self.MODEL_DIR}/{datestr}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.terminal_state_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

    def update_replay_memory(self, *args):
        """
        Adds step's data to a memory replay array
        
        Parameters
        ==========
        transition : tuple
            Contains (curr_state, action, next_state, reward, is_done) where
            - curr_state : or s_t, an RGB image of dimensions (10, 10, 3) 
            - action     : or a_t, an integer 0-9.
            - next_state : or s_{t+1}, an RGB image of dimensions (10, 10, 3)
            - reward     : or r_t, a number
            - is_done    : a boolean specifying whether s_t is a terminal state
                           where the simulation ended.
        """
        self.replay_memory.push(*args)
    
    def select_action(self, state, is_training=True):
        sample = random.random()
        if is_training and sample < self.epsilon:
            action = random.randrange(self.action_space_size)
            return torch.tensor([[action]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                _, action = self.policy_net(state / 255., is_training=is_training).max(1)
                return action.view(1, 1)

    def train(self, is_terminal_state=False):
        if len(self.replay_memory) < self.replay_memory_minlen:
            # Start training only if replay memory reached a minimum size
            return
        # Get minibatch
        transitions = self.replay_memory.sample(self.minibatch_size)
        batch = Transition(*zip(*transitions))
        # Compute the current Q values
        curr_states = torch.cat(batch.curr_state) / 255.
        curr_Qs = self.policy_net(curr_states)
        # Extrac the Q values corresponding to agent actions
        actions = torch.cat(batch.action)
        curr_Qs = curr_Qs.gather(1, actions).squeeze(1)
        # Get rewards, and is_done flags.
        reward = torch.cat(batch.reward)
        is_done = torch.cat(batch.is_done)
        # Get the next states
        next_states = torch.cat([s for s in batch.next_state if s is not None])
        next_states = next_states / 255.
        # Compute the max of next Q values
        next_Qs = self.target_net(next_states, is_training=False)
        max_next_Qs, _ = next_Qs.max(dim=1)
        # Compute the expected Q values for TD error
        expected_Qs = torch.zeros(self.minibatch_size,
                dtype=torch.float, device=device)
        expected_Qs[~is_done] = reward[~is_done] + self.discount*max_next_Qs
        expected_Qs[is_done] = reward[is_done]
        # Model training step
        self.optimizer.zero_grad()
        loss = self.criterion(curr_Qs, expected_Qs)
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # Update target model if necessary
        if is_terminal_state:
            self.terminal_state_counter += 1
        if self.terminal_state_counter > self.update_target_interval:
            self.__update_target_model()
        return loss.item()

def train(config):
    agent = DQNAgent(config)
    episode_rewards = [-200]
    avgs_reward = []
    mins_reward = []
    maxs_reward = []
    for episode in tqdm(range(1, config.episodes + 1)):
        agent.writer.add_scalar('epsilon', agent.epsilon, episode)
        curr_state = env.reset()
        # PIL.Image -> ndarray shape is (H,W,C) (check?)
        # torch.nn.Conv2d inputs are (N,C,H,W)
        curr_state = torch.tensor([curr_state.transpose(2, 0, 1)],
                dtype=torch.float, device=device)
        is_done = False
        episode_reward = 0.
        episode_losses = []
        while not is_done:
            action = agent.select_action(curr_state)
            next_state, reward, is_done = env.step(action.item())
            next_state = torch.tensor([next_state.transpose(2, 0, 1)],
                    dtype=torch.float, device=device)
            episode_reward += reward
            if config.show_preview and episode % config.aggregate_stats_interval == 0:
                env.render()
            agent.update_replay_memory(
                    curr_state,
                    action,
                    None if is_done else next_state,
                    torch.tensor([reward], dtype=torch.float, device=device),
                    torch.tensor([is_done],dtype=torch.bool, device=device))
            loss = agent.train(is_terminal_state=is_done)
            if isinstance(loss, numbers.Number):
                episode_losses.append(loss)
            curr_state = next_state

        agent.update_epsilon()
        episode_rewards.append(episode_reward)
        avg_losses = sum(episode_losses) / len(episode_losses) if len(episode_losses) > 0 else 0.
        agent.writer.add_scalar('avg_loss', avg_losses, episode)
        if episode % config.aggregate_stats_interval == 0 or episode == 1:
            _episode_rewards = episode_rewards[-config.aggregate_stats_interval:]
            avg_reward = sum(_episode_rewards) / len(_episode_rewards)
            min_reward = min(_episode_rewards)
            max_reward = max(_episode_rewards)
            agent.writer.add_scalar('avg_reward', avg_reward, episode)
            agent.writer.add_scalar('min_reward', min_reward, episode)
            agent.writer.add_scalar('max_reward', max_reward, episode)
            avgs_reward.append(avg_reward)
            mins_reward.append(min_reward)
            maxs_reward.append(max_reward)
            # print(f"on episode {episode}; epsilon {agent.epsilon}")
            # print(f"    stats aggregated over the last {config.aggregate_stats_interval} episodes:")
            # print(f"    avg reward {avg_reward}")
            # print(f"    min reward {min_reward}")
            # print(f"    max reward {max_reward}")
    
    plt.plot(np.arange(len(avgs_reward)), avgs_reward, label='avg reward')
    plt.plot(np.arange(len(mins_reward)), mins_reward, label='min reward')
    plt.plot(np.arange(len(maxs_reward)), maxs_reward, label='max reward')
    plt.legend()
    plt.ylabel(f"rewards aggregated over the last {config.aggregate_stats_interval} episodes")
    plt.xlabel(f"episode #")
    plt.savefig(f"{agent.log_dir}/training-summary.png")
    return agent

should_train = True
if should_train:
    agent = train(config)
    torch.save(agent.policy_net.state_dict(), f"{agent.log_dir}/dqnet.pth")
else:
    pass
