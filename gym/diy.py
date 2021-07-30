import time
import pickle
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
SIZE = 20
HM_EPISODES = 50000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 6000

start_q_table = None
LEARNING_RATE = 0.1
DISCOUNT = 0.95
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
d = {1: (255, 175,   0),
     2: (  0, 255,   0),
     3: (  0,   0, 255)}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def action(self, choice):
        if choice == 0:
            self.move(x=0, y= 1)
        elif choice == 1:
            self.move(x=0, y=-1)
        elif choice == 2:
            self.move(x= 1, y=0)
        elif choice == 3:
            self.move(x=-1, y=0)

    def move(self, x=0, y=0):
        self.x += x
        self.y += y
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f"on episode {episode}, epsilon: {epsilon}")
        ep_mean = np.mean(episode_rewards[-SHOW_EVERY:])
        print(f"    {SHOW_EVERY} episode mean {ep_mean}")
        show = True
    else:
        show = False
    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        player.action(action)
        
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE)*current_q \
                    + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
        
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[  food.y,   food.x] = d[FOOD_N]
            env[ enemy.y,  enemy.x] = d[ENEMY_N]
            env[player.y, player.x] = d[PLAYER_N]
            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300), resample=Image.BOX)
            cv2.imshow("", np.array(img))
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
        
        enemy.action(np.random.randint(0,4))
        food.action(np.random.randint(0,4))
    
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards,
        np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot(np.arange(moving_avg.size), moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel(f"episode #")
plt.show()

# with open(f"qtable-{int(time.time())}.plt", 'wb') as f:
with open("diy-qtable.plt", 'wb') as f:
    pickle.dump(q_table, f)