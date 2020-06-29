
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import gym
import random
from tensorflow.keras.models import Sequential
# from keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# from keras.activations import relu, linear

import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)



class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0
        self.gamma = .96
        self.batch_size = 64
        self.epsilon_min = 0
        self.lr = 0.01
        self.epsilon_decay = .99
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.state_space, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict(next_states), axis=1))*(1-dones)
        # print(f'targets: {targets}')
        targets_full = self.model.predict(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = gym.make('LunarLander-v2')

model = tf.keras.models.load_model(r'C:\Users\hyu\Downloads\lunar-lander-DQN-master\trained_model\DQN_Trained.h5')

agent = DQN(env.action_space.n, env.observation_space.shape[0])
agent.model = model

n_episodes = 2000
loss = []
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, (1, 8))
    score = 0
    max_steps = 1000

    for i in range(max_steps):
        action = agent.act(state)
        # env.render()
        next_state, reward, done, _ = env.step(action)
        score += reward
        next_state = np.reshape(next_state, (1, 8))
        # agent.remember(state, action, reward, next_state, done)
        state = next_state
        # agent.replay()
        if done:
            print(f'step {i} , score: {score}')
            # print("episode: {}/{}, score: {}".format(e, episode, score))
            break
    loss.append(score)

    # Average score of last 100 episode
    is_solved = np.mean(loss[-100:])
    if is_solved > 200:
        print('\n Task Completed! \n')
        break

    print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
