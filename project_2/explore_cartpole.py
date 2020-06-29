#%%
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

REPLAY_SIZE = 10000
BATCH_SIZE = 32
ENV_NAME = 'CartPole-v0'
EPISODE = 10000
STEP = 300
class DqnAgent():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.gamma = 0.7
        self.learning_rate = 0.01
        self.epsilon = 0.09
        self.max_epsilon = 0.9
        self.min_epsilon = 0.01
        self.epsilon_time_decay = 0.9
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.create_Q_network()

    def create_Q_network(self):
        print(f'n states: {self.n_states}')
        model = Sequential([
            layers.Dense(32, input_dim=self.n_states, activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros'),  #, kernel_initializer='he_uniform'),
            layers.Dense(32, activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros'),  #, kernel_initializer='he_uniform'),
            layers.Dense(self.n_actions,activation='linear',kernel_initializer='he_uniform',bias_initializer='zeros') ##, name='q_value')
        ])
        model.compile(optimizer=Adam(learning_rate=0.01),
                     loss='mse')
        self.model = model


    def one_hot_encode(self, action):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[action] = 1
        return one_hot_action

    def learn(self, state, action, reward, next_state, done):
        # one_hot_action = self.one_hot_encode(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train()


    def train(self):
        self.time_step += 1
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)
        for data in mini_batch:
            state, action, reward, next_state, done = data
            y_reward = reward
            if not done:
                y_reward = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            # state_input = np.reshape(state,[4,-1])
            y = self.model.predict(state)
            y[0][action] = y_reward
            self.model.fit(state, y, epochs=1, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_time_decay
        #
        # state_batch = [data[0] for data in mini_batch]
        # action_batch = [data[1] for data in mini_batch]
        # reward_batch = [data[2] for data in mini_batch]
        #
        # # calc y
        # y_batch = []
        # Q_value_batch = self.model.predict(state)
    def act(self, state):
        # state_input = np.reshape(state, [4,-1])
        print(state)
        next_values = self.model.predict(state)
        return np.argmax(next_values[0])

    def act_epsilon_gready(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            # state_input = np.reshape(state, [4,-1])
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

def test_dqn():
    env = gym.make('CartPole-v0')
    agent = DqnAgent(env)
    model =agent.model
    a = np.array([1, 2, 3, 4])
    b = np.reshape(a, [1, 4])
    # model.fit(np.random.random([10,4]),np.random.random([10,2]))
    p = model.predict(b)
    print(f'input dim: {agent.n_states} ')
    print(p)

def main():
    env = gym.make(ENV_NAME)
    agent = DqnAgent(env)
    
    for episode in range(EPISODE):
        state = env.reset()
        state = np.reshape(state, [1,4])
        for step in range(STEP):
            action = agent.act_epsilon_gready(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            reward_agent = -1 if done else 0.1
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break




if __name__ == '__main__':
    main()
    # agent = DqnAgent()
    # test_dqn()



