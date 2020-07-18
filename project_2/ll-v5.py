import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
wandb.init(name='DoubleDQN-150x120-v1', project="cs76420-rldm-project2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.996)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()
wandb.config.update(args)

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(150, activation='relu'),
            Dense(120, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(args.lr))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(1):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        all_loss = []
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            # n_steps = 0
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                # n_steps += 1

            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()

            all_loss.append(total_reward)
            is_solved = np.mean(all_loss[-100:])
            if is_solved > 200:
                print('\n Task done!')
                break
            wandb.log({"Average_100": is_solved})

            # wandb.log({'Steps': n_steps})
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward})
            if total_reward >= 100:
                self.model.model.save(f'trained_model\model_100_ep_{ep}')


def main():
    env = gym.make('LunarLander-v2')
    agent = Agent(env)
    agent.train(max_episodes=1000)


if __name__ == "__main__":
    main()