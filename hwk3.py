import gym
import numpy as np

amap = "SFFG"
gamma = 1.0
alpha = 0.24
epsilon = 0.09
n_episodes = 3000
seed = 202404
np.random.seed(seed)

class SarsaAgent:
    """ SARSA agent """
    def __init__(self, action_space):
        self.action_space = action_space
        self.Q_table = np.zeros()

    def act(self, observation, reward, done):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q_table[observation, :])
        return self.action_space.sample()

def reshape_amap(amap):
    n_elements = len(amap)
    n_rows = int(np.sqrt(n_elements))
    map_array = np.array(list(amap))
    return map_array.reshape(n_rows,n_rows)

if __name__ == '__main__':


    # reshape ampa to square shape
    desc = reshape_amap(amap)
    env = gym.make('FrozenLake-v0', desc=desc).unwrapped
    env.seed(seed)
    agent = SarsaAgent(env.action_space)

    done = False
    reward = 0

    for i in range(n_episodes):
        observation = env.reset()
        while True:
            env.render()
            action = agent.act(observation, reward, done)
            observation, reward, done, info= env.step(action)

            if done:
                break
    env.close()




