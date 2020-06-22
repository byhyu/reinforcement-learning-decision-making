import gym
import numpy as np
import time, pickle, os

# input
amap="SFFHFFHFG";
gamma=0.94;
alpha=0.25;
epsilon=0.06;
n_episodes=30017;
seed=222980

# amap = "SFFG"
# gamma = 1.0
# alpha = 0.24
# epsilon = 0.09
# n_episodes = 49553
# seed = 202404
#
# amap = "SFFFHFFFFFFFFFFG"
# gamma = 1.0
# alpha = 0.25
# epsilon = 0.29
# n_episodes = 14697
# seed = 741684


# test case 3
# amap="SFFFFHFFFFFFFFFFFFFFFFFFG";
# gamma=0.91
# alpha=0.12
# epsilon=0.13
# n_episodes=42271
# seed=983459

# end input
np.random.seed(seed)


def reshape_amap(amap):
    n_elements = len(amap)
    n_rows = int(np.sqrt(n_elements))
    map_array = np.array(list(amap))
    return map_array.reshape(n_rows, n_rows)


desc = reshape_amap(amap)
env = gym.make('FrozenLake-v0', desc=desc).unwrapped
env.seed(seed)

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action_q_learning(state):
    action = np.argmax(Q[state, :])
    return action

def choose_action(state):
    action = 0
    if np.random.random() >= epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(4)
    #
    # # if np.random.uniform(0, 1) < epsilon:
    #     action = env.action_space.sample()
    # else:
    #     action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + alpha * (target - predict)


def print_policy(Q):
    action_map = {0: '<', 1: 'v', 2: '>', 3: '^'}
    n_states = Q.shape[0]
    policy = []
    for s in range(n_states):
        a = choose_action(s)
        policy.append(a)
    return ''.join([action_map[k] for k in policy])


def print_policy2(Q):
    action_map = {0: '<', 1: 'v', 2: '>', 3: '^'}
    n_states = Q.shape[0]
    policy = []
    for s in range(n_states):
        a = np.argmax(Q[s, :])
        policy.append(a)
    return ''.join([action_map[k] for k in policy])


# Start
# rewards = 0

for episode in range(n_episodes):
    state = env.reset()
    action = choose_action_q_learning(state)

    while True:  # t < max_steps:
        # env.render()
        state2, reward, done, info = env.step(action)
        action2 = choose_action_q_learning(state2)
        learn(state, state2, reward, action, action2)
        state = state2
        action = action2
        # rewards += 1
        if done:
            break
        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # os.system('clear')
        # time.sleep(0.1)

# print("Score over time: ", rewards / total_episodes)
# print(Q)
policy = print_policy2(Q)
print(f'policy:{policy}')
# with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
#     pickle.dump(Q, f)

# ^>>>   ><>>  >vvv  >>vv  >>>>  v>>^<
