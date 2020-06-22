import gym
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.Logger(__name__, level=logging.DEBUG)
class QLearningAgent:
    """ Q-learning2 agent """

    def __init__(self, action_space, q_table):
        self.action_space = action_space
        self.q_table = q_table # np.zeros((n_states, action_space.n))
        self.alpha = 0.24
        self.gamma = 0.9
        self.epsilon = 0.09
        
        
        # self.Q_table = np.zeros()

    def act(self, observation):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action =self.action_space.sample()
        else:
            action = np.argmax(self.q_table[observation, :])
        return action

    def update(self, state, action, observation, reward):
        next_max = max(self.q_table[observation,:])
        
        self.q_table[state, action] = self.q_table[state, action] + \
                                      self.alpha*(reward + self.gamma*next_max - self.q_table[state, action])

check_points = [(462,4,-11.374),(398,3,4.348),(253,0,-0.585),(377,1,9.683),(83,5,-13.996)]

def check_q_convergence(q_table, check_points):
    total_error = 0.0
    for s,a,v in check_points:
        q = q_table[s,a]
        qv = trunc(q,3)
        total_error += (qv-v)**2
    return total_error / len(check_points)
        
def init_q_table(q_table_pickle, env):
    file = Path(q_table_pickle)
    if file.exists():
        with open(file, 'rb') as f:
            q_table = pickle.load(f)
            logging.info('q table pickle exists. Loading from file. ')
    else:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
        logging.info('q table pickle not exists. Init with np.zero. ')
    return q_table

def update_check_points(check_points, agent, env):

    pass      


def train_agent(agent, env, num_episodes=10,min_epsilon = 0.01, max_epsilon = 0.8, decay_rate = 0.01, check_points=check_points,
                min_alpha=0.01, max_alpha=0.3):
    q_mse_history = []
    episode = 0
    for i in tqdm(range(num_episodes)):
        old_q_table = agent.q_table.copy()
        state = env.reset()
        logging.info('env reset, state={state}')
        
        epochs = 0
        num_penalties, reward, total_reward = 0, 0, 0
        while reward != 20:
            action = agent.act(state)
            # action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            agent.update(state, action, observation, reward)
            state = observation
            total_reward += reward
            if reward == -10:
                num_penalties += 1
            epochs += 1
            q_mse = check_q_convergence(agent.q_table, check_points)
            if q_mse == 0:
                break
            
        new_q_table = agent.q_table
    
        q_diff = ((new_q_table - old_q_table)**2).mean()
        
        if episode % 10 == 0:
            logging.info(f'q_table first 10 rows: {new_q_table[0:10,:]}')
            with open('q_table_history.pickle','wb') as f:
                pickle.dump(agent.q_table, f)
                
        q_mse_history.append(q_mse)
        q_norm = np.linalg.norm(new_q_table)
        logging.info(f'at episode {episode}, q_mse = {q_mse}, q_diff={q_diff}, q_norm={q_norm},alpha={agent.alpha},epsilon={agent.epsilon}')
        if q_mse == 0:
            with open('q_table_converged.pickle','wb') as f:
                pickle.dump(agent.q_table,f)
        
                
        episode += 1
        agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        agent.alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-decay_rate * episode)
        
        logging.info(f'episode: {episode} done.')


        # total_penalties += num_penalties
        # total_epochs += epochs

    # print(agent.q_table)

def trunc(values, decs=3):
    return np.trunc(values*10**decs)/(10**decs)

def main():
    min_epsilon = 0.01
    max_epsilon = 1.0
    decay_rate = 0.0005
    min_alpha = 0.05
    max_alpha = 0.1

    logging.info('start')
    env = gym.make('Taxi-v3').unwrapped

    action_space = env.action_space
    
    q_table_pickle = 'q_table_history.pickle'
    q_table = init_q_table(q_table_pickle, env)
    agent = QLearningAgent(action_space=action_space, q_table=q_table)
    agent.epsilon = max_epsilon
    agent.alpha = max_alpha

    train_agent(agent=agent, env=env, num_episodes=10000, 
                min_epsilon=min_epsilon, max_epsilon=max_epsilon, decay_rate=decay_rate,
                min_alpha=min_alpha, max_alpha=max_alpha)
    Q = agent.q_table
    
    logging.info(f'Q[462,4] expected: -11.374, actual:{Q[462, 4]}')
    logging.info(f'Q[398,3] expected: 4.348, actual:{Q[398, 3]}')
    logging.info(f'Q[253,0] expected: -0.585, actual:{Q[253, 0]}')
    logging.info(f'Q[377,1] expected: 9.683, actual:{Q[377, 1]}')
    logging.info(f'Q[83,5] expected: -13.996, actual:{Q[83, 5]}')
    logging.info('finished.')

def check_quiz():
    file = 'q_table_history.pickle'
    with open(file, 'rb') as f:
        q = pickle.load(f)
        
    # quiz = [(126,0),(66,2),(11,3),(317,4),(146,1),(408,0),(172,1),(323,0),(293,3),(368,3)]
    quiz = [(132,3),(82,5),(392,5),(421,2),(361,0),(126,0),(119,5),(146,1),(96,2),(111,1)]
    for s,a in quiz:
        print(f'q({s},{a} = {q[s,a]}')
    
# def evaluate_agent(q_table, env, num_trials):
#     total_epochs, total_penalties = 0, 0
#
#     print("Running episodes...")
#     for _ in range(num_trials):
#         state = env.reset()
#         epochs, num_penalties, reward = 0, 0, 0
#
#         while reward != 20:
#             next_action = select_optimal_action(q_table,
#                                                 state,
#                                                 env.action_space)
#             state, reward, _, _ = env.step(next_action)
#
#             if reward == -10:
#                 num_penalties += 1
#
#             epochs += 1
#
#         total_penalties += num_penalties
#         total_epochs += epochs
#
#     average_time = total_epochs / float(num_trials)
#     average_penalties = total_penalties / float(num_trials)
#     print("Evaluation results after {} trials".format(num_trials))
#     print("Average time steps taken: {}".format(average_time))
#     print("Average number of penalties incurred: {}".format(average_penalties))
#

if __name__ == '__main__':
    main()

