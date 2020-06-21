# -*- coding: utf-8 -*-
"""
Created on Mon May 25 02:07:05 2020

@author: hyu
"""

import mdptoolbox
import numpy as np


# Calculate States
N = 9
isBadSide = np.array([0,1,1,1,0,0,0,1,0])
#%
run = 3
States = run*N+2  # from 0 to 2N, plus quit

isGoodSide = ~isBadSide+2  # [ 0, 0, 0, 1, 1, 1]
Die = np.arange(1, N+1)        # [1, 2, 3, 4, 5, 6]
dollar = Die * isGoodSide  # [0, 0, 0, 4, 5, 6]

# Create probability array==========================
prob = np.zeros((2, States, States))
# if leave
np.fill_diagonal(prob[0], 1)
# if roll
# Calculate probability for Input:
p = 1.0/N

# Create pro_1
# Create 1 X (1+run*N+2) array
zero = np.array([0]).repeat((run-1)*N+2) #Don't change it! It must have size= run-1)*N+1
isGoodSide_2 = np.concatenate((np.array([0]), isGoodSide, zero), axis=0) # rbind
# Create 1 X (run*N+3)*3 array
isGoodSide_N = np.concatenate((isGoodSide_2, isGoodSide_2), axis=0)
# Create 1 X ((run*N+3)^2 array
for i in range(0, run*N+2):
    isGoodSide_N = np.concatenate((isGoodSide_N, isGoodSide_2), axis=0)
    i = i + 1
# Create 1 X (2N+2)^2 array by trancation
isGoodSide_N = isGoodSide_N[:(States**2)]

isGoodSide_N = isGoodSide_N.reshape(States, States) # Reshaping (rows first)
prob[1] = np.triu(isGoodSide_N) # upper triangle matirx
prob[1] = prob[1]*p
prob_quit = 1 - np.sum(prob[1, :States, :States-1], axis=1).reshape(-1, 1) # last column

prob[1] = np.concatenate((prob[1, :States, :States-1], prob_quit), axis=1) #cbind
np.sum(prob[0], axis=1) # test row sum
np.sum(prob[1], axis=1) # test column sum

# Create rewards array==========================

rewards = np.zeros((2, States, States))
# if leave
rewards[0] = np.zeros((States, States))

# if roll
# Create roll reward array
# Create 1 X (1+run*N+2) array
#zero = np.array([0]).repeat((run-1)*N+2) #Don't change it! It must have size= run-1)*N+1
dollar_2 = np.concatenate((np.array([0]), dollar, zero), axis=0) # rbind
# Create 1 X (run*N+3)*3 array
dollar_N = np.concatenate((dollar_2, dollar_2), axis=0)
# Create 1 X ((run*N+3)^2 array
for i in range(0, run*N+2):
    dollar_N = np.concatenate((dollar_N, dollar_2), axis=0)
    i = i + 1
# Create 1 X (2N+2)^2 array by trancation
dollar_N = dollar_N[:(States**2)]

dollar_N = dollar_N.reshape(States, States) # Reshaping (rows first)
rewards[1] = np.triu(dollar_N) # upper triangle matirx
rewards[1] = rewards[1]
rewards_quit = - np.array(range(0 ,States)).reshape(-1, 1) #convert vector to n X 1 matrix
rewards[1] = np.concatenate((rewards[1, :States, :States-1], rewards_quit), axis=1) #cbind


vi = mdptoolbox.mdp.ValueIteration(prob, rewards, 1)
vi.run()

optimal_policy = vi.policy
expected_values = vi.V

print(optimal_policy)
print(expected_values)
print(f'answer:{max(expected_values)}')
