# Project 2 readme

## Problem introduction



## Lunar Lander Environment

## Deep Q-Learning implementation

My implementation is inspired by the Deep Q-Learning algorithm as described in reference [2]. The input to my Deep Q-Learner are the observations of the Lunar Lander environment. The Deep Neural Network I used, is implemented in Keras (using Tensor Flow as backend). In short, the Deep Q-Learning algorithm selects actions according an ε-greedy policy. Each experience tuple <s, a, r, s’> is stored in a Replay Memory structure. On each algorithm iteration, a random sample of these stored memories (minibatch) is selected and Q-Learning updates are applied to these samples. The detailed algorithm and the advantages of this approach are described in detail in reference [2].

## How to run

### Dependencies

```
- h5py=2.9.0
- numpy=1.17.3
- matplotlib=3.0.0 
- python=3.6.7
- scipy=1.2.1
- setuptools=39.1.0
- tensorflow=1.15.0 
- cython=0.29
- pip:
  - gym==0.17.0
  - box2D-py 
```

### Run

The script for DQN implementation and main function to train the agent is in a single file `lunar_lander_dqn.py`. It can be run from IDE such as Pycharm, or from command line by `$python lunar_lander_dqn.py`

## Training example



