# CS7642 RLDM Project #2



## Problem definition

The state has 8 components: horizontal and vertical position, horizontal and vertical velocity, angel and angular velocity, and left and right leg ground contact (binary). The action space consists of 1)do noting, 2)fire main engine, 3) fire left engine (push right), and 4) fire right engine (push left). The objective is to land the vehicle on the target fast and safely. Negative reward (punishment) will be imposed for incorrect landing and number of steps taken to finish the task. The criteria for solving this task is the average reward for consecutive 100 episodes should be larger than 200.

Since the state space is discrete, this problem can be solved with a table based approach, like the  Q-table. In this project, the deep q-learning approach is explored instead of a conventional Q-table method. 

## Deep Q-learning

The fundamental idea of deep-Q network is the combination of deep network and Q learning method. A neural network is used to estimate the Q value for each state and action:

$ Q = f(S, A)$

where $f()$ is the neural network to be implemented. The essence in this method is to estimate the Q value using function approximation. While a deep neural network can represent complex nonlinear behaviors, a simpler alternative is to use linear function to approximate the Q values. 

A replay mechanism is utilized to break the temporal dependency of the training steps. As the simulation steps advance, the state and action transition together  with the corresponding reward are added to the replay buffer. On each algorithm iteration, a random batch of samples from the replay buffer is selected and Q-Learning updates are applied to these samples, same as in a traditional Q-learning method. Deep Q-network (DQN) differs from Q-table in that the Q value is predicted by the underlying neural network, while in Q-table method, Q value is retrieve from Q-table. 

The tricky part is to find the Q value target to train the neural network. Since the goal of the DQN is to estimate the Q value, we can set the target using the Bellman equation. According to the optimal solution to the Bellman equation, the Q value target is just the reward of taking a action at that state plus the discounted highest Q values for the next state:

![image-20200701020048392](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701020048392.png)

The Deep Q-Learning algorithm selects actions according an ε-greedy policy.

I implemented the DQN algorithm proposed by DeepMind [1]  with Tensorflow. A DQN is built with the following structure:

1. Neural network with 2 hidden layers, each with 64 nodes
2. Activation function for both hidden layers is Rectified Linear activation (`ReLU`)
3. output layer has a linear activation function with `mse` loss function
4. Adam optimizer with learning rate of 0.001 (a hyper parameter that can be tuned )
5. $\epsilon$ -gready policy was used during training. During test, gready policy is utilized.

The training is finished after about 1100 episodes, taking about 7 hours on a laptop. Figure 1 plots the episode reward together with moving average reward using window size 100. 

![image-20200701022311454](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701022311454.png)

Figure 1. Episode reward and averaged reward over 100 episode during training. Training is finished at around 1100 episode, taking 7 hours on a laptop. 



Figure 2 plots the test results using the training model. Note that in the test, greedy policy is utilized to chose actions, instead of the  $\epsilon$-greedy policy used in training. 

![image-20200701032431558](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701032431558.png)

Figure 2. Test result using the trained agent. There still exist quite some fluctuations. 



## Experiments

 There are a lot of decision-making involved in the implementation: how many hidden layers, how many nodes in each layer, how to initialized the parameters, what activation function to use, etc. In this section,  I reported three experiments that explore the impact of hyper parameters.  `wandb` ([https://app.wandb.ai](https://app.wandb.ai/))  is used to log the model configuration and results. 

### Experiment 1:  Explore network structure

The difference between the three models in this experiment is the hidden layer size. The three models all have two hidden layers with equal number of nodes, the number of nodes in the hidden layers are 32, 64, and 128 respectively. Figure 3 plots the impact of network structure on the average reward. It seems that the difference between the models is not significant for current task.

![image-20200701013544496](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701013544496.png)

Figure 3. Impact of network structure on the average reward. The three models all have two hidden layers with equal number of nodes, the number of nodes in the hidden layers are 32, 64, and 128 respectively. Difference between the models is not significant for current task.



To further explore the impact of network depth, two network structures are configured. The first one is a deeper model with 3 hidden layers, with 128, 256, 128 nodes in the corresponding hidden layers. The second model is a simplistic network with 2 hidden years, with 32 nodes in each layer. Figure 4 plots the number of steps in each episode and the average reward over 100 episodes. It seems that increasing network depth does not necessarily improve the model performance, as indicated from the right figure. It's interesting to observe from the left figure that a deeper model reduces the number of steps taken in each episode, indicating that the lander is less prone to get stuck at local traps. 

![image-20200701004934342](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701004934342.png)

Figure 4. Impact of network depth. While a deeper model does not necessarily improve the model performance, it reduces the number of steps taken in each episode, indicating that the lander is less prone to get stuck at local traps. 

### Experiment 2: Change discount rate $\gamma$

In this experiment, three $\gamma$ values are tested:[0.9, 0.95, 0.99]. Small $\gamma$ values seem to stop the agent from finding a good solution (improving the episode reward). A value of 0.99 is able to get the agent eventually finish the training. 

![image-20200701030524045](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701030524045.png)

Figure 5. Impact of discount rate $\gamma$. Small values stop the agent from finishing the training.



### Experiment 3: Change learning rate ( $\alpha $ )

The learning rate determines how aggressive the q values are updated. A smaller learning rate means the q values is updated slower. A extreme case of 0 indicates that q values are never updated. Learning rate is an important parameter that impact how fast the model can be trained. In this experiment, I explored three learning rate value: [0.001, 0.01, 0.1].  The remaining parameters are intact. Figure 1 plots the impact of learning rate on number of steps  and the average rewards over the last 100 episodes. From the middle figure where the average rewards is plotted, it can be shown that larger learning rate yields more fluctuating results. The large fluctuation related to large learning rate is even more clear from the rightmost plot, where reward is not averaged. The agent is not able to learn at the higher learning rate.

The large learning rate also shows different pattern on the number of stops per episode. It seems that smaller learning rate tends to lead the lander to local "traps" and exhausts steps. Note that the current environment poses a 1000 step limit, as demonstrated from the truncated caps in the left figure. A larger learning rate yields smaller number of steps, indicating that the lander is less constrained by local traps. 

![image-20200701005211794](C:\Users\hyu\AppData\Roaming\Typora\typora-user-images\image-20200701005211794.png)

Figure 6. Impact of learning rate.  Larger learning rate yields more fluctuating results.



The plot on the left demonstrate the number of steps in a episode. A larger learning rate (0.1) seems to push the learning to occur quickly and number of steps is significantly smaller. 

## Reflections

As I read more about DQN, I discovered some modifications to DQN such as double DQN and Dueling DQN that seems promising. The idea of using two networks, one that changes slowly and one changes frequently and sync the parameters every now and then, makes sense since it tries to temporarily fix the target while updating the model.  If more time is allocated, I will try  and compare these variations of DQN. 

Tuning the hyper parameter of reinforcement learning models is by no means an easy job. It was quite frustrating to watch the reward growing at almost no speed, not knowing if it's an issue of incorrect implementation of the model or inappropriate setting of the hyper parameters. It was only towards the end of this project that I discovered the useful visualization tool `wandb`. It makes the logging of experiments so much easier and fun. It definitely deserves more use in the coming homework and projects. 

