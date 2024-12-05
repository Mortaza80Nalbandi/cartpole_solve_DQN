## Description
This repository hosts a Python implementation of the Deep Q-Network (DQN) algorithm, a powerful method within the realm of Deep Reinforcement Learning (DRL). We apply DQN to solve the Cartpole problem.
 The implementation aims to showcase the effectiveness of DQN in mastering this navigation problem while also serving as a reference for those interested in utilizing and practicing/learning DQN.
 we implemented the main alghoritm and checked the effectiveness of different hyperparameters.after that we implemented boltzman action policy method instead of epsilon-greedy and checked it's effects.


## Deep Q-Network (DQN)
The DQN algorithm is a value-based, model-free, and off-policy approach renowned for its capacity to learn optimal policies from high-dimensional input spaces. Originating from the efforts of researchers at DeepMind, DQN merges deep neural networks with traditional Q-learning to approximate the optimal state-action value function (Q function). The major pros and cons of the algorithm are as follows:



###### Advantages:
1. 	**Experience Replay Memory:** By utilizing exploration strategies like the epsilon-greedy policy and employing techniques such as experience replay, DQN significantly enhances sample efficiency and stabilizes the learning process for its main policy. This approach allows the algorithm to learn more effectively from past experiences and facilitates smoother convergence toward optimal policies.

###### Disadvantages:
1. 	**Hyperparameter Sensitivity:** DQN performance relies on tuning many hyperparameters, which makes it challenging to achieve optimal results in different environments.

2. 	**Training Instability:** During training, DQN may encounter instability, primarily originating from the dynamic nature of the target network. Furthermore, performance collapse can occur, presenting a scenario where DQN struggles to recover through learning, potentially hindering its training progress.


## Requirements
The code is implemented in Python 3.8.10 and has been tested on Windows 10 without encountering any issues. Below are the non-standard libraries and their corresponding versions used in writing the code:
<pre>
gymnasium==0.29.1
matplotlib==3.5.1
numpy==1.22.0
pygame==2.5.2
torch==2.0.1+cu118
</pre>

**Note:** This repository uses the latest version of Gymnasium for compatibility and optimization purposes. This code does not utilize any deprecated or old versions of the Gym library.



## Usage
instal gymnasium and run the code on test mode with wanted weights.
   

The network weights have been pre-saved in the many directories. Therefore, there's no requirement to start training from scratch for testing the code.



## Showcase
every approaches results can be shown in the said png of results

