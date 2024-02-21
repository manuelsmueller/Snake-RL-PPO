import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from PPOa import ActorCritic

import os

if __name__ == '__main__':
    test_env2 = gym.make('CartPole-v1',render_mode="rgb_array")
    test_env2.reset(seed=1236)

    INPUT_DIM = test_env2.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = test_env2.action_space.n

    policy = ActorCritic.load(INPUT_DIM,HIDDEN_DIM, OUTPUT_DIM, file_name='model.pth')
    policy.evaluate(test_env2, True)