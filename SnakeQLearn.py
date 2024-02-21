import torch
import random
import numpy as np

from collections import deque
from game import SnakeGameAI, Direction, Point

from model import Linear_QNet, QTrainer
from helper import plot

from agent import Agent

MAX_MEMORY = 100_000

BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

"""
Oviously there is something missing. Loading the model differs from a completely untrained model but is significantly less
effective compared to the state when the system actually reached this state. 
For now: concentrate on applying PPO to snake game.
"""


if __name__ == '__main__':
    agent =Agent()
    game = SnakeGameAI()
    # dim = (11,256,3).. hardcoded ...
    input_dim = 11
    hidden_dim = 256
    output_dim = 3

    agent.model.load(input_dim, hidden_dim, output_dim, 'model-Snake150.pth')

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    max_i = 10

    while True:

        #get tge old state
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if game_over:
            #train long memory, plot result
            game.reset()
            agent.number_of_games+=1

            if score > record:
                record = score
                # agent.model.save()

            print('Game:',agent.number_of_games,' Score:',score,' Record:',record)
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

        if agent.number_of_games > max_i:
            break