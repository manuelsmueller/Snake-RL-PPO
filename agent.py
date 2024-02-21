import torch
import random
import numpy as np

from collections import deque
from game import SnakeGameAI, Direction, Point

from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000

BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

class Agent:
    def __init__(self) -> None:
        self.number_of_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 # discount rate, smaller than 1
        self.memory = deque(maxlen = MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma) #Todo
        # TODO: model, trainer

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

             #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),    
            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),  

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x, #food left
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int) #convert bool to 0 or 1

    def remember(self,state,action,reward,next_state, game_over):
        self.memory.append((state,action,reward,next_state, game_over))# if max mem exceeded pop left

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, newxt_step, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, newxt_step, game_overs ) # todo

        # zip is equivalent for the following code.
        # for states, actions, rewards, newxt_step, game_overs in mini_sample:
        #     self.trainter.train_step(states, actions, rewards, newxt_step, game_overs ) # todo


    def train_short_memory(self,state,action,reward,next_state, game_over):
        self.trainer.train_step(state,action,reward,next_state, game_over)

    def get_action(self,state):
        #random_moves: trade-off between exploration and exploitation.
        self.epsilon = 80 - self.number_of_games
        final_move = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else: 
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)# executes forward function
            move = torch.argmax(prediction).item() #conver to int
            final_move[move] = 1

        return final_move 

def train(max_number_of_episodes=200, save_interval = 50):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent =Agent()
    game = SnakeGameAI()

    i = 1

    while i<max_number_of_episodes:
        #get tge old state
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old,final_move,reward,state_new, game_over)
        agent.remember(state_old,final_move,reward,state_new, game_over)

        if game_over:
            i+=1
            #train long memory, plot result
            game.reset()
            agent.number_of_games+=1
            agent.train_long_memory()
            if score > record:
                record = score
                # agent.model.save()

            print('Game:',agent.number_of_games,' Score:',score,' Record:',record)
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

            

        if i % save_interval == 0:
            print('safe agent...')
            filename="model-Snake"+str(i)+".pth"
            print(filename)
            agent.model.save(filename)
            i+=1 # Todo: actually not proper as this reduces the number of epoches...


if __name__ == '__main__':
    train(156,50)

