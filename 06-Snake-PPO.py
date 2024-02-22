import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
# import gymnasium as gym
from game2 import SnakeGameAI2

import os

# MLP = Multy Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # print("dbg forward 01")
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    # run forward through the network
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
    
    def load(input_dim, hidden_dim, output_dim, dropout = 0.5, file_name='model.pth'):
        model_folder_path = './model'
        print("loading model from "+model_folder_path+"/"+file_name)
        actor_critic_model = ActorCritic(MLP(input_dim,hidden_dim,output_dim),MLP(input_dim,hidden_dim,1))

        try:
            # todo: make try catch around!
            file_name = os.path.join(model_folder_path, file_name)
            loaded_model = torch.load(file_name)
            
            actor_critic_model.load_state_dict(loaded_model)
            actor_critic_model.eval()
        except Exception as e:
            # Dieser Block wird ausgeführt, wenn ein anderer Fehler auftritt
            print(f"Ein Fehler ist aufgetreten: file konnte nicht gefunden werden")

        return actor_critic_model

    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'

        print("saving model to "+model_folder_path+"/"+file_name)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(),file_name)

    def evaluate(policy,env, plot_trial = False):
        """
        enrole the trained policy to practical application.
        Therefore, take always the most likely outcome.
        """
        policy.eval()
        
        rewards = []
        done = False
        truncated = False
        episode_reward = 0

        state = env.reset()
        i = 0
        while not (done or truncated):
            i+=1
            state = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_pred, _ = policy(state)
                action_prob = F.softmax(action_pred, dim = -1)
                    
            action = torch.argmax(action_prob, dim = -1)    
            state, reward, done,truncated, _ = env.step(action.item())
            episode_reward += reward

            if plot_trial and i % 7 == 0:
                img = env.render()
                plt.imshow(img)
                plt.pause(0.2)
                print("i is ", i)
            
        return episode_reward

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def train(env:SnakeGameAI2, policy: ActorCritic, optimizer, discount_factor:float, ppo_steps:int, ppo_clip:float):
           
    policy.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    truncated = False
    episode_reward = 0

    state = env.reset()
    info = {}

     # print("state is",state)

    while not (done or truncated):
        state = torch.FloatTensor(state).unsqueeze(0)#convert (4,) to [1,4] tensor.
        #append state here, not after we get the next state from env.step()
        states.append(state)
        action_pred, value_pred = policy.forward(state)# policy(state) == policy.forward(state)
                
        # apply softmax = exp(xi)/(sum(exp(xj))), making a probability vector.
        # It is applied to all slices along dim, and will re-scale them so that the elements 
        # lie in the range [0, 1]  and sum to 1.
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()

        log_prob_action = dist.log_prob(action)
      
        # apply the action to the environment and simulate the result.
        # interface is: state, reward and (done, truncated)=(game over,abbort)
        state, reward, done,truncated, info = env.step(action.item())

        # store results in a list
        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward#calc cumulated reward
    
    # concatinate list of tensors / arrays to a proper tensor
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    # advantage is difference between estimated reward (value) and actual one (rewards). Normalize it.
    advantages = calculate_advantages(returns, values)
    
    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward, info

def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    # weight in reverse order => the more in the future the less influence
    # starting from origin
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        ref = returns.std()
        # edge case management.
        if ref == 0:
            ref = 1
            print("W: instability@calculate_returns. returns.std() is 0. Changing ref to 1.")
        returns = (returns - returns.mean()) / ref
        
    return returns

def calculate_advantages(returns, values, normalize = True):
    
    advantages = returns - values
    
    if normalize:
        ref = advantages.std()
        # edge case management.
        if ref == 0:
            ref = 1
            print("W: Instability@calculate_advantages. advantages.std() is 0. Changing ref to 1.")
        advantages = (advantages - advantages.mean()) / ref
        
    return advantages

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    """
    This is the update policy of PPO. 
    pre-condition: lists representing the experience from the last batch, i.e.
    - the states observed
    - the actions leading to the states
    - the probability of the actions
    - the error between expected reward and actual one
    - the weighted rewards (returns)

    1. sample 'ppo_steps' times 
        - state action pairs and
        - estimate the rewards
        - estimate the distribution of the action
        - determine, how likely the actions would have been chosen based on the latest policy.
        - calculate Kpis
    2. run backpropagation 
    3. return avg loss of plicy and value estimation
    """
    total_policy_loss = 0 
    total_value_loss = 0
    
    # get rid of additional info
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):               
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states) # retrieve predicted rewar and action for all states
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        # apply backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps




if __name__ == '__main__':

    env = SnakeGameAI2()

    # (11,256,3)
    INPUT_DIM = 13 #train_env.observation_space.shape[0]
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3 #train_env.action_space.n

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM) # actor is a 3 layer neural network with 1 hidden layer. Produces action vector
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1) # actor is a 3 layer neural network with 1 hidden layer. Produces scalar (value)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights) # applies init weights to all children

    LEARNING_RATE = 0.01 # LEARNING_RATE = 0.01

    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE) # use adam optimizer

    MAX_EPISODES = 200
    DISCOUNT_FACTOR = 0.90 # 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 1000
    PRINT_EVERY = 10
    PPO_STEPS = 5
    PPO_CLIP = 0.2

    train_rewards = []

    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward, info = train(env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
        
        train_rewards.append(train_reward)
        
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        
        if episode % PRINT_EVERY == 0:
            # print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} |')
            print(info)

        # if episode > 30:
        #     DISCOUNT_FACTOR = 0.9
        # if episode > 100:
        #     DISCOUNT_FACTOR = 0.95

        # if episode > 130:
        #     DISCOUNT_FACTOR = 0.98
        
        if mean_train_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    policy.save('SnakePPO.pth')
    

    