import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # no further processing here
        return x
    
    def save(self, file_name='model-Snake.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(),file_name)

    
    def load(model, input_dim, hidden_dim, output_dim, file_name='model-Snake.pth'):
        model_folder_path = './model'
        print("loading model from "+model_folder_path+"/"+file_name)
        model = Linear_QNet(input_dim,hidden_dim,output_dim)

        try:
            # todo: make try catch around!
            file_name = os.path.join(model_folder_path, file_name)
            loaded_model = torch.load(file_name)
            
            model.load_state_dict(loaded_model)
            model.eval()
        except Exception as e:
            # Dieser Block wird ausgeführt, wenn ein anderer Fehler auftritt
            print(f"Ein Fehler ist aufgetreten: file konnte nicht gefunden werden")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1,x)
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            game_over = (game_over, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # 2: r + gamma + max(nextPredQvalue)
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()
