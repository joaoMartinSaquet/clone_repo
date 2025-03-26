import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque
from torch.utils.data import DataLoader


from dataset_handling import *
from model_handling import *


import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import json
# to remove pandas error
pd.options.mode.chained_assignment = None

# Define a simple neural network model_type
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleNN, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.fcI = nn.Linear(input_size, hidden_size)
        self.fa = nn.Tanh()
        self.fco = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fcI(x)
        out = self.fa(out)
        for l in self.hidden_layers:
            out = l(out)
            out = self.fa(out)
        out = self.fco(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def training_loops(epochs, train_dl, val_dl, model, criterion, opt, device, log_mod = 1):

    train_losses = []
    val_losses = []
    for e in range(epochs):
        epoch_loss = 0.0
        for x, y in train_dl:
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(x)
            loss = criterion(y,y_pred)
            # opt step
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
        train_losses.append(epoch_loss/len(train_dl))
        epoch_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.float().to(device)
                y = y.float().to(device)

                y_pred = model(x)
                loss = criterion(y, y_pred)
                
                epoch_loss += loss.item()

        val_losses.append(epoch_loss/len(val_dl))

        if e % log_mod == 0:
            print(f'Epoch [{e+1}/{epochs}], training loss: {train_losses[-1]:.4f}, validation loss: {val_losses[-1]:.4f}')


        if epoch_loss/len(train_dl) < 1e-4:
            print("traininf ended")
            break

    return train_losses, val_losses

# Load configuration from YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_model(model, val_dl, trial, seq_l,  model_type, scaler, device):
    

    with torch.no_grad():
        ratio = 0
        for x, y in val_dl:
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = torch.round(model(x))

            d = torch.linalg.norm(y - y_pred)
            ratio += 1/(d+1)
            # print(f"input : {x}, y target : {y}, pred : {y_pred}")
        ratio /= len(val_dl)
    print(f"validation ends Action Agreement Ratio {ratio}")
    if model_type == "LSTM":
        index_init = seq_l
    else:
        index_init = 1



    x_init = torch.tensor(trial[0][0]).float().to(device)
    xs = deque([], seq_l)
    for k in range(seq_l):
        xs.append(x_init)
    

    
    # xs.appendleft(x_init)
    model_positions = [xs[-1][0:2]]
    model_dx = [] # get x, y
    targets = [xs[-1][2:4]]
    t_to = torch.tensor(trial[0 ][0][2:4]).float().to(device)
    targets = [t_to]
    for i in range(index_init,len(trial) - 1):
        x, y = trial[i]
        if model_type == "LSTM":
            x_in = torch.vstack(list(xs)).to(device).unsqueeze(0)
        else:
            x_in = torch.tensor(xs[0])
        y_pred = torch.round(model(x_in))

        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        # get [x_to, y_to] 
            
            
        unscale_x = scaler.inverse_transform(xs[-1].cpu().detach().numpy().reshape(1, 5)) 
        disp = y_pred.cpu().detach().numpy()
        if model_type == "LSTM":
            new_unscale_x = unscale_x[0:2] + [disp[0][0], disp[0][1], 0, 0, 8]
        else:
            new_unscale_x = unscale_x[0:2] + [disp[0], disp[1], 0, 0, 8]

        x_next = torch.tensor(scaler.transform(new_unscale_x)[0]).to(device)
        # t_to = torch.tensor(trial[i][0][2:4]).float().to(device)
        targets.append(t_to)
        # print("taget to reach : ", t_to)

        # new_x = torch.tensor([x_next[0], x_next[1], t_to[0], t_to[1], trial[i+1][0][-1]]).float().to(device)
        new_x = torch.tensor([x_next[0], x_next[1], t_to[0], t_to[1], trial[i][0][-1]]).float().to(device)
        xs.append(new_x)
            
        model_positions.append(x_next[0:2])
        model_dx.append(y_pred[0])


    return torch.vstack(model_positions).cpu().detach().numpy(), torch.vstack(model_dx).cpu().detach().numpy(), torch.vstack(targets).cpu().detach().numpy()
    # x = deque([])


            
if __name__ == "__main__":
    dataset_file = "datasets/P2_C0.csv"
    logs_dir = "resulat/ann/P2_C0/log.txt"
    load = False
    
    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    else:
        print("using CPU")
        torch.device("cpu")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load hyperparameters
    config = load_config('ann_config.yaml')
    hyperparameters = config['hyperparameters']
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    # Assign hyperparameters to variables
    hidden_size = hyperparameters['hidden_size']
    learning_rate = hyperparameters['learning_rate']
    num_epochs = hyperparameters['num_epochs']
    batch_size = hyperparameters['batch_size']
    num_layers = hyperparameters['num_layers']
    model_type = hyperparameters['model']
    sequence_length = hyperparameters['sequence_length']

    logs_dir = "resultat/ann/" + model_type + "/"

    print("logging on " + logs_dir)
    print(f"Training on {dataset_file}")
    data, y = read_dataset(dataset_file)
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data, val_data, train_y, val_y = train_test_split(data.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42, shuffle=False)


    train_data = scaler.fit_transform(train_data)
    val_data = scaler.fit(val_data)

    if model_type == 'ANN':
        train_dataset = FittsDataset(train_data, train_y)
        val_dataset = FittsDataset(val_data, val_y)
    else:
        train_dataset = FittsDatasetSeq(train_data, train_y, sequence_length)
        val_dataset = FittsDatasetSeq(val_data, val_y, sequence_length)

    if load:
        model = torch.load(logs_dir + "best_one/model.pt", weights_only=False)
    else:
        if model_type == 'ANN':
            model = SimpleNN(train_dataset[0][0].shape[0], hidden_size, num_layers, output_size=2).to(device)
        else:

            model = LSTMModel(train_dataset[0][0].shape[-1], hidden_size, num_layers, output_size=2).to(device)


    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader =  DataLoader(train_dataset, batch_size, shuffle=False)
    # criterion
    criterion = torch.nn.MSELoss().to(device)
    # 


    if not load:
        opt = torch.optim.Adam(model.parameters(), learning_rate)
        loss_train, loss_val = training_loops(num_epochs, train_dataloader, val_dataloader, model, criterion, opt, device, 100)
        torch.save(model, logs_dir + "model.pt")
    
    first_trial_x_df = data.iloc[0:210, :].to_numpy()
    first_trial_y = y.iloc[0:210, :].to_numpy()

    first_trial_x = scaler.transform(first_trial_x_df)
    trial_dataset = FittsDataset(first_trial_x, first_trial_y)

    position, dx, targets = validate_model(model, val_dataloader, trial_dataset, sequence_length, model_type, scaler, device)
    log_sim = np.hstack((position, targets, 0.8 * np.ones((position.shape[0], 1))))
    log_sim = scaler.inverse_transform(log_sim)
    
    if not load:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(loss_train, label="train")
        ax.plot(loss_val, label="val loss")
        ax.set_title("training curves")
        ax.set_xlabel("num epochs")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True)
        ax.set_ylim((0, 10))
        fig.savefig(logs_dir + "training_curves.png")
                
        with open(logs_dir + "log.txt", "w") as f:
            f.write("loss_train: " + str(loss_train) + "\n")
            f.write("loss_val: " + str(loss_val) + "\n")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(log_sim[:,0], log_sim[:, 1], '.', label="agent movement")
    
    n = len(log_sim)
    ax.plot(first_trial_x_df[:n,0], first_trial_x_df[:n, 1], '.',  label="target movement")
    ax.plot(log_sim[:, 2], log_sim[:, 3], 'rx', label="target")
    ax.set_title("agent movement")
    ax.legend()


  

    

    with open(logs_dir + "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f)

    fig.savefig(logs_dir + "agent_movement.png")


    plt.show()
    # data