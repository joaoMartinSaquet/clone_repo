import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import deque

import pandas as pd
import yaml
import matplotlib.pyplot as plt

# to remove pandas error
pd.options.mode.chained_assignment = None


# Define a simple dataset class

def read_dataset(datasets : str):
        df = pd.read_csv(datasets)

        data = df[["x", "y","x_to","y_to"]]
        # data['xlag'] = data["x"].shift(1)
        # data['ylag'] = data["y"].shift(1)

        data.fillna(0, inplace=True)
        # print(df.columns)
        y = df[['dx', 'dy']]
        return data, y

class FittsDataset(Dataset):
    def __init__(self,x, y):        
        self.data = x
        self.y_gt = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data.iloc[idx, :].to_numpy(), self.y_gt.iloc[idx, :].to_numpy()
        return self.data[idx, :], self.y_gt[idx, :]
    
class FittsDatasetSeq(Dataset):
    def __init__(self,x, y, sequence_length):        
        self.data = x
        self.y_gt = y
        self.seq_l = sequence_length

    def __len__(self):
        return len(self.data) - self.seq_l

    def __getitem__(self, idx):
        # return self.data.iloc[idx:idx+self.seq_l, :].to_numpy(), self.y_gt.iloc[idx + self.seq_l, :].to_numpy()
        return self.data[idx:idx+self.seq_l, :], self.y_gt[idx + self.seq_l, :]
        

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


    return train_losses, val_losses

# Load configuration from YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_model(model, val_dl, seq_l,  model_type, device):
    
    with torch.no_grad():
        ratio = 0
        for x, y in val_dl:
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = torch.round(model(x))

            d = torch.linalg.norm(y - y_pred)
            ratio += 1/(d+1)
            print(f"input : {x}, y target : {y}, pred : {y_pred}")
        ratio /= len(val_dl)
    print(f"validation ends Action Agreement Ratio {ratio}")

    model_values = []
    if model_type == "LSTM":

        i = 0
        x_init = val_dl[0][0].float().to(device)
        xs = deque([], seq_l)
        for k in range(seq_l):
            xs.appendleft(x_init)
        
        # xs.appendleft(x_init)
        model_positions = [xs[-1][0:2]] # get x, y
        for x, y in val_dl:

            torch.tensor(list(xs)).to(device)
            y_pred = torch.round(model(x))

            x = x.float().to(device)
            y = y.float().to(device)
            # get [x_to, y_to] 
            t_to = x[2:4]
            
            # (x,y)_t+1
            x_next = xs[-1][0:2] + y_pred

            i += 1


        model_positions = torch.cat(model_positions)
        x_in = torch.zeros((1,6), dtype=torch.float32).to(device)
        # x_in[0] = 
        x = deque(x_in, seq_l)
    
    # x = deque([])

            
if __name__ == "__main__":
    dataset_file = "datasets/P0_C0.csv"

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



    print(f"Training on {dataset_file}")
    data, y = read_dataset(dataset_file)
    scaler = StandardScaler()
    train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.2, random_state=42, shuffle=False)


    train_data = scaler.fit_transform(train_data)
    val_data = scaler.fit(val_data)
    if model_type == 'ANN':
        train_dataset = FittsDataset(train_data, train_y.to_numpy())
        val_dataset = FittsDataset(val_data, val_y.to_numpy())
        model = SimpleNN(train_dataset[0][0].shape[0], hidden_size, num_layers, output_size=2).to(device)
    else:
        train_dataset = FittsDatasetSeq(train_data, train_y.to_numpy(), sequence_length)
        val_dataset = FittsDatasetSeq(val_data, val_y.to_numpy(), sequence_length)
        model = LSTMModel(train_dataset[0][0].shape[-1], hidden_size, num_layers, output_size=2).to(device)


    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader =  DataLoader(train_dataset, batch_size, shuffle=True)
    # criterion
    criterion = torch.nn.MSELoss().to(device)
    # 



    opt = torch.optim.Adam(model.parameters(), learning_rate)
    loss_train, loss_val = training_loops(num_epochs, train_dataloader, val_dataloader, model, criterion, opt, device, 10)


    first_trial_x = data.iloc[0, :].to_numpy()
    first_trial_y = data.iloc[0, :].to_numpy()



    validate_model(model, val_dataloader, sequence_length, model_type, device)
    plt.plot(loss_train, label="train")
    plt.plot(loss_val, label="val loss")
    plt.title("training curves")
    plt.xlabel("num epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.ylim((0, 10))
    plt.show()
    # data