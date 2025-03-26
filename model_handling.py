import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json



def load_config(config_path):
    if config_path.endswith(".yaml"):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    elif config_path.endswith(".json"):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

def load_hyperparameters(config_path):

    config  = load_config(config_path)
    hyperparameters = config['hyperparameters']
    # Assign hyperparameters to variables
    hidden_size = hyperparameters['hidden_size']
    learning_rate = hyperparameters['learning_rate']
    num_epochs = hyperparameters['num_epochs']
    batch_size = hyperparameters['batch_size']
    num_layers = hyperparameters['num_layers']
    model_type = hyperparameters['model']
    sequence_length = hyperparameters['sequence_length']

    return hidden_size, learning_rate, num_epochs, batch_size, num_layers, model_type, sequence_length

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
