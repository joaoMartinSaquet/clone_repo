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
import numpy as np
import json


def validate_model(act_fun, val_dl, trial, seq_l,  model_type, scaler, device):
    

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

    i = 0
    x_init = torch.tensor(trial[0][0]).float().to(device)
    xs = deque([], seq_l)
    for k in range(seq_l):
        xs.appendleft(x_init)
        
    # xs.appendleft(x_init)
    model_positions = [xs[0][0:2]]
    model_dx = [] # get x, y
    targets = [xs[0][2:4]]

    for x, y in trial:
        if model_type == "LSTM":
            x_in = torch.vstack(list(xs)).to(device).unsqueeze(0)
        else:
            x_in = torch.tensor(xs[0])
        y_pred = torch.round(model(x_in))

        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        # get [x_to, y_to] 
            
            
        unscale_x = scaler.inverse_transform(xs[0].cpu().detach().numpy().reshape(1, 5)) 
        disp = y_pred.cpu().detach().numpy()
        if model_type == "LSTM":
            new_unscale_x = unscale_x[0:2] + [disp[0][0], disp[0][1], 0, 0, 8]
        else:
            new_unscale_x = unscale_x[0:2] + [disp[0], disp[1], 0, 0, 8]

        x_next = torch.tensor(scaler.transform(new_unscale_x)[0]).to(device)
        t_to = x[2:4]
        targets.append(t_to)
        # print("taget to reach : ", t_to)
        new_x = torch.tensor([x_next[0], x_next[1], t_to[0], t_to[1], 8]).float().to(device)
        xs.appendleft(new_x)
            
        model_positions.append(x_next[0:2])
        model_dx.append(y_pred[0])

        i += 1


    return torch.vstack(model_positions).cpu().detach().numpy(), torch.vstack(model_dx).cpu().detach().numpy(), torch.vstack(targets).cpu().detach().numpy()



if __name__ == "__main__":

    model = torch.load("resultat/ann/LSTM/best_one/model.pt")
        
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_model(model, val_dl, trial, seq_l,  model_type, scaler, device)
