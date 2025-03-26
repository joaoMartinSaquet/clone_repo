import pandas as pd
from torch.utils.data import Dataset


def read_dataset(datasets : str):
        df = pd.read_csv(datasets)

        data = df[["x", "y","x_to","y_to", "dt"]]
        # data['dt'] = df["dt"]
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
        