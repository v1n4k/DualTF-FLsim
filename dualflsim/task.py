from torch.utils.data import Dataset, DataLoader


class TimeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(f"[TimeDataset DEBUG] Getting item at index: {idx}")
        return self.data[idx], self.labels[idx]


class FreqDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(f"[FreqDataset DEBUG] Getting item at index: {idx}")
        return self.data[idx], self.labels[idx]
