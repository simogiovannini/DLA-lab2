from torch.utils.data import Dataset


class AGWrapper(Dataset):
    # defining values in the constructor
    def __init__(self, data: Dataset, transform=None):
        self.dataset = data
        self.transform = transform
        self.len = len(data['text'])

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.dataset['text'][idx], self.dataset['label'][idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Getting data size/length
    def __len__(self):
        return self.len
