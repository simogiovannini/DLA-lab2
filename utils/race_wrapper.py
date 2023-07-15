from torch.utils.data import Dataset


class RaceWrapper(Dataset):
    # defining values in the constructor
    def __init__(self, data: Dataset, transform=None):
        self.dataset = data
        self.transform = transform
        self.len = len(data['example_id'])

    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.dataset['example_id'][idx], self.dataset['article'][idx], self.dataset['options'][idx], self.dataset['question'][idx], self.dataset['answer'][idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Getting data size/length
    def __len__(self):
        return self.len
