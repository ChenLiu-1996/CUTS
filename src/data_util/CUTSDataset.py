
from torch.utils.data import DataLoader, Dataset


class CUTSDataset(Dataset):

    def __init__(self, ids, preprocessed_data, labels, transform=False):
        self.dataset = preprocessed_data[ids]
        self.label = labels[ids]
        self.id = ids
        self.transform = transform
        self.generator = ContrastiveLearningViewGenerator()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        label = self.label[idx]
        if self.transform:
            sample = self.generator(sample)
        return sample, label
