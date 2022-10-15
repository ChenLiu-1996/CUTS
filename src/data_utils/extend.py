from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class ExtendedDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 desired_len: int):
        self.dataset = dataset
        self.desired_len = desired_len

    def __len__(self) -> int:
        return self.desired_len

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        return self.dataset.__getitem__(idx % len(self.dataset))
