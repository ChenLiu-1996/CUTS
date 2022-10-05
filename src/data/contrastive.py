import numpy as np
import torch
import torchvision


class ContrastiveViewGenerator(object):
    """
    Generator for multiple views of the same instance, under the
    "Single-Instance Multi-View" Contrastive Learning paradigm.
    """

    def __init__(self, n_views=1):
        self.color_transform = torchvision.transforms.ColorJitter(
            brightness=0.7, contrast=0.8, saturation=0.6, hue=0.3)
        self.n_views = n_views

    def __call__(self, x):
        return x
        return torch.from_numpy(np.array([self.color_transform(x).detach().numpy() for i in range(self.n_views)]))[0]


def collate_contrastive(batch):
    """
    TODO: Currently only allows batch_size == 2. Need to generalize this.
    """
    generator = ContrastiveViewGenerator()

    data = np.array([item[0].detach().numpy() for item in batch])
    target = np.array([item[1].detach().numpy() for item in batch])

    data2 = np.flip(data, 2).copy()
    data3 = np.flip(data, 3).copy()

    data = torch.from_numpy(data)
    new_data1 = generator(data)
    new_data2 = torch.from_numpy(data2)
    new_data3 = torch.from_numpy(data3)

    new_data = torch.cat((new_data1, new_data2, new_data3), 0)

    target = torch.from_numpy(target)
    return [new_data, target]
