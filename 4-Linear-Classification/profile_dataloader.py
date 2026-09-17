import time
import torch
from torchvision import transforms
import torchvision
from d2l import torch as d2l

class FashionMNIST(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(32, 32)):
        super().__init__()
        self.save_hyperparameters()

        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True,
            transform=trans, download=True
        )

        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False,
            transform=trans, download=True
        )

@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(
        data,
        self.batch_size,
        shuffle=train,
        num_workers=self.num_workers
    )

data = FashionMNIST(batch_size=64, resize=(32, 32))
data.num_workers = 4

for X, y in data.val_dataloader():
    pass