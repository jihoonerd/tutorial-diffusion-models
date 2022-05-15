from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CIFAR10(
            root=self.hparams.data_dir,
            train=True,
            transform=self.transforms,
            download=True,
        )
        self.test_dataset = CIFAR10(
            root=self.hparams.data_dir,
            train=False,
            transform=self.transforms,
            download=True,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )
