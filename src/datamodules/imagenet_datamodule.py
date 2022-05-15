from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = False,
        res: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(res),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage=None):
        self.train_dataset = ImageNet(
            root=self.hparams.data_dir,
            split="train",
            transform=self.transforms,
        )
        self.val_dataset = ImageNet(
            root=self.hparams.data_dir,
            split="val",
            transform=self.transforms,
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

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )
