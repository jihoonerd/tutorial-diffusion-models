from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from src.models.ema import EMAModel
from src.models.components.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from src.models.components.resample import (
    create_named_schedule_sampler,
    LossAwareSampler,
)
from torchvision.utils import make_grid
import math


class DDPMLitModule(LightningModule):
    """Improved DDPM Model"""

    def __init__(
        self,
        net: torch.nn.Module,
        diffusion_steps: int = 1000,
        beta_schedule: str = "cosine",
        model_mean_type: str = "epsilon",
        model_var_type: str = "learned_range",
        loss_type: str = "rescaled_mse",
        schedule_sampler: str = "uniform",
        lr: float = 0.0001,
        weight_decay: float = 0.0,
        ema_start: int = 5000,
        ema_update: int = 100,
        ema_decay: float = 0.995,
        sample_every: int = 10000,
        num_sample_imgs: int = 9,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(beta_schedule, diffusion_steps),
            model_mean_type=ModelMeanType[model_mean_type.upper()],
            model_var_type=ModelVarType[model_var_type.upper()],
            loss_type=LossType[loss_type.upper()],
        )
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

        self.ema_model = EMAModel(model=self.net, decay=self.hparams.ema_decay)

    def reset_ema_parameters(self):
        self.ema_model.model.load_state_dict(self.net.state_dict())

    def step_ema(self):
        if self.global_step <= self.hparams.ema_start:
            self.reset_ema_parameters()
        else:
            self.ema_model.update(self.net)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch):
        img, label = batch
        t, weights = self.schedule_sampler.sample(img.shape[0], self.device)
        losses = self.diffusion.training_losses(self.net, img, t)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        self.log("loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):

        if self.global_step % self.hparams.ema_update == 0:
            self.step_ema()

        if self.global_step % self.hparams.sample_every == 0:
            res = self.net.image_size
            sampled_img = self.diffusion.p_sample_loop(
                model=self.ema_model.model,
                shape=(self.hparams.num_sample_imgs, 3, res, res),
            )
            grid_img = make_grid(
                sampled_img, nrow=int(math.sqrt(self.hparams.num_sample_imgs))
            )
            self.logger.experiment.add_image(f"x_0", grid_img, self.global_step)
