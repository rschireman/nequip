"""NequIP integration with PyTorch Lightning

WARNING: Please note that there is NO GUERANTEE that this will give
the same results as our `Trainer`, and that it does not have
feature parity. Options that are valid for normal `nequip-train`
may be silently ignored.

Notable features that are not supported currently through their
NequIP YAML options include, BUT ARE NOT LIMITED TO:
 - Early stopping
 - LR scheduling
 - Gradient clipping
 - EMA
"""
import torch
import pytorch_lightning as pl

from torch_ema import ExponentialMovingAverage

from nequip.data import AtomicDataDict
from nequip.utils import Config, instantiate, instantiate_from_cls_name
from nequip.model import model_from_config
from nequip.train.loss import Loss, LossStat
from nequip.train.metrics import Metrics
from nequip.train._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION


class LitNequIP(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        # save the parameters so that they are visible to lighting
        self.save_hyperparameters(dict(config))
        # use the ones from lightning to enable lightning savenload
        # object is pretty similar to our Config so should work
        self.model = model_from_config(self.hparams)

        # now, we build other objects - copied from trainer.py
        self.loss, _ = instantiate(
            builder=Loss,
            prefix="loss",
            positional_args=dict(coeffs=self.hparams.loss_coeffs),
            all_args=self.hparams,
        )
        self.loss_stat = LossStat(self.loss)
        self.train_on_keys = self.loss.keys

        if self.hparams.use_ema:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.hparams.ema_decay,
                use_num_updates=self.hparams.ema_use_num_updates,
            )

        if hasattr(self.model, "irreps_out"):
            for key in self.train_on_keys:
                if key not in self.model.irreps_out:
                    raise RuntimeError(
                        "Loss function include fields that are not predicted by the model"
                    )

        # -- init metrics --
        if "metrics_components" not in self.hparams:
            metrics_components = []
            for key, func in self.loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.lower().startswith("perspecies"),
                }
                metrics_components.append((key, "mae", params))
                metrics_components.append((key, "rmse", params))
        else:
            metrics_components = self.hparams.metrics_components

        self.metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=self.hparams,
        )

        if not (
            self.hparams.metrics_key.lower().startswith(VALIDATION)
            or self.hparams.metrics_key.lower().startswith(TRAIN)
        ):
            raise RuntimeError(
                f"metrics_key should start with either {VALIDATION} or {TRAIN}"
            )

    def forward(self, data: AtomicDataDict) -> AtomicDataDict:
        # need enable_grad because lightning by default turns off
        # grad for validation, etc., but our models always (or almost
        # always) need it.
        # TODO: only do this when force/stress/etc. training
        with torch.enable_grad():
            return self.model(data)

    def training_step(self, batch, batch_idx):
        # lightning requires batch to be a list of tensors
        # so we need a special dataloader and a default key order
        # custom keys may not be possible?
        # TODO ^^^

        # TODO: reconstruct atomicdatadict
        data = {}

        if hasattr(self.model, "unscale"):
            # This means that self.model is RescaleOutputs
            # this will normalize the targets
            # in validation (eval mode), it does nothing
            # in train mode, if normalizes the targets
            data_unscaled = self.model.unscale(data)
        else:
            data_unscaled = data

        # Run model
        # We make a shallow copy of the input dict in case the model modifies it
        input_data = data_unscaled.copy()
        out = self.model(input_data)
        del input_data

        # compute the loss on the normal space output
        loss, loss_contrib = self.loss(pred=out, ref=data_unscaled)

        # If we are in training mode, we need to bring the prediction
        # into real units for metrics
        out = self.model.scale(out, force_process=True)
        # TODO: log these
        batch_losses = self.loss_stat(loss, loss_contrib)
        batch_metrics = self.metrics(pred=out, ref=data)

        # TODO: log loss contrib
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: rebuild atomicdatadict
        data: AtomicDataDict = {}

        # no need to call self.model.unscale, since it's no-op in validation
        input_data = data.copy()
        out = self.model(input_data)
        del input_data

        if hasattr(self.model, "unscale"):
            # loss function always needs to be in normalized unit
            scaled_out = self.model.unscale(out, force_process=True)
            _data_unscaled = self.model.unscale(data, force_process=True)
            loss, loss_contrib = self.loss(pred=scaled_out, ref=_data_unscaled)
            del _data_unscaled
        else:
            loss, loss_contrib = self.loss(pred=out, ref=data)

        # TODO log these
        # TODO: correctly accumulate these across multi GPU batches
        batch_losses = self.loss_stat(loss, loss_contrib)
        # in validation mode, data is in real units and the network scales
        # out to be in real units interally.
        # in training mode, data is still in real units, and we rescaled
        # out to be in real units above.
        batch_metrics = self.metrics(pred=out, ref=data)

    def configure_optimizers(self):
        # initialize optimizer
        optim, _ = instantiate_from_cls_name(
            module=torch.optim,
            class_name=self.hparams.optimizer_name,
            prefix="optimizer",
            positional_args=dict(
                params=self.parameters(), lr=self.hparams.learning_rate
            ),
            all_args=self.hparams,
            optional_args=self.hparams.get("optimizer_kwargs", {}),
        )
        return optim
