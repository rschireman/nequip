""" Train a network."""
import logging

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch
import pytorch_lightning as pl

from nequip.data import dataset_from_config
from nequip.interfaces.lightning import LitNequIP

from nequip.scripts.train import default_config, parse_command_line, _set_global_options


def main(args=None, running_as_script: bool = True):

    config = parse_command_line(args)
    _set_global_options(config)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    from nequip.train.trainer import Trainer

    nq_trainer = Trainer(model=None, **dict(config))
    nq_trainer.set_dataset(dataset, validation_dataset)

    litmod = LitNequIP(config)

    trainer = pl.Trainer()
    trainer.fit(litmod, nq_trainer.dl_train, nq_trainer.dl_val)


if __name__ == "__main__":
    main(running_as_script=True)
