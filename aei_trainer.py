import os
from omegaconf import OmegaConf
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from aei_net import AEINet


def main(args):
    model = AEINet(args)
    hp = OmegaConf.load(args.config)
    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(hp.log.chkpt_dir, args.name),
        monitor='val_loss',
        verbose=True,
        save_top_k=args.save_top_k,  # save all
    )

    trainer = Trainer(
        logger=pl_loggers.TensorBoardLogger(hp.log.log_dir),
        early_stop_callback=None,
        checkpoint_callback=checkpoint_callback,
        weights_save_path=save_path,
        gpus=-1 if args.gpus is None else args.gpus,
        distributed_backend='ddp',
        num_sanity_val_steps=1,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=hp.model.grad_clip,
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_interval,
        progress_bar_refresh_rate=1,
        max_epochs=10000,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-g', '--gpus', type=str, default=None,
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")
    parser.add_argument('-s', '--save_top_k', type=int, default=-1,
                        help="save top k checkpoints, default(-1): save all")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument('--val_interval', type=float, default=0.01,
                        help="run val loop every * training epochs")

    args = parser.parse_args()

    main(args)
