import torch
import os.path as osp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
from sd_3dgf import SD3DGFNetwork
from config import CONFIG
from utils.utils import build_model_name
from pytorch_lightning.callbacks import ModelSummary
logger = TensorBoardLogger(save_dir=CONFIG.METADATA.LOG_PATH)


    

model = SD3DGFNetwork()

model_name = build_model_name()

model_checkpoint = ModelCheckpoint(every_n_epochs=5)
early_stopping = EarlyStopping(monitor='epoch_loss', patience=20, mode='min')

# Set your own setting
trainer = Trainer(
    accelerator='gpu',
    max_epochs=CONFIG.TRAIN.MAX_EPOCH,
    callbacks=[model_checkpoint, early_stopping, ModelSummary(max_depth=0)],
    logger=logger,
    log_every_n_steps=1,
)


if CONFIG.TRAIN.RESUME is not None:
    ckpt_path=CONFIG.TRAIN.RESUME
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    trainer.fit(model=model)
else:
    trainer.fit(model=model)

torch.save(model.state_dict(), osp.join(CONFIG.METADATA.SAVE_PATH, model_name))


