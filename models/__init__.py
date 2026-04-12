import logging
from typing import Tuple, Type, Union

from torch import nn

from .classifier import Classifier
from .vid_model import get_SD_3DGF

__all__ = ['build_models']
__factory = {
    'sd_3dgf': get_SD_3DGF
}

def build_models(config, num_ids: int = 150, train=True):

    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        model = __factory[config.MODEL.NAME](config)
    
    if train:
        id_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM,
                                                    num_classes=num_ids)

        return model, id_classifier
    else:
        return model


            
        