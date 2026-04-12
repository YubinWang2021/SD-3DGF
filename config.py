from dataclasses import dataclass
import os.path as osp
import torch 
from torchvision import transforms as T

@dataclass
class CONFIG:
    @dataclass
    class METADATA:
        LOG_PATH = 'work_space'
        SAVE_PATH = 'work_space/save'
        PRETRAINED_PATH = 'checkpoint'
        
    @dataclass
    class DATA:
        ROOT = 'data'
        DATASET = 'ccvid' # vccr, ccvid, ccvs, ccvr
        TEST_SET = 'ccvid'
        SAMPLE = True
        TRAIN_BATCH = 16 
        SAMPLING_STEP = 64
        NUM_WORKERS = 4
        HEIGHT = 256
        WIDTH = 128
        TEST_BATCH = 80
        NUM_INSTANCES = 4

    @dataclass
    class AUG:
        RE_PROB = 0.0
        TEMPORAL_SAMPLING_MODE = 'stride'
        SEQ_LEN = 8
        SAMPLING_STRIDE = 4

    @dataclass
    class MODEL:

        # @dataclass
        # class AP3D:
        #     TEMPERATURE = 4
        #     CONTRACTIVE_ATT = True
        DCE_CHAN = 64 
        NAME = 'sd_3dgf'
        RES4_STRIDE = 1
        BACKBONE = 'efficientnet' # others: 1. resnet, 2. clip, 3 .dinov2
        APP_FEATURE_DIM = 1024
        


    @dataclass
    class LOSS:
        CLA_LOSS = 'crossentropy'
        ETA = 0.90
        CST1_LOSS_WEIGHT = 0.40
        CST2_LOSS_WEIGHT = 0.25
        CLA_LOSS_WEIGHT = 1.
        ORG_LOSS_WEIGHT = 0.1
        GF3D_LOSS_WEIGHT = 0.9
        CLA_S = 16.
        CLA_M = 0.
        CLOTHES_CLA_LOSS = 'cosface'
        PAIR_LOSS = 'triplet'
        PAIR_LOSS_WEIGHT = 0.85
        PAIR_S = 16.
        PAIR_M = 0.3
        EPSILON = 0.1
        MOMENTUM = 0.

    @dataclass
    class TRAIN:
        @dataclass
        class LR_SCHEDULER:
            STEPSIZE = 8
            DECAY_RATE = 0.1

        @dataclass
        class OPTIMIZER:
            NAME = 'adam'
            LR = 0.0002
            WEIGHT_DECAY = 5e-4

        #TYPE = 'cloth' # cloth, pose
        TRAIN_MODE = 'standard' # 'one_cloth', 'standard','side_back'
        START_EPOCH = 0
        MAX_EPOCH = 120
        RESUME = None # add checkpoint here
    
    @dataclass
    class TEST:
        TYPE = 'pose' # pose, cloth
        TEST_MODE = 'all' # up, down, front, back, side, front_back, front_side, 
        