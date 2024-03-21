# -*- coding: utf-8 -*-
# Copyright (c) IDEA, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for CoTDet.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_task"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # CoTDet model config
    cfg.MODEL.CoTDet = CN()
    cfg.MODEL.CoTDet.LEARN_TGT = False
    
    cfg.MODEL.CoTDet.KNOWLEDGE = CN()
    cfg.MODEL.CoTDet.KNOWLEDGE.TASK_NAME = ''
    cfg.MODEL.CoTDet.KNOWLEDGE.KNOWLEDGE_BASE = ''

    # loss
    cfg.MODEL.CoTDet.PANO_BOX_LOSS = False
    cfg.MODEL.CoTDet.SEMANTIC_CE_LOSS = False
    cfg.MODEL.CoTDet.DEEP_SUPERVISION = True
    cfg.MODEL.CoTDet.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.CoTDet.CLASS_WEIGHT = 4.0
    cfg.MODEL.CoTDet.DICE_WEIGHT = 5.0
    cfg.MODEL.CoTDet.MASK_WEIGHT = 5.0
    cfg.MODEL.CoTDet.BOX_WEIGHT = 5.
    cfg.MODEL.CoTDet.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.CoTDet.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.CoTDet.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.CoTDet.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.CoTDet.COST_BOX_WEIGHT = 5.
    cfg.MODEL.CoTDet.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.CoTDet.NHEADS = 8
    cfg.MODEL.CoTDet.DROPOUT = 0.1
    cfg.MODEL.CoTDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.CoTDet.ENC_LAYERS = 0
    cfg.MODEL.CoTDet.DEC_LAYERS = 6
    cfg.MODEL.CoTDet.INITIAL_PRED = True
    cfg.MODEL.CoTDet.PRE_NORM = False
    cfg.MODEL.CoTDet.BOX_LOSS = True
    cfg.MODEL.CoTDet.HIDDEN_DIM = 256
    cfg.MODEL.CoTDet.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.CoTDet.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.CoTDet.TWO_STAGE = True
    cfg.MODEL.CoTDet.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.CoTDet.DN="seg"
    cfg.MODEL.CoTDet.DN_NOISE_SCALE=0.4
    cfg.MODEL.CoTDet.DN_NUM=100
    cfg.MODEL.CoTDet.PRED_CONV=False

    cfg.MODEL.CoTDet.EVAL_FLAG = 1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # CoTDet inference config
    cfg.MODEL.CoTDet.TEST = CN()
    cfg.MODEL.CoTDet.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.CoTDet.TEST.SEMANTIC_ON = True
    cfg.MODEL.CoTDet.TEST.INSTANCE_ON = False
    cfg.MODEL.CoTDet.TEST.PANOPTIC_ON = False
    cfg.MODEL.CoTDet.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.CoTDet.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.CoTDet.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.CoTDet.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.CoTDet.TEST.PANO_TEMPERATURE = 0.06
    # cfg.MODEL.CoTDet.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.CoTDet.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "CoTDetEncoder"

    # transformer module
    cfg.MODEL.CoTDet.TRANSFORMER_DECODER_NAME = "CoTDetDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.CoTDet.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.CoTDet.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.CoTDet.IMPORTANCE_SAMPLE_RATIO = 0.75

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.Default_loading=True  # a bug in my d2. resume use this; if first time ResNet load, set it false
