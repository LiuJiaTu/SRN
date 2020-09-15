from easydict import EasyDict as edict
import numpy as np

cfg = edict()

#parse_args
cfg.dataset = 'widerface'
cfg.image_set = 'train'
cfg.dataset_path = 'data/widerface'
cfg.frequent = 20
cfg.kvstore = 'device' 
cfg.shuffle = True
cfg.pretrained = 'model/resnet-50'
cfg.prefix = 'model/srn'
cfg.lr = 0.01
cfg.lr_step = '100, 120, 130'

cfg.batch_size = 1

#data augmentation
cfg.hue_alpha = 18
cfg.contrast_alpha = 0.5
cfg.saturation_alpha = 0.5
cfg.brightness_alpha = 0.125
cfg.pixel_means = np.array([103.939, 116.779, 123.68])
cfg.pixel_stds = np.array([1.0, 1.0, 1.0])
cfg.pixel_scale = 1.0

#network params
cfg.image_size = 1024
cfg.use_init = False
cfg.use_deconv = False
cfg.alpha = 0.25
cfg.gamma = 2.0
cfg.stc_theta = 0.99

#anchor params
cfg.fs_neg_overlap = 0.3  #first step IOU threshold
cfg.fs_pos_overlap = 0.7  
cfg.ss_neg_overlap = 0.4  #second step IOU threshold
cfg.ss_pos_overlap = 0.5
cfg.num_anchors = 2

cfg.rpn_feat_stride = [4, 8, 16, 32, 64, 128]
cfg.rpn_anchor_cfg = {}
_ratio = (1.25,)
for _stride in [4, 8, 16, 32, 64, 128]:
    key = str(_stride)
    value = {'BASE_SIZE': _stride, 'RATIOS':_ratio, 'ALLOWED_BORDER':9999}
    scales = [2, 2**(1.0/2)*2]
    value['SCALES'] = tuple(scales)
    cfg.rpn_anchor_cfg[key] = value
        
    



