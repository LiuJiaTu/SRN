from __future__ import print_function
import sys
import logging
import datetime
import numpy as np

from rcnn.logger import logger
from rcnn.config import cfg
from .image import get_crop_image
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform

def get_crop_batch(roidb):
    data_list = []
    label_list = []
    imgs, roidb = get_crop_image(roidb)
    assert len(imgs)==len(roidb)
    
    for i in range(len(imgs)):
        data = {'data':imgs[i]}
        label = {'gt_boxes':roidb[i]['boxes']}
        data_list.append(data)
        label_list.append(label)
    
    return data_list, label_list

class AnchorLoader:
    def __init__(self, feat_shape):
        feat_strides = cfg.rpn_feat_stride
        feat_infos = []
        anchors_num_list = []
        anchors_list = []
        inds_inside_list = []
        DEBUG = False
        
        for i in range(len(feat_strides)):
            #generate base anchors_list
            stride = feat_strides[i]
            _stride = str(stride)
            base_size = cfg.rpn_anchor_cfg[_stride]['BASE_SIZE']
            allowed_border = cfg.rpn_anchor_cfg[_stride]['ALLOWED_BORDER']  #allowed_border=9999
            ratios = cfg.rpn_anchor_cfg[_stride]['RATIOS']
            scales = cfg.rpn_anchor_cfg[_stride]['SCALES']
            base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32), stride = stride)
            
            #generate all anchors_list
            feat_height, feat_width = feat_shape[i][-2:]
            feat_infos.append([feat_height, feat_width])
            A = cfg.num_anchors
            K = feat_height * feat_width
            all_anchors = anchors_plane(feat_height, feat_width,stride, base_anchors)
            all_anchors = all_anchors.reshape((K * A, 4))
            total_anchors = int(K * A)
            anchors_num_list.append(total_anchors)
            
            #only keep anchors inside the image
            inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                                   (all_anchors[:, 1] >= -allowed_border) &
                                   (all_anchors[:, 2] < cfg.image_size + allowed_border) &
                                   (all_anchors[:, 3] < cfg.image_size + allowed_border))[0]
            if DEBUG:
                print('total_anchors', total_anchors)
                print('inds_inside', len(inds_inside))
            anchors = all_anchors[inds_inside, :]
            anchors_list.append(anchors)
            inds_inside_list.append(inds_inside)
        
        #all filtered anchors
        anchors = np.concatenate(anchors_list)
        for i in range(1, len(inds_inside_list)):
            inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
        inds_inside = np.concatenate(inds_inside_list)
        
        self.anchors = anchors
        self.inds_inside = inds_inside
        self.anchors_num_list = anchors_num_list
        self.feat_infos = feat_infos
        self._times = [0.0]
        
    @staticmethod
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    def assign_anchor_fpn(self, gt_label, prefix='face'):
        #ta = datetime.datatime.now()
        gt_boxes = gt_label['gt_boxes']
        
        feat_strides = cfg.rpn_feat_stride
        bbox_pred_len = 4
        landmark_pred_len = 10
        anchors = self.anchors
        inds_inside = self.inds_inside
        anchors_num_list = self.anchors_num_list
        total_anchors = sum(anchors_num_list)
        feat_infos = self.feat_infos
        
        #generate cls labels
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        # label: 1 is positive, 0 is negative, -1 is ignore
        labels.fill(-1)
        if gt_boxes.size > 0:
            #overlap between the anchors and the gt boxes
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            labels[max_overlaps < cfg.fs_neg_overlap] = 0
            labels[max_overlaps >= cfg.fs_pos_overlap] = 1
        else:
            labels[:] = 0
        
        #generate bbox loc labels
        bbox_targets = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
        if gt_boxes.size > 0:
            bbox_targets[:,:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
        bbox_weights = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
        bbox_weights[labels==1, :] = 1.0
        
        labels = AnchorLoader._unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = AnchorLoader._unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_weights = AnchorLoader._unmap(bbox_weights, total_anchors, inds_inside, fill=0)
        
        label = {}
        anchors_num_range = [0] + anchors_num_list
        for i in range(len(feat_strides)):
            stride = feat_strides[i]
            feat_height, feat_width = feat_infos[i]
            A = cfg.num_anchors
            
            #split labels for every feature map
            _label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
            bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
            bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]] 
            
            #reshape labels
            _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
            _label = _label.reshape((1, A * feat_height * feat_width))
            bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
            bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
            label['%s_label_stride%d'%(prefix, stride)] = _label
            label['%s_bbox_target_stride%d'%(prefix,stride)] = bbox_target
            label['%s_bbox_weight_stride%d'%(prefix,stride)] = bbox_weight
            
            gt_box = np.zeros((1,2000,4),dtype=np.float32)
            if gt_boxes.size > 0:
                gt_box[:,:gt_boxes.shape[0],:] = gt_boxes.copy()
            label['gt_box_stride%d' % stride] = gt_box
            
        return label