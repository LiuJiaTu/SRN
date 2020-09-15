from __future__ import print_function
import mxnet as mx 
import numpy as np
import random
from mxnet.executor_manager import _split_input_slice

from rcnn.config import cfg
from .rpn import get_crop_batch, AnchorLoader
from .image import tensor_vstack

class CropLoader(mx.io.DataIter):
    def __init__(self, sym, roidb, batch_size, shuffle, ctx, work_load_list=None):
        super(CropLoader, self).__init__()
        
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        self.work_load_list = work_load_list
        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.data = None
        self.label = None
        self.data_name = ['data']
        self.label_name = []
        
        #get label_name
        names = ['face_label', 'face_bbox_target', 'face_bbox_weight']
        for stride in cfg.rpn_feat_stride:
            for n in names:
                name = "%s_stride%d"%(n,stride)
                self.label_name.append(name)
        for stride in [32, 64, 128]:
            name = "gt_box_stride%d" % stride
            self.label_name.append(name)
            
            
        #get every feature maps shape for rpn
        feat_shape_list = []
        for i,stride in enumerate(cfg.rpn_feat_stride):
            feat_sym = sym.get_internals()['p%d_rpn_cls_score_stride%d_output'%(i+2,stride)]
            _, out, _ = feat_sym.infer_shape(data=(1,3,cfg.image_size, cfg.image_size))
            feat_shape = [int(x) for x in out[0]]
            feat_shape_list.append(feat_shape)
        
        self.al = AnchorLoader(feat_shape_list)
        
        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()
    
    @property
    def provide_data(self):
        return [(k, v.shape) for k,v in zip(self.data_name, self.data)]
    
    @property
    def provide_label(self):
        return [(k, v.shape) for k,v in zip(self.label_name, self.label)]
    
    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
    
    def iter_next(self):
        return self.cur + self.batch_size <= self.size
    
    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(), provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration
    
    def getindex(self):
        return self.cur/self.batch_size
    
    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0
    
    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        assert cur_to==cur_from+self.batch_size
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        
        ctx = self.ctx
        work_load_list = self.work_load_list
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list)==len(ctx), 'Invalid settings for work load.'
        slices = _split_input_slice(self.batch_size, work_load_list)
        
        #a batch images after data augmentation
        data_list = []
        label_list = []
        for slice in slices:
            roi = [roidb[i] for i in range(slice.start, slice.stop)]
            data, label = get_crop_batch(roi)
            data_list += data
            label_list += label
        
        #get all labels
        for data,label in zip(data_list, label_list):
            face_label_dict = self.al.assign_anchor_fpn(label, prefix='face')
            for k in self.label_name:
                label[k] = face_label_dict[k]
        
        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])
            #print(key,all_data[key].shape)
            
        all_label = dict()
        for key in self.label_name:
            pad = 0 if key.startswith('bbox_') else -1
            all_label[key] = tensor_vstack([batch[key] for batch in label_list], pad=pad)
            #print(key,all_label[key].shape)
        
        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]
        
        
        
            