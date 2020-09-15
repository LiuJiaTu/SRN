from __future__ import print_function
import mxnet as mx
import numpy as np
from rcnn.config import cfg

class FilteredAnchorOperator(mx.operator.CustomOp):
    def __init__(self):
        super(FilteredAnchorOperator, self).__init__()
    
    def forward(self, is_train, req, in_data, out_data, aux):
        cls_score = in_data[0].asnumpy() #BS, 2, ANCHORS
        labels_raw = in_data[1].asnumpy() #BS, ANCHORS
        
        for ibatch in range(labels_raw.shape[0]):
            labels = labels_raw[ibatch]
            bg_score = cls_score[ibatch, 0, :]
            
            bg_inds = np.where(labels==0)[0]
            neg_scores = bg_score[bg_inds]
            filter_inds = np.where(neg_scores>=cfg.stc_theta)[0]
            ignore_inds = bg_inds[filter_inds]
            labels[ignore_inds] = -1
        
        self.assign(out_data[0], req[0], mx.nd.array(labels_raw))
                
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register('STC_filtered')
class FilteredAnchorProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(FilteredAnchorProp, self).__init__(need_top_grad=False)
        
    def list_arguments(self):
        return ['cls_score', 'labels']
    
    def list_outputs(self):
        return ['filtered_labels']
    
    def infer_shape(self, in_shape):
        out_shape = in_shape[1]
        return in_shape, [out_shape], []
    
    def create_operator(self, ctx, shapes, dtypes):
        return FilteredAnchorOperator()
    
    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
    
    
    
    
        