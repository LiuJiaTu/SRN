import mxnet as mx
import numpy as np
from ..config import cfg
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps,bbox_transform

class AdjustedAnchorOperator(mx.operator.CustomOp):
    def __init__(self, stride):
        super(AdjustedAnchorOperator, self).__init__()
        self.stride = int(stride)
        
        #generate base anchors
        sstride = str(self.stride)
        base_size = cfg.rpn_anchor_cfg[sstride]['BASE_SIZE']
        ratios = cfg.rpn_anchor_cfg[sstride]['RATIOS']
        scales = cfg.rpn_anchor_cfg[sstride]['SCALES']
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32), stride = self.stride)
        
        #total anchors
        feat_height = cfg.image_size//self.stride
        feat_width = cfg.image_size//self.stride 
        N = feat_height * feat_width
        A = base_anchors.shape[0]
        all_anchors = anchors_plane(feat_height, feat_width, self.stride, base_anchors)
        all_anchors = all_anchors.reshape((N * A, 4))
        
        self.anchors = all_anchors 

        
        
    def forward(self, is_train, req, in_data, out_data, aux):
        labels = in_data[0].asnumpy()    #BS, ANCHORS
        gt_boxes = in_data[1].asnumpy()  
        box_deltas = in_data[2].asnumpy()    #BS, 4*2, H, W   
        A = cfg.num_anchors
        height = box_deltas.shape[2]
        width = box_deltas.shape[3]
        
        labels_raw = np.zeros(labels.shape, dtype=np.float32)
        bbox_targets = np.zeros((box_deltas.shape[0], 4*A, height*width), dtype=np.float32)
        bbox_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        
        for ibatch in range(labels.shape[0]):
            label = labels[ibatch]
            label = label.reshape((A, -1)).transpose((1,0)).reshape((-1,))
            gt_box = gt_boxes[ibatch]
            box_delta = box_deltas[ibatch]
            
            #get valid gt_box
            for ix in range(gt_box.shape[0]):
                if gt_box[ix, 0]>=gt_box[ix, 2]:
                    break
            gt_box = gt_box[:ix, :]
            
            #get proposals
            box_delta = box_delta.transpose((1,2,0)).reshape((-1, 4))
            proposals = self.bbox_pred(self.anchors, box_delta)
            
            #generate cls labels
            new_labels = np.empty(label.shape, dtype=np.float32)
            new_labels.fill(-1)
            if gt_box.size > 0:
                overlaps = bbox_overlaps(proposals, gt_box.astype(np.float))
                argmax_overlaps = overlaps.argmax(axis=1)
                max_overlaps = overlaps[np.arange(proposals.shape[0]), argmax_overlaps]
                new_labels[max_overlaps < cfg.ss_neg_overlap] = 0
                new_labels[max_overlaps >= cfg.ss_pos_overlap] = 1
                #new_labels[label==1] = 1
            else:
                new_labels[:] = 0
            
            #generate bbox loc labels
            new_targets = np.zeros((proposals.shape[0], 4), dtype=np.float32)
            if gt_box.size > 0:
                new_targets[:,:] = bbox_transform(proposals, gt_box[argmax_overlaps, :])
            new_weights = np.zeros((proposals.shape[0], 4), dtype=np.float32)
            new_weights[new_labels==1, :] = 1.0
            
            #reshape
            new_labels = new_labels.reshape((-1, A)).transpose((1, 0)).reshape((-1,))
            new_targets = new_targets.reshape((-1, A*4)).transpose((1, 0))
            new_weights = new_weights.reshape((-1, A*4)).transpose((1, 0))
            
            labels_raw[ibatch] = new_labels
            bbox_targets[ibatch] = new_targets
            bbox_weights[ibatch] = new_weights
            
        for ind,val in enumerate([labels_raw, bbox_targets, bbox_weights]):
            val = mx.nd.array(val)
            self.assign(out_data[ind], req[ind], val)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)
    
    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
        Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
        :param boxes: !important [N 4]
        :param box_deltas: [N, 4 * num_classes]
        :return: [N 4 * num_classes]
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]    
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(np.minimum(dw,6)) * widths[:, np.newaxis]
        pred_h = np.exp(np.minimum(dh,6)) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1]>4:
            pred_boxes[:,4:] = box_deltas[:,4:]

        return pred_boxes

@mx.operator.register('STR_adjusted')
class AdjustedAnchorProp(mx.operator.CustomOpProp):
    def __init__(self, stride):
        #super(AdjustedAnchorProp, self).__init__(need_top_grad=False)
        super(AdjustedAnchorProp, self).__init__()
        self.stride = stride
        
    def list_arguments(self):
        return ['labels', 'gt_boxes', 'box_deltas']
    
    def list_outputs(self):
        return ['adjusted_labels', 'box_targets', 'box_weights']
    
    def infer_shape(self, in_shape):
        labels_shape = in_shape[0]
        shape = in_shape[2]
        box_targets_shape = (shape[0], shape[1], shape[2]*shape[3])
        box_weights_shape = box_targets_shape
        return in_shape, [labels_shape, box_targets_shape, box_weights_shape], []
    
    def create_operator(self, ctx, shapes, dtypes):
        return AdjustedAnchorOperator(self.stride)
        
    #def declare_backward_dependency(self, out_grad, in_data, out_data):
        #return []
        