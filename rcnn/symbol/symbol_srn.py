import mxnet as mx
from rcnn.config import cfg
from rcnn.PY_OP import focal_loss, filtered_anchor, adjusted_anchor

def conv_act_layer(from_layer, name, kernel, pad, stride, act_type, dilate=(1,1), num_filter=256, bias_wd_mult=0.0):
    if cfg.use_init:
        weight = mx.sym.Variable(name="{}_weight".format(name), init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
        bias = mx.sym.Variable(name="{}_bias".format(name), init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
        conv = mx.sym.Convolution(data=from_layer, kernel=kernel, pad=pad, dilate=dilate, stride=stride, num_filter=num_filter, name="{}".format(name), weight=weight, bias=bias)
    else:
        conv = mx.sym.Convolution(data=from_layer, kernel=kernel, pad=pad, dilate=dilate, stride=stride, num_filter=num_filter, name="{}".format(name)) 
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    relu = mx.sym.Activation(data=bn, act_type=act_type, name="{}_{}".format(name,act_type))
    return relu

def upsampling(data, name):
    if cfg.use_deconv:
        net = mx.sym.Deconvolution(data=data, num_filter=256, kernel=(4,4), stride=(2, 2), pad=(1,1), name="{}_deconv".format(name))
    else:
        net = mx.sym.UpSampling(data, scale=2, sample_type='nearest', workspace=512, name=name, num_args=1)
    return net

def multi_layer_feature(sym):
    input_size = cfg.image_size
    all_layers = sym.get_internals()
    _, out_shapes, _ = all_layers.infer_shape(data=(1, 3, input_size, input_size))
    outputs = all_layers.list_outputs()
    stride2name = {}
    stride2shape = {}
    stride2layer = {}
    for i in range(len(outputs)):
        name = outputs[i]
        shape = out_shapes[i]
        if not name.endswith('_output'):
            continue
        if len(shape)!=4:
            continue
        stride = input_size//shape[-1]
        stride2name[stride] = name
        stride2shape[stride] = shape
        stride2layer[stride] = all_layers[name]    
    strides = sorted(stride2name.keys())
    for stride in strides:
        print("stride", stride, stride2name[stride], stride2shape[stride])
    
    c2 = stride2layer[4]
    c3 = stride2layer[8]
    c4 = stride2layer[16]
    c5 = stride2layer[32]
    c6 = conv_act_layer(c5, 'c6', kernel=(3,3), pad=(1,1), stride=(2,2), act_type='relu', num_filter=1024, bias_wd_mult=1.0)
    c7 = conv_act_layer(c6, 'c7', kernel=(3,3), pad=(1,1), stride=(2,2), act_type='relu', num_filter=256, bias_wd_mult=1.0)    
    
    c2_r = conv_act_layer(c2, 'c2_r', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c3_r = conv_act_layer(c3, 'c3_r', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c4_r = conv_act_layer(c4, 'c4_r', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c5_r = conv_act_layer(c5, 'c5_r', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c6_r = conv_act_layer(c6, 'c6_r', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c7_r = conv_act_layer(c7, 'c7_r', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    
    c5_lateral = conv_act_layer(c5, 'c5_lateral', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    p5 = conv_act_layer(c5_lateral, 'p5', kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    p6 = conv_act_layer(p5, 'p6', kernel=(3,3), pad=(1,1), stride=(2,2), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    p7 = conv_act_layer(p6, 'p7', kernel=(3,3), pad=(1,1), stride=(2,2), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    
    c4_lateral = conv_act_layer(c4, 'c4_lateral', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c5_up = upsampling(c5_lateral, 'c5_upsampling')
    sum_4 = c5_up + c4_lateral
    p4 = conv_act_layer(sum_4, 'p4', kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    
    c3_lateral = conv_act_layer(c3, 'c3_lateral', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c4_up = upsampling(sum_4, 'c4_upsampling')
    sum_3 = c4_up + c3_lateral
    p3 = conv_act_layer(sum_3, 'p3', kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    
    c2_lateral = conv_act_layer(c2, 'c2_lateral', kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    c3_up = upsampling(sum_3, 'c3_upsampling')
    sum_2 = c3_up + c2_lateral
    p2 = conv_act_layer(sum_2, 'p2', kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    
    return [c2_r, c3_r, c4_r, c5_r, c6_r, c7_r], [p2, p3, p4, p5, p6, p7]

def rfe(name, layer):
    branch = []
    #branch-one
    branch1 = conv_act_layer(layer, '{}_rfe_branch1_1'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch1 = conv_act_layer(branch1, '{}_rfe_branch1_2'.format(name), kernel=(1,3), pad=(0,1), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch1 = conv_act_layer(branch1, '{}_rfe_branch1_3'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch.append(branch1)
    
    #branch-two
    branch2 = conv_act_layer(layer, '{}_rfe_branch2_1'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch2 = conv_act_layer(branch2, '{}_rfe_branch2_2'.format(name), kernel=(1,5), pad=(0,2), stride=(1,1), act_type='relu',
    num_filter=64, bias_wd_mult=1.0)
    branch2 = conv_act_layer(branch2, '{}_rfe_branch2_3'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch.append(branch2)
    
    #branch-three
    branch3 = conv_act_layer(layer, '{}_rfe_branch3_1'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch3 = conv_act_layer(branch3, '{}_rfe_branch3_2'.format(name), kernel=(3,1), pad=(1,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch3 = conv_act_layer(branch3, '{}_rfe_branch3_3'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch.append(branch3)
    
    #branch-four
    branch4 = conv_act_layer(layer, '{}_rfe_branch4_1'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch4 = conv_act_layer(branch4, '{}_rfe_branch4_2'.format(name), kernel=(5,1), pad=(2,0), stride=(1,1), act_type='relu',
    num_filter=64, bias_wd_mult=1.0)
    branch4 = conv_act_layer(branch4, '{}_rfe_branch4_3'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=64, bias_wd_mult=1.0)
    branch.append(branch4)
    
    #concat
    out = mx.sym.Concat(*branch, dim=1, name='{}_rfe'.format(name))
    out = conv_act_layer(out, '{}_rfe_out'.format(name), kernel=(1,1), pad=(0,0), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
    out = out + layer
    
    return out

def rpn(names, layers):
    net = []
    for name,layer in zip(names,layers):
        #first conv3X3
        layer1 = conv_act_layer(layer, '{}_rpn_first_conv3X3'.format(name), kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
        #RFE
        layer2 = rfe(name, layer1)
        #second conv3X3
        layer3 = conv_act_layer(layer2, '{}_rpn_second_conv3X3'.format(name), kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu', num_filter=256, bias_wd_mult=1.0)
        net.append(layer3)

    return net
        
def conv_head(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), bias_wd_mult=0.0, shared_weight=None, shared_bias = None):
    if shared_weight is None:
        if cfg.use_init:
            weight = mx.symbol.Variable(name="{}_weight".format(name), init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
            bias = mx.symbol.Variable(name="{}_bias".format(name), init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
            conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name="{}".format(name), weight = weight, bias=bias)
        else:
            conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name="{}".format(name))
    else:
        conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name="{}".format(name), weight = shared_weight, bias=shared_bias)
  
    return conv       
        
def get_out(feat_c, feat_p, stride, prefix, shared_vars):
    net_group = []
    gt_box = mx.sym.Variable(name='gt_box_stride{}'.format(stride))
    label = mx.sym.Variable(name='face_label_stride{}'.format(stride))
    bbox_target = mx.sym.Variable(name='face_bbox_target_stride{}'.format(stride))
    bbox_weight = mx.sym.Variable(name='face_bbox_weight_stride{}'.format(stride))
    
    
    valid_count = mx.sym.sum(label>0)
    valid_count = valid_count + 0.001 #avoid zero
    
    if stride<32:
        #feature C for cls layer
        c_rpn_cls_score = conv_head(feat_c, 'c%d_r_rpn_cls_score_stride%d'%(prefix,stride), 2*cfg.num_anchors, shared_weight=shared_vars[0], shared_bias=shared_vars[1])
        c_rpn_cls_score_reshape = mx.sym.Reshape(data=c_rpn_cls_score,
                                                  shape=(0, 2, -1),
                                                  name="c%d_r_rpn_cls_score_reshape_stride%d" % (prefix,stride))
        c_rpn_cls_prob = mx.sym.Custom(op_type='FocalLoss',
                                         alpha=cfg.alpha,
                                         gamma=cfg.gamma, 
                                         normalization=True,
                                         data=c_rpn_cls_score_reshape,
                                         labels=label)
        net_group.append(c_rpn_cls_prob)
        net_group.append(mx.sym.BlockGrad(label))
    else:
        #feature C for loc layer
        c_rpn_bbox_pred = conv_head(feat_c, 'c%d_r_rpn_bbox_pred_stride%d'%(prefix,stride), 4*cfg.num_anchors)
        c_rpn_bbox_pred_reshape = mx.sym.Reshape(data=c_rpn_bbox_pred,
                                                    shape=(0, 0, -1),
                                                    name="c%d_r_rpn_bbox_pred_reshape_stride%d" % (prefix,stride))
        bbox_diff = c_rpn_bbox_pred_reshape-bbox_target
        bbox_diff = bbox_diff * bbox_weight
        c_rpn_bbox_loss_ = mx.symbol.smooth_l1(name='c%d_r_rpn_bbox_loss_stride%d_'%(prefix,stride), scalar=3.0, data=bbox_diff)
        c_rpn_bbox_loss_ = mx.symbol.broadcast_div(c_rpn_bbox_loss_, valid_count)
        c_rpn_bbox_loss = mx.sym.MakeLoss(name='c%d_r_rpn_bbox_loss_stride%d'%(prefix,stride), data=c_rpn_bbox_loss_, grad_scale=1.0)
        net_group.append(c_rpn_bbox_loss)
        net_group.append(mx.sym.BlockGrad(bbox_weight))
    
    #feature P for cls layer
    p_rpn_cls_score = conv_head(feat_p, 'p%d_rpn_cls_score_stride%d'%(prefix,stride), 2*cfg.num_anchors, shared_weight=shared_vars[0], shared_bias=shared_vars[1])
    p_rpn_cls_score_reshape = mx.sym.Reshape(data=p_rpn_cls_score,
                                            shape=(0, 2, -1),
                                            name="p%d_rpn_cls_score_reshape_stride%d" % (prefix,stride))
    if stride<32:
        #filtered anchor
        new_label = mx.sym.Custom(op_type='STC_filtered',
                                cls_score=c_rpn_cls_prob,
                                labels=label)
        p_rpn_cls_prob = mx.sym.Custom(op_type='FocalLoss',
                                 alpha=cfg.alpha,
                                 gamma=cfg.gamma, 
                                 normalization=True,
                                 data=p_rpn_cls_score_reshape,
                                 labels=new_label)
        net_group.append(p_rpn_cls_prob)
        net_group.append(mx.sym.BlockGrad(new_label))
    else:
        #adjusted anchor
        new_label, bbox_target, bbox_weight = mx.sym.Custom(op_type='STR_adjusted',
                                                            stride=stride,
                                                            labels=label,
                                                            gt_boxes=gt_box,
                                                            box_deltas=c_rpn_bbox_pred)
        valid_count = mx.sym.sum(new_label>0)
        valid_count = valid_count + 0.001
        #p_rpn_cls_prob = mx.sym.SoftmaxOutput(data=p_rpn_cls_score_reshape,
        #                                       label=new_label,
        #                                       multi_output=True,
        #                                       normalization='valid', use_ignore=True, ignore_label=-1,
        #                                       grad_scale = 1.0,
        #                                       name='p%d_rpn_cls_prob_stride%d'%(prefix,stride))

        p_rpn_cls_prob = mx.sym.Custom(op_type='FocalLoss',
                                 alpha=cfg.alpha,
                                 gamma=cfg.gamma, 
                                 normalization=True,
                                 data=p_rpn_cls_score_reshape,
                                 labels=new_label)
        net_group.append(p_rpn_cls_prob)
        net_group.append(mx.sym.BlockGrad(new_label))
        
    
    
    #feature P for loc layer
    p_rpn_bbox_pred = conv_head(feat_p, 'p%d_rpn_bbox_pred_stride%d'%(prefix,stride), 4*cfg.num_anchors)
    p_rpn_bbox_pred_reshape = mx.sym.Reshape(data=p_rpn_bbox_pred,
                                                shape=(0, 0, -1),
                                                name="p%d_rpn_bbox_pred_reshape_stride%d" % (prefix,stride))
    bbox_diff = p_rpn_bbox_pred_reshape-bbox_target
    bbox_diff = bbox_diff * bbox_weight
    p_rpn_bbox_loss_ = mx.symbol.smooth_l1(name='p%d_rpn_bbox_loss_stride%d_'%(prefix,stride), scalar=3.0, data=bbox_diff)
    p_rpn_bbox_loss_ = mx.symbol.broadcast_div(p_rpn_bbox_loss_, valid_count)
    p_rpn_bbox_loss = mx.sym.MakeLoss(name='p%d_rpn_bbox_loss_stride%d'%(prefix,stride), data=p_rpn_bbox_loss_, grad_scale=1.0)
    net_group.append(p_rpn_bbox_loss)
    net_group.append(mx.sym.BlockGrad(bbox_weight))    
    
    return net_group

def get_sym_train(sym):
    net_group = []
    layer_c, layer_p = multi_layer_feature(sym)
    
    #RPN
    name_c = ['c2_r', 'c3_r', 'c4_r', 'c5_r', 'c6_r', 'c7_r']
    name_p = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    rpn_c= rpn(name_c, layer_c)
    rpn_p = rpn(name_p, layer_p)
    
    #shared cls convolutional layers
    shared_vars = []
    _name = 'face_rpn_cls_score_share'
    shared_weight = mx.symbol.Variable(name="{}_weight".format(_name), init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    shared_bias = mx.symbol.Variable(name="{}_bias".format(_name), init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(0.0)})
    shared_vars.append(shared_weight)
    shared_vars.append(shared_bias)
    
    strides = cfg.rpn_feat_stride
    for i,stride in enumerate(strides):
        net = get_out(rpn_c[i], rpn_p[i], stride, i+2, shared_vars) 
        net_group += net
    return mx.sym.Group(net_group)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    