import mxnet as mx
import os
import argparse
import pprint
from rcnn.config import cfg
from rcnn.logger import logger
from rcnn.dataset.WiderFace import load_gt_roidb
from rcnn.symbol.symbol_srn import get_sym_train
from rcnn.core.loader import CropLoader
from rcnn.symbol import  metric


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRN")
    parser.add_argument('--dataset', help='dataset name', default=cfg.dataset, type=str)
    parser.add_argument('--image_set', help='image_set name', default=cfg.image_set, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=cfg.dataset_path, type=str)
    parser.add_argument('--frequent', help='frequency of logging', default=cfg.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=cfg.kvstore, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--shuffle', help='random shuffle images for train', default=cfg.shuffle, type=bool)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=cfg.pretrained, type=str)
    parser.add_argument('--prefix', help='save model', default=cfg.prefix, type=str)
    parser.add_argument('--lr', help='base learning rate', default=cfg.lr, type=float)
    parser.add_argument('--lr_step', help='learning rate steps(in epoch)', default=cfg.lr_step, type=str)
    
    args = parser.parse_args()
    
    return args

def train_net(args, ctx):
    #print config
    logger.info(pprint.pformat(cfg))
    
    #load dataset for training
    roidb = load_gt_roidb(args.dataset, args.image_set, args.dataset_path)
    
    #load symbol and params
    logger.info('loading %s,%d' %(args.pretrained, 0))
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, 0)
    sym = get_sym_train(sym)

    #get anchor sample for training
    batch_size = cfg.batch_size * len(ctx)
    train_data = CropLoader(sym, roidb, batch_size, args.shuffle, ctx, args.work_load_list)
    
    #create Module
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = mx.mod.Module(sym, data_names=data_names, label_names=label_names, logger=logger, context=ctx, work_load_list=args.work_load_list)
    
    #metric
    eval_metrics = mx.metric.CompositeEvalMetric()
    mid = 2
    for m in range(len(cfg.rpn_feat_stride)):
        stride = cfg.rpn_feat_stride[m]
        _metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid+1, name='RPNAcc_s%s'%stride)
        eval_metrics.add(_metric)
        mid+=2
        _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid+1, name='RPNL1Loss_s%s'%stride)
        eval_metrics.add(_metric)
        mid+=4
        
    #learning rate steps
    lr_epoch = [int(epoch) for epoch in args.lr_step.split(',')]
    iters = int(len(roidb)/batch_size)
    lr_iters = [epoch * iters for epoch in lr_epoch]
    lr_steps = []
    for li in lr_iters:
        lr_steps.append((li, 0.1))
    logger.info('lr %f lr_epoch %s lr_steps %s' % (args.lr, lr_epoch, lr_steps))
    
    #optimizer
    opt = mx.optimizer.SGD(learning_rate=args.lr, momentum=0.9, wd=0.0001, rescale_grad=1.0/len(ctx), clip_gradient=None)
    initializer = mx.init.Xavier()
    
    def save_model(epoch):
        arg, aux = mod.get_params()
        all_layers = mod.symbol.get_internals()
        outs = []
        num_anchors = cfg.num_anchors
        
        for i,stride in enumerate(cfg.rpn_feat_stride):
            _name = 'c%d_r_rpn_cls_score_stride%d_output'%(i+2, stride)
            c_rpn_cls_score = all_layers[_name]
            c_rpn_cls_score_reshape = mx.sym.Reshape(data=c_rpn_cls_score,
                                                    shape=(0, 2, -1, 0),
                                                    name="c%d_r_rpn_cls_score_reshape_stride%d"%(i+2,stride) )
            c_rpn_cls_prob = mx.symbol.SoftmaxActivation(data=c_rpn_cls_score_reshape,
                                                        mode="channel",
                                                        name="c%d_r_rpn_cls_prob_stride%d"%(i+2,stride) )
            c_rpn_cls_prob_reshape = mx.symbol.Reshape(data=c_rpn_cls_prob,
                                                      shape=(0, 2 * num_anchors, -1, 0),
                                                      name='c%d_r_rpn_cls_prob_reshape_stride%d'%(i+2,stride) )
            _name = 'c%d_r_rpn_bbox_pred_stride%d_output'%(i+2, stride)
            c_rpn_bbox_pred = all_layers[_name]
            outs.append(c_rpn_cls_prob_reshape)
            outs.append(c_rpn_bbox_pred)
            
            _name = 'p%d_rpn_cls_score_stride%d_output'%(i+2, stride)
            p_rpn_cls_score = all_layers[_name]
            p_rpn_cls_score_reshape = mx.sym.Reshape(data=p_rpn_cls_score,
                                                    shape=(0, 2, -1, 0),
                                                    name="p%d_rpn_cls_score_reshape_stride%d"%(i+2,stride) )
            p_rpn_cls_prob = mx.symbol.SoftmaxActivation(data=p_rpn_cls_score_reshape,
                                                        mode="channel",
                                                        name="p%d_rpn_cls_prob_stride%d"%(i+2,stride) )
            p_rpn_cls_prob_reshape = mx.symbol.Reshape(data=p_rpn_cls_prob,
                                                      shape=(0, 2 * num_anchors, -1, 0),
                                                      name='p%d_rpn_cls_prob_reshape_stride%d'%(i+2,stride) )
            _name = 'p%d_rpn_bbox_pred_stride%d_output'%(i+2, stride)
            p_rpn_bbox_pred = all_layers[_name]
            outs.append(p_rpn_cls_prob_reshape)
            outs.append(p_rpn_bbox_pred)
            

        _sym = mx.sym.Group(outs)
        mx.model.save_checkpoint(prefix, epoch, _sym, arg, aux)
    
    train_data = mx.io.PrefetchingIter(train_data)
    _cb = mx.callback.Speedometer(train_data.batch_size, frequent=args.frequent, auto_reset=False)
    global_step = [0]
    def _batch_callback(param):
        #global global_step
        _cb(param)
        global_step[0]+=1
        mbatch = global_step[0]
        for step in lr_steps:
            if mbatch==step[0]:
                opt.lr *= step[1]
                print('lr change to', opt.lr,' in batch', mbatch, file=sys.stderr)
                break
        if mbatch==lr_steps[-1][0]:
            print('saving final checkpoint', mbatch, file=sys.stderr)
            save_model(0)
            sys.exit(0)
    
    # train
    end_epoch = 1000
    epoch_end_callback = None  
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=_batch_callback, kvstore=args.kvstore,
            optimizer=opt,
            initializer = initializer,
            allow_missing=True,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=0, num_epoch=end_epoch)
    

def main():
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
        for x in range(len(cvd.split(','))):
            ctx.append(mx.gpu(x))
    if len(ctx)==0:
        ctx.append(mx.cpu(0))
    else:
        print("gpu num:%d" % len(ctx))
    
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    
    train_net(args, ctx)

if __name__=='__main__':
    main()