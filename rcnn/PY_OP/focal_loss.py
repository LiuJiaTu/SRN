import mxnet as mx

class FocalLoss(mx.operator.CustomOp):
    def __init__(self, alpha, gamma, normalization):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalization = normalization
        
    def forward(self, is_train, req, in_data, out_data, aux):
        y = mx.nd.SoftmaxActivation(data=in_data[0], mode='channel')
        self.assign(out_data[0], req[0], y)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        label = mx.nd.reshape(in_data[1], (0, 1, -1))  #(bs,1,anchors)
        p = mx.nd.pick(out_data[0], label, axis=1, keepdims=True)
        
        u = mx.nd.power(1 - p, self.gamma - 1)*(1 - p - self.gamma * p * mx.nd.log(mx.nd.maximum(p, 1e-14)))
        a = (label > 0) * self.alpha + (label == 0) * (1 - self.alpha)
        gf = u * a
        
        #one-hot
        num_class = out_data[0].shape[1]
        label_mask = mx.nd.one_hot(mx.nd.reshape(label, (0, -1)), num_class, on_value=1, off_value=0)
        label_mask = mx.nd.transpose(label_mask, (0,2,1))
        grad = (out_data[0] - label_mask) * gf 
        grad *= (label >= 0)
        
        if self.normalization:
            grad /= max(1.0, mx.nd.sum(label > 0).asscalar())
        
        self.assign(in_grad[0], req[0], grad)
        self.assign(in_grad[1], req[1], 0)
        
@mx.operator.register("FocalLoss")
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, alpha=0.25, gamma=2.0, normalization=True):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.normalization = normalization
    
    def list_arguments(self):
        return ['data','labels']
    
    def list_outputs(self):
        return ['focal_loss']
    
    def infer_shape(self,in_shape):
        out_shape = in_shape[0]        
        return in_shape, [out_shape], []
    
    def create_operator(self,ctx, shapes, dtypes):
        return FocalLoss(self.alpha, self.gamma, self.normalization)
    