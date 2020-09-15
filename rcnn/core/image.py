from __future__ import print_function
import numpy as np
import cv2
import os
import math
import sys
import random
from ..config import cfg

def brightness_aug(src):
    x = cfg.brightness_alpha
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        alpha = 1.0 + random.uniform(-x, x)
        src *= alpha
    return src

def contrast_aug(src):
    x = cfg.contrast_alpha
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        src *= alpha
        src += gray
    return src

def saturation_aug(src):
    x = cfg.saturation_alpha
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
    return src 

def hue_aug(src):
    x = cfg.hue_alpha
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        alpha = np.random.uniform(-x, x)
        img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + alpha
        src = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return src

def color_aug(img):
    augs = [brightness_aug, contrast_aug, saturation_aug, hue_aug]
    random.shuffle(augs)
    for aug in augs:
        img = aug(img)
    return img

def expand(im, roi_rec):
    height, width, depth = im.shape
    ratio = random.uniform(1,2)
    left = random.uniform(0, width*ratio-width)
    top = random.uniform(0, height*ratio-height)
    
    expand_im = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=im.dtype)
    expand_im[:,:,:] = cfg.pixel_means
    expand_im[int(top):int(top+height), int(left):int(left+width), :] = im
    
    new_rec = roi_rec.copy()
    boxes = roi_rec['boxes'].copy()
    boxes[:, :2] += (int(left), int(top))
    boxes[:, 2:] += (int(left), int(top))
    new_rec['boxes'] = boxes
    
    return expand_im, new_rec

def random_crop(im, roi_rec):
    new_rec = roi_rec.copy()
    height, width, _ = im.shape
    #select a patch size
    if random.randint(0,1):
        size = min(height, width)
    else:
        size = min(height, width)
        ratio = random.uniform(0.5,1.0)
        size = int(size * ratio)
    
    
    #crop a patch
    retry = 0
    LIMIT = 25
    while retry<LIMIT:
        left = random.uniform(0, width-size)
        top = random.uniform(0, height-size)
        crop_im = im[int(top):int(top+size), int(left):int(left+size), :]
        boxes = roi_rec['boxes'].copy()
        boxes[:, :2] -= (int(left), int(top))
        boxes[:, 2:] -= (int(left), int(top))
        
        valid = []
        valid_boxes = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            centerx = (box[0]+box[2])/2
            centery = (box[1]+box[3])/2
            if centerx<0 or centery<0 or centerx>=crop_im.shape[1] or centery>=crop_im.shape[0]:
                continue
            valid.append(i)
            valid_boxes.append(box)
        
        if len(valid)>0 or retry==LIMIT-1:
            im = crop_im
            new_rec['boxes'] = np.array(valid_boxes)
            break
        retry += 1
                    
    return im, new_rec

def percentCoords(image, boxes):
    height, width, channels = image.shape
    boxes[:, 0] /= width
    boxes[:, 2] /= width
    boxes[:, 1] /= height
    boxes[:, 3] /= height
    
    return boxes

def absoluteCoords(image, boxes):
    height, width, channels = image.shape
    boxes[:, 0] *= width
    boxes[:, 2] *= width
    boxes[:, 1] *= height
    boxes[:, 3] *= height
    
    return boxes

def flip_image(im, roi_rec):
    new_rec = roi_rec.copy()
    boxes = roi_rec['boxes'].copy()
    #ramdom flip
    prob = np.random.uniform(0, 1)
    if prob < 0.5 and boxes.size>0:
        im = im[:, ::-1, :]
        new_rec['flipped'] = True
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = im.shape[1] - oldx2 - 1
        boxes[:, 2] = im.shape[1] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
    
    #resize image
    if boxes.size>0:
        boxes = percentCoords(im, boxes)
        im = cv2.resize(im, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
        boxes = absoluteCoords(im, boxes)
    else:
        im = cv2.resize(im, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
    new_rec['boxes'] = boxes
    
    return im, new_rec

TMP_ID = -1
def get_crop_image(roidb):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    #roidb and each roi_rec can not be changed as it will be reused in next epoch
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        #load image
        roi_rec = roidb[i]
        if 'stream' in roi_rec:
            im = cv2.imdecode(roi_rec['stream'], cv2.IMREAD_COLOR)
        else:
            assert os.path.exists(roi_rec['image']), '{} does not exist'.format(roi_rec['image'])
            im = cv2.imread(roi_rec['image'])
            
        #color jittering
        im = im.astype(np.float32)
        im = color_aug(im)
        
        #expand image
        im, roi_exp= expand(im, roi_rec)
        
        #random crop patch
        im, roi_crop = random_crop(im, roi_exp) 
        
        #random flip
        im, new_rec = flip_image(im, roi_crop)
        
        global TMP_ID
        if TMP_ID>=0 and TMP_ID<20:          
          tim = im.copy().astype(np.uint8)
          for i in range(new_rec['boxes'].shape[0]):
            box = new_rec['boxes'][i].copy().astype(np.int)
            cv2.rectangle(tim, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            print('draw box:', box)
          filename = './trainimages/train%d.png' % TMP_ID
          print('write', filename)
          print(new_rec['image'])
          cv2.imwrite(filename, tim)
          TMP_ID+=1
        
        im_tensor = transform(im, cfg.pixel_means, cfg.pixel_stds, cfg.pixel_scale)
        processed_ims.append(im_tensor)
        processed_roidb.append(new_rec)
        
    return processed_ims, processed_roidb

def transform(im, pixel_means, pixel_stds, pixel_scale):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel(RGB), height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = (im[:, :, 2 - i]/pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    return im_tensor

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    elif ndim == 5:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3], :tensor.shape[4]] = tensor
    else:
        print(tensor_list[0].shape)
        raise Exception('Sorry, unimplemented.')
    
    return all_tensor

if __name__ == '__main__':  
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    
    with open('/home/liujiatu/srn/data/cache/widerface_train_gt_roidb.pkl','rb') as f:
        roidb = pickle.load(f)
    print(len(roidb))
    get_crop_image(roidb)

