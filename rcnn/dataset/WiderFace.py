import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
from PIL import Image    
    
class widerface():
    def __init__(self, dataset, image_set, dataset_path):
        self.image_set = image_set
        self.dataset_path = dataset_path
        self.name = dataset + '_' + image_set
        self.fp_bbox = {}
        
        #load dataset labels
        label_path = os.path.join(dataset_path, image_set, 'label.txt') 
        for line in open(label_path, 'r'):
            line = line.strip()
            if line.startswith('#'):
                name = line[1:].strip()
                self.fp_bbox[name] = []
                continue
            assert name in self.fp_bbox
            self.fp_bbox[name].append(line)
        print('origin image size:', len(self.fp_bbox))
    
    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join('data','cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def gt_roidb(self):
        #load cache file
        cache_file = os.path.join(self.cache_path, '{}_gt_roidb.pkl'.format(self.name))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                roidb = pickle.load(file)
            print("{} gt roidb loaded from {}".format(self.name, cache_file))
            self.num_images = len(roidb)
            return roidb
        
        #if not cache file
        roidb = []
        box_num = 0
        max_num_boxes = 0
        for fp in self.fp_bbox:
            #load the Dataset(test) information
            if self.image_set=='test':
                image_path = os.path.join(self.dataset_path, self.dataset, 'images', fp)
                with open(image_path, 'rb') as f:
                    stream = f.read()
                stream = np.frombuffer(stream, dtype=np.uint8)
                roi = {
                    'image':image_path,
                    'stream':stream
                }
                roidb.append(roi)
                continue
            
            #load the Dataset(train or val) information
            ix = 0
            boxes = np.zeros((len(self.fp_bbox[fp]), 4), dtype=np.float)
            for line in self.fp_bbox[fp]:
                #boxes
                image_path = os.path.join(self.dataset_path, self.image_set, 'images', fp)
                imsize = Image.open(image_path).size
                values = [float(x) for x in line.split()]
                x1 = values[0]
                y1 = values[1]
                x2 = min(x1 + values[2], imsize[0])
                y2 = min(y1 + values[3], imsize[1])
                if x1>=x2 or y1>=y2:
                    continue
                boxes[ix, :] = np.array([x1, y1, x2, y2], dtype=np.float)
                box_num += 1
                ix += 1
            max_num_boxes = max(max_num_boxes, ix)
            
            if self.image_set=='train' and ix==0:
                continue
            boxes = boxes[:ix, :]
            
            #image coding
            with open(image_path, 'rb') as f:
                stream = f.read()
            stream = np.frombuffer(stream, dtype=np.uint8)
            roi = {
                'image':image_path,
                'stream':stream,
                'height':imsize[1],
                'width':imsize[0],
                'boxes':boxes,
                'flipped':False
            }
            roidb.append(roi)
        
        self.num_images = len(roidb)
        print('box num:', box_num)
        print('max num boxes:', max_num_boxes)
        
        #make cache file
        with open(cache_file, 'wb') as file:
            pickle.dump(roidb, file, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        
        return roidb
            

def load_gt_roidb(dataset, image_set, dataset_path):
    imdb = widerface(dataset, image_set, dataset_path)
    roidb = imdb.gt_roidb()
    print('roidb size:', len(roidb))
    
    return roidb

if __name__ == '__main__':
    import cv2
    imdb = widerface('widerface', 'train', '/home/liujiatu/jiatunet/data/widerface')
    roidb = imdb.gt_roidb()
    print(len(roidb))
    for x in range(10):
        im = cv2.imdecode(roidb[x]['stream'], cv2.IMREAD_COLOR)
        for i in range(roidb[x]['boxes'].shape[0]):
            box = roidb[x]['boxes'][i].astype(np.int)
            cv2.rectangle(im, (box[0],box[1]), (box[2],box[3]), (255,0,0), 1)
        cv2.imwrite('testimages/%d.jpg'%x, im)
        
