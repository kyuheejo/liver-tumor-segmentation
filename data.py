import numpy as np
import keras
import os
import cv2
import glob

class DataGen(keras.utils.Sequence):
    def __init__(self, path, batch_size = 8, image_size = 128, shuffle = True):
        self.path = path
        self.ids = glob.glob(self.path +'/images/*')
        self.batch_size = batch_size 
        self.image_size = image_size 
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __load__(self, id_name):
        image_path = os.path.join(self.path, 'images/', id_name) 
        mask_path = os.path.join(self.path, 'masks/', id_name) 
    
        image = cv2.imread(image_path, 0) # read as Grayscale image 
        image = cv2.resize(image, (self.image_size, self.image_size))[..., np.newaxis]
        
        mask = cv2.imread(mask_path, 0) # read as Grayscale image 
        mask = cv2.resize(mask, (self.image_size, self.image_size))[..., np.newaxis]
        
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index + 1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
            
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask = []
        
        for id_name in files_batch:
            id_name = id_name.replace(self.path + '/images/', '')
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
        
        image = np.array(image)
        mask = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.ids)
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))