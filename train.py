import os 
import sys 
import random
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint

# import custom modules 
from model import unet
from data import DataGen
from loss import dice_loss, bce_dice_loss, focal_tversky, binary_crossentropy

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed
os.environ['CUDA_VISIBLE_DEVICES']= '1'

def train(args):
    # prepare data
    gen = DataGen(args.train_path, batch_size = args.batch_size, image_size = args.image_size, shuffle = True)
    valid_gen = DataGen(args.valid_path, image_size = args.image_size, batch_size = args.batch_size)

    # define model
    model = unet()

    # define callbacks
    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    mc = ModelCheckpoint('best_model_unet.h5', monitor='val_loss', mode='min', verbose = 1, save_best_only=True)

    train_gen = DataGen(args.train_path, image_size = args.image_size, batch_size = args.batch_size)
    valid_gen = DataGen(args.valid_path, image_size = args.image_size, batch_size = args.batch_size)

    train_steps = len(glob.glob(args.train_path))//args.batch_size
    valid_steps = len(glob.glob(args.valid_path))//args.batch_size

    model_checkpoint = ModelCheckpoint('unet_v4_0911.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    history = model.fit_generator(train_gen, 
                                    validation_data = valid_gen, 
                                    steps_per_epoch = train_steps, 
                                    validation_steps = valid_steps, 
                                    epochs = 40, 
                                    callbacks = [model_checkpoint])

    os.makedirs("./model")

    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['my_iou_metric'][1:])
    plt.plot(history.history['val_my_iou_metric'][1:])
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.title('model IOU')
    plt.savefig('./result/model_iou', '')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.title('model loss')
    plt.savefig('./result/model_loss', '')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train LSTM')
    parser.add_argument('--image_size', default=256, type=int, help='size of image')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--train_path', default='./final_dataset/LITS_raw/train', type=str)
    parser.add_argument('--valid_path', default='./final_dataset/LITS_raw/validation')
    args = parser.parse_args()
    train(args)
