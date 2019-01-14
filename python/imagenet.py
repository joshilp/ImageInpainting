import gc
import datetime

import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

from libs.pconv_model import PConvUnet
from libs.util import random_mask

from sys import platform

#from IPython.display import display
#from pil import Image

#%load_ext autoreload
#%autoreload 2
plt.ioff()

# SETTINGS
"""
if platform == "linux" or platform == "linux2" or platform == "darwin":
     WEIGHT_PATH = "data/logs/50_weights_2018-11-14-03-48-39.h5" 
elif platform == "win32":
     WEIGHT_PATH = r"data\logs\50_weights_2018-11-14-03-48-39.h5"
     TRAIN_DIR = r"D:\ILSVRC\Data\CLS-LOC\traintest\\"
     TEST_DIR = r"D:\ILSVRC\Data\CLS-LOC\test\\"
     VAL_DIR = r"D:\ILSVRC\Data\CLS-LOC\val\\"

BATCH_SIZE = 4


class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

# Create training generator
train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

# Create testing generator
test_datagen = DataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data

# Show side by side
for i in range(len(ori)):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[i,:,:,:])
    axes[1].imshow(mask[i,:,:,:] * 1.)
    axes[2].imshow(ori[i,:,:,:])
    plt.show()


def plot_callback(model):
    """"""Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk""""""

    # Get samples & Display them
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i, :, :, :])
        axes[1].imshow(pred_img[i, :, :, :] * 1.)
        axes[2].imshow(ori[i, :, :, :])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')

        plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
        plt.close()

# Instantiate the model
model = PConvUnet(weight_filepath='data/logs/')
#model.load(r"C:\Users\MAFG\Documents\Github-Public\PConv-Keras\data\logs\50_weights_2018-06-01-16-41-43.h5")

# Run training for certain amount of epochs
model.fit(
    train_generator,
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=50,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='../data/logs/initial_training', write_graph=False)
    ]
)

"""
# Load weights from previous run
WEIGHT_PATH = ""
if platform == "linux" or platform == "linux2" or platform == "darwin":
     WEIGHT_PATH = "data/logs/50_weights_2018-11-14-03-48-39.h5"
elif platform == "win32":
     WEIGHT_PATH = r"data\logs\50_weights_2018-11-14-03-48-39.h5"
"""
model = PConvUnet(weight_filepath='data/logs/')
model.load(
    WEIGHT_PATH,
    train_bn=False,
    lr=0.00005
)

# Run training for certain amount of epochs
model.fit(
    train_generator,
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=20,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='../data/logs/fine_tuning', write_graph=False)
    ]
)
"""
# Load weights from previous run
model = PConvUnet(weight_filepath='data/logs/')
model.load(
    WEIGHT_PATH,
    train_bn=False,
    lr=0.00005
)

n = 0
for (masked, mask), ori in tqdm(test_generator):

    # Run predictions for this batch of images
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(masked[i, :, :, :])
        axes[1].imshow(pred_img[i, :, :, :] * 1.)
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[0].xaxis.set_major_formatter(NullFormatter())
        axes[0].yaxis.set_major_formatter(NullFormatter())
        axes[1].xaxis.set_major_formatter(NullFormatter())
        axes[1].yaxis.set_major_formatter(NullFormatter())

        plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
        plt.close()
        n += 1

    # Only create predictions for about 100 images
    if n > 100:
        break


