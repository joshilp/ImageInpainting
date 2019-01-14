import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf

from copy import deepcopy
from PIL import Image
from libs.pconv_model import PConvUnet
from libs.util import random_mask
from libs.util import get_mask

BATCH_SIZE = 4

def plot_images(images, s=5):
    _, axes = plt.subplots(1, len(images), figsize=(s*len(images), s))
    if len(images) == 1:
        axes = [axes]
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.show()



# weights = r'data\logs\70_weights_2018-11-16-15-07-15.h5'
weights = r'data\logs\100_weights_2018-11-23-20-46-06.h5'
# weights = r'data\logs\157_weights_2018-12-03-14-14-46.h5'

with tf.device('/cpu:0'):
    model = PConvUnet(weight_filepath='data/logs/')
    model.load(weights, train_bn=False)



num = ['00','01','02','03','04','05','06','07']

for i in num:

	footage_number = i

	footage = 'data/footage/' + footage_number +'/footage/*.png'
	masks = 'data/footage/' + footage_number +'/mask/*.png'
	prediction = 'data/footage/' + footage_number + '/prediction'

	print(footage)

	img_lst = []
	mask_lst = []

	for filename in glob.glob(footage):
	    im = Image.open(filename)
	    img_lst.append(im)
	    
	for filename in glob.glob(masks):
	    mask_lst.append(filename)

	_, footage_axes = plt.subplots((int)(len(img_lst)/3)+1, 3, figsize=(15, 15))

	imgs, msks = [], []
	for im, mk, ax in zip(img_lst, mask_lst, footage_axes.flatten()):
	    im = np.array(im) / 255
	    mask = get_mask(mk)
	    im[mask==0] = 1
	    imgs.append(im)
	    msks.append(mask)
	    # ax.imshow(im)
	    
	# print("img_lst size:", len(img_lst), "| mask_lst size:", len(mask_lst))

	fig, pred_axes = plt.subplots((int)(len(img_lst)/3)+1, 3, figsize=(15, 15))

	counter = 0

	for im, mk, ax in zip(imgs, msks, pred_axes.flatten()):
	    pred = model.scan_predict((im, mk))
	    filename = prediction + '/prediction_' + i + '_%03d.png' % (counter)
	    plt.imsave(filename, pred) 
	    counter = counter +1




