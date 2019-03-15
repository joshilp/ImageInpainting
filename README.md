# Image-Inpainting

Attempts to automate the painting process done by VFX artists using machine learning, thus removing or reducing the bottleneck in the VFX pipeline. The idea behind this project is to modify TensorFlow's `retrain.py` by adding multiple activation layers before the final soft-max layer, in hopes of increasing accuracy and stability of the model.

## Languages Used

* Python

## Project Highlights

To see the a project overview and to dive into the math, check out [Applying Image Inpainting to Video Post Production](http://joshpatel.ca/image_inpainting) at my portfolio [JoshPatel.ca](http://joshpatel.ca/). However, to see the full details of the implementation, please have a read of the [original paper](https://arxiv.org/abs/1804.07723).

## Contributors

* Joshil Patel
* Di Wang
* Bryce Haley
* Kino Roy
* Hyukho Kwon

## Getting Started

### Dependencies
Install the following:
* Python
* Jupiter Notebooks
* Keras
* Tensorflow

### Training Data

* Download the Imagenet training data from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/download/imagenet_object_localization.tar.gz) (155 GB compressed):

## Creating Masks, Training, Predicting

Use the `.ipnyb` files in the original pull folder to create masks, train the network, and make predictions:

* Step1 - Mask Generation.ipynb
* Step2 - Partial Convolution Layer.ipynb
* Step3 - UNet Architecture.ipynb
* Step4 - Imagenet Training.ipynb
* Step5 - Prediction.ipynb
* Step6 - New Prediction.ipynb

To see predictions work:
* Pull weights from here:
```
git clone https://joshilp@bitbucket.org/joshilp/imageinpainting.git
```
* Find the `logs` folder and move it here: `ImageInpainting\original pull\data\logs`
* Create  a folder `footage`: `ImageInpainting\original pull\data\footage`
* In the `footage` folder, add sub folders `00, 01, 02,` etc.
* Within these folders, add three subfolders: `footage, mask, prediction`
* Add your footage and masks to the relevant folders
* Update `predicty.py` to reflect the folders you have created, and run it:
```
python predict.py
```

## Predictions Results

View the results of the predictions [here](https://vimeo.com/310712744).

## Reference

Keras implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions", https://arxiv.org/abs/1804.07723. 

### Authors from NVIDIA Corporation:
* Guilin Liu
* Fitsum A. Reda
* Kevin J. Shih
* Ting-Chun Wang
* Andrew Tao
* Bryan Catanzaro