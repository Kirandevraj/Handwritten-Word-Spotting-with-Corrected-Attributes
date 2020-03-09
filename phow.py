# phow.py
from scipy import shape, dstack, sqrt, floor, array, mean, ones, vstack, hstack, ndarray
# from vlfeat import vl_imsmooth
from cyvlfeat.sift import dsift as vl_dsift
from sys import maxsize
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
def vl_rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def vl_imsmooth(img, sigma):
    return gaussian_filter(img, sigma)
def vl_phow(im,
            verbose=False,
            fast=True,
            sizes=[4, 6, 8, 10],
            step=2,
            color='rgb',
            floatdescriptors=False,
            magnif=6,
            windowsize=1.5,
            contrastthreshold=0.005):

    opts = Options(verbose, fast, sizes, step, color, floatdescriptors,
                   magnif, windowsize, contrastthreshold)
    dsiftOpts = DSiftOptions(opts)
    im = array(im, 'float32')
    imageSize = shape(im)
    if im.ndim == 3:
        if imageSize[2] != 3:   
            raise ValueError("Image data in unknown format/shape")
    if opts.color == 'gray':
        numChannels = 1
        if (im.ndim == 2):
            im = vl_rgb2gray(im)
    else:
        numChannels = 3
        if (im.ndim == 2):
            im = dstack([im, im, im])
        if opts.color == 'rgb':
            pass
        elif opts.color == 'opponent':
            mu = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
            alpha = 0.01
            im = dstack([mu,
                         (im[:, :, 0] - im[:, :, 1]) / sqrt(2) + alpha * mu,
                         (im[:, :, 0] + im[:, :, 1] - 2 * im[:, :, 2]) / sqrt(6) + alpha * mu])
        else:
            raise ValueError('Color option ' + str(opts.color) + ' not recognized')
    if opts.verbose:
        print('{0}: color space: {1}'.format('vl_phow', opts.color))
        print('{0}: image size: {1} x {2}'.format('vl_phow', imageSize[0], imageSize[1]))
        print('{0}: sizes: [{1}]'.format('vl_phow', opts.sizes))

    frames_all = []
    descrs_all = []
    for size_of_spatial_bins in opts.sizes:
        off = floor(3.0 / 2 * (max(opts.sizes) - size_of_spatial_bins)) + 1
        sigma = size_of_spatial_bins / float(opts.magnif)
        ims = vl_imsmooth(im, sigma)
        frames = []
        descrs = []
        for k in range(numChannels):
            size_of_spatial_bins = int(size_of_spatial_bins)
            f_temp, d_temp = vl_dsift(image=ims[:, :, k],
                                      step=dsiftOpts.step,
                                      size=size_of_spatial_bins,
                                      fast=dsiftOpts.fast,
                                      verbose=dsiftOpts.verbose,
                                      norm=dsiftOpts.norm,)
            frames.append(f_temp.T)
            descrs.append(d_temp.T)
        frames = array(frames)
        descrs = array(descrs)
        d_new_shape = [descrs.shape[0] * descrs.shape[1], descrs.shape[2]]
        descrs = descrs.reshape(d_new_shape)
        if (opts.color == 'gray') | (opts.color == 'opponent'):
            contrast = frames[0][2, :]
        elif opts.color == 'rgb':
            contrast = mean([frames[0][2, :], frames[1][2, :], frames[2][2, :]], 0)
        else:
            raise ValueError('Color option ' + str(opts.color) + ' not recognized')
        descrs = descrs[:, contrast > opts.contrastthreshold]
        frames = frames[0][:, contrast > opts.contrastthreshold]
        frames_temp = array(frames[0:3, :])
        padding = array(size_of_spatial_bins * ones(frames[0].shape))
        frames_to_add = vstack([frames_temp, padding])

        frames_all.append(vstack([frames_temp, padding]))
        descrs_all.append(array(descrs))


    frames_all = hstack(frames_all)
    descrs_all = hstack(descrs_all)
    return frames_all.T[:,:2], descrs_all.T


class Options(object):
    def __init__(self, verbose, fast, sizes, step, color,
                 floatdescriptors, magnif, windowsize,
                 contrastthreshold):
        self.verbose = verbose
        self.fast = fast
        if (type(sizes) is not ndarray) & (type(sizes) is not list):
            sizes = array([sizes])
        self.sizes = sizes
        self.step = step
        self.color = color
        self.floatdescriptors = floatdescriptors
        self.magnif = magnif
        self.windowsize = windowsize
        self.contrastthreshold = contrastthreshold


class DSiftOptions(object):
    def __init__(self, opts):
        self.norm = True
        self.windowsize = opts.windowsize
        self.verbose = opts.verbose
        self.fast = opts.fast
        self.floatdescriptors = opts.floatdescriptors
        self.step = opts.step

if __name__ == "__main__":
    from scipy.misc import imread 
    im = imread('image_0001.jpg')
    frames, descrs = vl_phow(array(im, 'float32') / 255.0) 