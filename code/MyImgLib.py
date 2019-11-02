#!/usr/bin/env python3

################################################################################
#                                                                              #
# Opencv/Matplotlib Image Module                                               #
#                                                                              #
# Author: ES Alexander                                                         #
# Date Created: 18 Apr 2019                                                    #
# Last Modified: 19 Aug 2019                                                   #
#                                                                              #
################################################################################

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sys import float_info
epsilon = float_info.epsilon

def imshow(img, converted=False, padding=False, show=False, **kwargs):
    ''' Add the specified image to the current figure.

    'converted' is a boolean flag to specify if the image and kwargs
        combination should already display the image correctly. If left as
        False, image colour is adjusted to appropriately display opencv images
        with matplotlib.
    'padding' is a boolean flag to disable the removal of surrounding
        whitespace. If left as False, surrounding whitespace is removed.
    'show' is a boolean flag to enable calling plt.show().
    'kwargs' is a set of key-word arguments to pass to matplotlib.pyplot.imshow.

    imshow(cv2.img, *bool, *bool, **kwargs) -> None

    '''
    plt.axis('off') # hide axes and axis markings

    if not padding:
        # remove surrounding whitespace
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0, hspace=0)

    if not converted:
        # perform colour-adjustment for plotting
        if (len(img.shape) < 3 or img.shape[2] == 1):
            # black and white/grayscale image
            kwargs['cmap'] = 'gray'
        elif img.shape[2] == 3:
            # BGR image (convert to RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img, **kwargs) # add the image to the current figure
    if show:
        plt.show()

def imsshow(images, padding={'left':0,'right':1,'bottom':0,'top':1,
                             'wspace':0,'hspace':0}, ss = [], **kwargs):
    ''' Display the given set of images.

    'padding' is a dictionary of surrounding whitespace for each subplot. If
        replaced it can only contain keywords 'left', 'right', 'bottom', 'top',
        'wspace', and 'hspace', which can be given values between 0 and 1.
        left + right <= 1, bottom + top <= 1, wspace <= 1, hspace <= 1.
    'ss' is a list specifier of subplot organisation. If left empty, subplots
        are determined as an approximate square root, with preference for
        the larger dimension in the horizontal. If specified but
        ss[0] * ss[1] < len(images), the ratio of ss[0], ss[1] is used to
        determine the relevant organisation. To specify only one value, set
        the other to -1.
    'kwargs' is a set of key-word arguments to pass to matplotlib.pyplot.imshow.

    imsshow(list[cv2.img], *dict, *list, **kwargs) -> None

    '''
    # determine sizing
    s = len(images)
    if ss:
        s1, s2 = ss # extract specified values

        # if only one specified
        if s1 < 0:
            s1 = np.ceil(s / s2)
        elif s2 < 0:
            s2 = np.ceil(s / s1)

        # check for invalidity
        if s1 * s2 < s:
            # invalid -> recalculate with same ratio (where possible)
            ratio = s / (s1 * s2)
            s1, s2 = np.ceil(np.array([abs(s1), abs(s2)]) * abs(ratio))
    else:
        s1 = np.floor(np.sqrt(s))
        s2 = np.ceil(s/s1)

    title = kwargs.pop('title', False)

    # plot the images in their subplot positions (colour-correcting if needed)
    for index, img in enumerate(images):
        plt.subplot(s1, s2, index+1)
        imshow(img, padding=True, **kwargs)

    if title:
        plt.title(title)

    plt.subplots_adjust(**padding) # adjust padding as specified
    plt.show()

def imadjust(img, low_in='min', high_in='max', low_out=0, high_out=255):
    ''' Returns the adjusted image (equivalent of Matlab imadjust).

    'low_in' is the lower bound of the input image, below which points are
        removed. If left as 'min' it is set to the minimum value in the
        provided image.
    'high_in' is the upper bound of the input image, above which points are
        removed. If left as 'max' it is set to the maximum value in the
        provided image.
    'low_out' is the lower bound being mapped to for output.
    'high_out' is the upper bound being mapped to for output.

    imadjust(np.arr[arr[int/float]], str/*float, str/*float, *float, *float)
            -> arr[arr[float32]]

    '''
    if low_in == 'min':
        low_in = img.min()
    if high_in == 'max':
        high_in = img.max()
    return ((np.maximum(np.minimum(np.float32(img), high_in), low_in) - low_in)\
            * (high_out - low_out) / (high_in - low_in + epsilon))\
            + low_out

def imshowpair(img1, img2, colour_channels='green-magenta', display=False,
               show=False):
    ''' Returns a difference comparison between img1 and img2.

    'img1' and 'img2' are greyscale/black and white images of equal shape.
    'colour_channels' is a string or vector describing which colour channel(s)
        to assign each image to.
        'green-magenta': equivalent to [2,1,2]
            ->  img1 && img2   ==>  White
                img1 && !img2  ==>  Green
                !img1 && img2  ==>  Magenta
        'red-cyan': equivalent to [2,2,1]
            ->  img1 && img2   ==>  White
                img1 && !img2  ==>  Red
                !img1 && img2  ==>  Cyan
        [B,G,R]: 3-element vector specifying which image to assign to the
            blue, green, and red channels. 0 -> blank, 1 -> img1, 2 -> img2.
    'display' is a boolean specifier to display the result on a plot.
    'show' is a boolean specifier to plt.show() the displayed result.
        Ignored if 'display' is False.

    imshowpair(cv2.img[1], cv2.img[1], *str/list[int(3)], *bool, *bool)
               -> cv2.img[3]

    '''
    if colour_channels == 'green-magenta':
        colour_channels = [2,1,2]
    elif colour_channels == 'red-cyan':
        colour_channels = [2,2,1]

    blank = np.zeros(img1.shape)
    images = [blank, np.uint8(img1), np.uint8(img2)]
    result = cv2.merge([images[colour_channels[0]], images[colour_channels[1]],
                        images[colour_channels[2]]])
    if display:
        imshow(result, show=show)
    return result

def imfill(grey):
    ''' Returns 'grey' with all holes filled in.

    Fills dark regions (holes) within light regions.

    'grey' is a single-channel greyscale or black and white image.

    imfill(cv2.img[1]) -> cv2.img[1]

    '''
    filled = np.array(grey, dtype=np.uint8)
    _, contours, hier = cv2.findContours(filled, cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED, hierarchy=hier,
                     maxLevel=1)

    return filled

def skeletonise(img):
    r''' Morphological skeletonisation.

    Algorithm modified from:
        http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-
        python.html

    '''
    size = np.size(img)
    skel = np.zeros(img.shape, dtype=np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while 'not done':
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,
                              cv2.subtract(img, cv2.dilate(eroded,element)))
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            return skel # done

