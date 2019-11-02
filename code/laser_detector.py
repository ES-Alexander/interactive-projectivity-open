#!/usr/bin/env python3

from interaction_detector import * # np, cv2, InteractionDetector
from overrides import overrides

class LaserDetector(InteractionDetector):
    ''' A class for detecting laser-pointer interactions with a screen. '''
    def __init__(self, colour='g', state_size=4, measurement_size=2,
                 *kalman_args, kalman=True, **kalman_kwargs):
        ''' Creates an interaction detector for tracking a laser-pointer.

        'colour' is the colour of the laser-pointer being tracked.
            Options are 'g'/'green', 'r'/'red', 'p'/'purple', and 'b'/'blue'.

        The remaining arguments are passed to a cv2.KalmanFilter instance
            to compensate for processing delay between measurements, unless
            'kalman' is set to False, in which case no compensation is used.

        '''
        super().__init__(state_size, measurement_size, *kalman_args,
                         **kalman_kwargs)

        self.colour = colour.lower()
        # create a kernel to help look for a laser-pointer within an image.
        self._kernel = np.array([[-4,-4,-4,-4,-4],
                                 [-4,-3, 2,-3,-4],
                                 [-4, 2,68, 2,-4],
                                 [-4,-3, 2,-3,-4],
                                 [-4,-4,-4,-4,-4]])
        self._first = True

    @overrides
    def detect_interaction(self, img):
        if len(img.shape) == 2 or img.shape[2] == 1:
            grey = img
        else:
            # convert to float image, and split into colour channels
            b,g,r = cv2.split(img / 255.0)

            if self.colour in ['g', 'green']:
                grey = g
            elif self.colour in ['r', 'red']:
                grey = r
            elif self.colour in ['p', 'purple']:
                grey = (r + b) / 2
            elif self.colour in ['b', 'blue']:
                grey = b
            else:
                # colour not supported
                grey = cv2.COLOUR_BGR2GRAY(img)

        # attempt to detect laser-pointer shaped objects
        filtered = cv2.filter2D(np.array(grey, dtype=np.float32), -1,
                                self._kernel)
        # filter out previous to remove DC errors
        if not self._first:
            adjusted = filtered - self._prev
            adjusted[adjusted < 0] = 0
        self._prev = filtered
        if self._first:
            self._first = False
            return
        blur = cv2.GaussianBlur(adjusted, (9,9), 0)

        # find the strongest point of detection in the image
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
        if maxVal > 0.3:
            detection = True
            measurement = maxLoc
        else:
            detection = False
            measurement = None
        self._handle_detection(detection, measurement)
        return
