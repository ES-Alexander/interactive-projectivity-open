#!/usr/bin/env python3

import numpy as np
import cv2
from abc import ABC as AbstractClass
from abc import abstractmethod
from time import time # for Kalman filtering

class InteractionDetector(AbstractClass):
    r''' An abstract interface for detecting interactions in an image.

    The Kalman filtering included in this class contains code adapted in part
    from `https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-
    using-kalman-filter/`, available under LGPLv3.0. It has been modified to
    track just position and velocity, changed from C++ to Python, and does not
    use the ball-tracker image processing.

    '''
    def __init__(self, *kalman_args, kalman=True, not_found_thresh=2,
                 **kalman_kwargs):
        ''' Creates an interaction detector instance.

        Kalman compensation (if used) assumes tracking a single point over x,y
        space, with velocity.

        'kalman' is a boolean specifier determining if Kalman compensation is
            used to predict the state of the interaction on demand, corrected
            for with measurements.
        'kalman_args' and 'kalman_kwargs' are arguments and keyword-arguments
            passed to cv2.KalmanFilter.
        'not_found_thresh' is an integer limit on how many subsequent
            predictions are allowed once detection has dropped out, generally
            to attempt to continue tracking in temporary losses of the
            interaction, due to occlusions and the like.

        '''
        if kalman:
            self.init_kalman(*kalman_args, **kalman_kwargs)
        self._found = False
        self._not_found_count = 0
        self._not_found_thresh = not_found_thresh
        self._kalman = kalman

    def init_kalman(self, state_size=4, measurement_size=2, *kalman_args,
                    **kalman_kwargs):
        ''' Creates and initialises a Kalman filter to predict the state of
            the interaction with consideration of its position and velocity.

        Used as compensation for delay, to track the interaction more closely
            than pure measurements.

        Arguments are passed to a cv2.KalmanFilter instance.

        '''
        kf = cv2.KalmanFilter(state_size, measurement_size, *kalman_args,
                              **kalman_kwargs)
        # Transition State Matrix (set dT at each processing step)
        # [ 1 0 dT 0  ]
        # [ 0 1 0  dT ]
        # [ 0 0 1  0  ]
        # [ 0 0 0  1  ]
        kf.transitionMatrix = np.eye(state_size, dtype=np.float32)

        # Measurement Matrix
        # [ 1 0 0 0 ]
        # [ 0 1 0 0 ]
        kf.measurementMatrix = np.zeros((measurement_size, state_size),
                                        dtype=np.float32)
        kf.measurementMatrix[[0,1],[0,1]] = np.float32(1.0)

        # Process Noise Covariance Matrix
        # position and speed error assumed uncorrelated
        # [ Ex 0  0    0    ] pixels
        # [ 0  Ey 0    0    ] pixels
        # [ 0  0  Ev_x 0    ] pixels/second
        # [ 0  0  0    Ev_y ] pixels/second
        kf.processNoiseCov = np.eye(state_size, dtype=np.float32)
        if state_size == 2 * measurement_size:
            # assume first half are position, second half velocity
            processNoise = [500]*measurement_size + [10]*measurement_size
            for i, E in enumerate(np.float32(processNoise)):
                kf.processNoiseCov[i,i] = E
        else:
            kf.processNoiseCov *= np.float32(1e-2)

        # Measurement Noise Covariance Matrix
        kf.measurementNoiseCov = np.eye(measurement_size,
                                        dtype=np.float32) * 1e-2 # pixel

        self._kf = kf
        self._correction_time = time()
        self._state_size = state_size

    @abstractmethod
    def detect_interaction(self, img):
        ''' Attempts to detect an interaction in the image, updating the
            internal state.

        To get the latest estimate for the interaction location, use
            self.predict().

        'img' is the image being processed.

        self.detect_interaction(np.arr[arr[int](x3)]) -> None

        '''
        pass

    def _update_time(self):
        ''' Updates transition matrix with the time since the last correction.

        Not used when Kalman filter compensation is off.

        By default assumes position and velocity tracking of a single point in
            2D space.

        '''
        self._kf.transitionMatrix[[0,1],[2,3]] = \
                np.float32(time() - self._correction_time)

    def _handle_detection(self, detected, measurement):
        ''' Handles a detection attempt, tracking if the target is found, and
            updating the kalman filter with valid measurements (if used).

        'detected' is a boolean specifying if the interaction was detected in
            the attempt being handled.
        'measurement' is the detected measurement, or not used.

        self._handle_detection(bool, np.array[float32]) -> None

        '''
        if not detected and self._found:
            self._not_found_count += 1
            if self._not_found_count > self._not_found_thresh:
                self._found = False
        elif detected:
            self._not_found_count = 0
            self._last = measurement
            if not self._found: # first detection
                self._found = True
                if self._kalman:
                    self._kf.errorCovPre = np.eye(self._state_size,
                                                  dtype=float) * 1.0 # px
                    self._kf.statePost = np.array([*measurement, 0, 0],
                                                  dtype=np.float32)
                    self._correction_time = time()
            elif self._kalman:
                self._correct(np.array(measurement, dtype=np.float32))
        # else not tracking

    def predict(self):
        ''' Return the currently expected state from the filter.
        If not currently found/tracking, returns None.
        If not using Kalman filtering, returns last detected measurement.

        self.predict() -> arr[int(x2)]/None

        '''
        if not self._found:
            return None

        if not self._kalman:
            return np.array(self._last)

        self._update_time()
        return self._kf.predict()

    def _correct(self, measurement):
        ''' Corrects the state of the Kalman Filter with measurement.

        'measurement' should be an array of 32-bit floats.

        '''
        self._update_time()            # update to the current point in time
        self._kf.correct(measurement)  # correct with the new measurement
        self._correction_time = time() # set now as the latest correction time

