#!/usr/bin/env python3

import cv2
import numpy as np
from Edge2 import Edge
import pyautogui as device # used for screen resolution

class Screen(object):
    ''' A class for tracking a physical screen display. '''
    def __init__(self, image_points, resolution=None, downsample=4):
        ''' Creates a Screen object from 'image_points'.

        'points' is a list of points in clockwise order from the top left
            e.g. [[top_left], [top_right], [bottom_right], [bottom_left]].
        'resolution' is the resolution of the screen, used for mapping.
            If left as None, resolution is determined as the computer's
            resolution.
        'downsample' is the amount to downsample the screen's resolution by
            in the transform from image to screen. Defaults to 4 if not set.

        Constructor: Screen(np.arr[arr[int(x2)](x4)], [int,int])

        '''
        self.downsample = downsample
        self._points = np.array(image_points, dtype=np.float32)

        # automatically determine (downsampled) screen resolution
        if resolution is None:
            resolution = self.get_resolution(downsample)
        elif downsample is not None:
            resolution = resolution // downsample # replace w/ downsampled copy
        self.resolution = resolution

        width, height = np.array(resolution) - 1
        self._dst_points = np.array([[0, 0], [width, 0],
                                    [width, height], [0, height]],
                                    dtype=np.float32)
        self._inverse_transform = np.matrix(cv2.getPerspectiveTransform(
            self._dst_points, self._points))

    def transform(self, img, display=False):
        ''' Returns the transformation result, from sampling 'img' using
            the internal transform, which maps to a downsampled view of the
            screen.
        '''
        # transform already inverted, so no need for warpPerspective to invert
        transformed = cv2.warpPerspective(img, self._inverse_transform,
                tuple(self.resolution), flags=cv2.INTER_LINEAR+\
                cv2.WARP_FILL_OUTLIERS+cv2.WARP_INVERSE_MAP)

        if display:
            cv2.imshow('transformed', transformed)
            print('press any key to continue')
            cv2.waitKey(0)

        return transformed

    def recalibrate(self, measured_pts, expected_pts):
        ''' Adjusts the stored defining points and transform using the mapping
            between the measured and expected points.

            'measured_pts' should be from an image pre-transformed with the
                current screen transform.

        '''
        adjustment = cv2.getPerspectiveTransform(np.float32(expected_pts),
                                                 np.float32(measured_pts))
        self._inverse_transform *= adjustment
        self._points = self.transform_pts(self._dst_points,
                                          self._inverse_transform)

    @staticmethod
    def transform_pts(pts, M):
        ''' Returns 2D pts transformed by transformation matrix M.

        Screen.transform_pts(np.arr[[float(x2)](xN)],
                             np.matrix[[float(x3)](x3)])
                -> np.arr[[float(x2)](xN)]

        '''
        in_pts = np.ones((3, len(pts)))
        in_pts[:2] = pts.T
        out_pts = M * in_pts
        out_pts = (out_pts[:2] / out_pts[2]).T
        return out_pts

    @classmethod
    def points_from_img(cls, img, colour=None, display=False):
        ''' Returns the corner points for a detected screen in 'img'.

        'img' is an image as returned from cv2.imread.
        'colour' is the colour of the screen being detected ('R','G','B','W').
            If left as None, the screen is assumed to be white.
        'display' is a boolean flag specifying if the detected screen corners
            should be displayed on 'img'.

        cls.points_from_img(np.arr[arr[int](x3)], str)
                -> np.arr[arr[int(x2)](x4)]

        '''
        blur = cv2.GaussianBlur(img, (5,5), 0)
        if len(img.shape) == 3 and img.shape[2] == 4:
            blur = cv2.cvtColor(blur, cv2.COLOR_BGRA2BGR)

        if len(img.shape) < 3 or img.shape[2] == 1:
            grey = img
        elif colour is None or colour == 'W':
            grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        else:
            B,G,R = cv2.split(blur)
            if colour == 'R':
                grey = R
            elif colour == 'B':
                grey = B
            elif colour == 'G':
                grey = G
            else:
                raise Exception(('Invalid colour: '+ str(colour)))

        edges = cls._process(grey)
        lines = cls._get_screen_lines(edges)

        if display: # display detected edges
            edges2 = cv2.merge([edges]*3) # convert to colour image
            for line in lines:
                # add lines to image
                p1, p2 = np.array(line.reshape((2,2)), dtype=int)
                cv2.line(edges2, tuple(p1), tuple(p2), (255,0,0), 2)
            cv2.imshow('lines', edges2)
            if cv2.waitKey(0) == ord('q'): # wait for user to continue or quit
                exit()

        points = cls._get_screen_points(lines)

        if display and points: # display detected points and screen polygon
            cv2.polylines(blur,np.int32([points]),True,(0,0,255),2)
            for point in points.T:
                blur[point[0],point[1]] = [0,0,255]
            cv2.imshow('lines', blur)
            cv2.waitKey(1) # show to the display
        return points

    @classmethod
    def _process(cls, grey, canny_lower=0, canny_upper=100, canny_aperture=3):
        ''' Performs pre-processing on the provided greyscale image.

        cls._process(np.arr[arr[int]]) -> np.arr[arr[int]]

        '''
        # Otsu binarisation (threshold assuming bimodal intensity histogram)
        th, bw = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Canny edge detection
        edges = cv2.Canny(bw, canny_lower, canny_upper,
                          apertureSize=canny_aperture)
        return edges

    @staticmethod
    def _get_screen_lines(edges, col_thresh=5, dist_thresh=0):
        ''' Gets the bounding lines of the screen in the 'edges' image.

        'col_thresh' is a collinearity threshold (angle in degrees) for
            combining similar lines in the image.
        'dist_thresh' is a minimum distance threshold (in pixels) for combining
            similar lines in the image.

        Screen._get_screen_lines(np.arr[arr[int]], *float, *float)
            -> arr[arr[int(x2)](x4)]

        '''
        ymax, xmax = edges.shape[:2] # get height, width
        # detect lines in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50,
                                maxLineGap=50).reshape((-1,4))

        # scale lines to the image boundaries
        boundaries = [0,xmax,0,ymax]
        for ind, line in enumerate(lines):
            lines[ind] = Edge.scale_to_boundaries(line, boundaries)

        # combine lines that are too similar (assumed to be the same line)
        return Edge.reduce_lines(np.array(lines), col_thresh=col_thresh,
                                 dist_thresh=dist_thresh)

    @staticmethod
    def _get_screen_points(lines):
        ''' Returns the 4 corner points detected at 'lines' intersections.

        If lines intersect in fewer than 4 places, returns None.
        If lines intersect in more than 4 places, k-means clustering is used
            to attempt to determine the 4 corner points.

        Screen._get_screen_points(np.arr[arr[float(x4)]]) -> arr[arr[float(x2)]]

        '''
        ip = Edge.get_intersections(lines, positions=True)
        # extract points from itersections' (ip[0]) positions (ip[1:3])
        points = np.array(Edge.get_intersection_points(ip[1:3],ip[0]),
                          dtype=np.float32)

        # check if detected points are valid (>=4 --> 4, <4 --> invalid)
        if len(points) > 4:
            # use k-means clustering to find 4 most likely points
            criteria = (0, 10, 1.0)
            _, _, points=cv2.kmeans(points, 4, None, criteria, 10,
                                     cv2.KMEANS_RANDOM_CENTERS)
        elif len(points) < 4:
            raise Exception('Insufficient points to detect screen')

        # order points consistently
        X, Y = np.array(points.T)
        for i in range(4):
            x,y = X[i], Y[i]
            point = np.array([x,y])
            if x < X.mean():
                if y < Y.mean():
                    points[0] = point
                else:
                    points[3] = point
            else:
                if y < Y.mean():
                    points[1] = point
                else:
                    points[2] = point
        return points

    @staticmethod
    def get_resolution(downsample=1):
        ''' Returns the resolution of the screen - optionally downsampled. '''
        return np.array(device.size()) // downsample


if __name__ == '__main__':
    # test script for detecting a screen
    import tkinter as tk
    from time import sleep
    width, height = device.size()
    root = tk.Tk()
    root.geometry('{}x{}'.format(width, height))
    root.overrideredirect(True)  # allow true fullscreen
    root.overrideredirect(False) # restore normal shortcuts
    def fullscreen():
        #root.config(bg='white')
        root.update()
        root.deiconify()
        root.attributes('-fullscreen', True, '-topmost', True)
        root.focus_force() # bring into focus
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('cannot open camera')
        exit()
    try:
        while cv2.waitKey(1) != ord('c'):
            success, frame = cap.read()
            if success:
                cv2.imshow('frame', frame)
            else:
                print('failure')
        cv2.imwrite('proj.png', frame)
        #while cv2.waitKey(1) != ord('c'): pass

        while cv2.waitKey(0) & 0xFF != ord('q'):
            # capture frame-by-frame
            fullscreen()
            sleep(0.3)
            read_success, frame = cap.read()
            if not read_success:
                print("Can't receive frame (stream end?). Exiting...")
                break
            root.withdraw()

            # operate on the frame
            try:
                sc = Screen(Screen.points_from_img(frame, colour='W'))
                cv2.polylines(frame, np.int32([sc.points]),True,(0,0,255),2)
            except Exception:
                pass
            # display the result
            cv2.imshow('frame', frame)
    finally:
        # done, so release the capture
        cap.release()
        cv2.destroyAllWindows()
