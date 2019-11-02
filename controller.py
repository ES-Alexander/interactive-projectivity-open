#!/usr/bin/env python3

from GUI import *                        # GUI for calibration/testing
                                         #   np, cv2, Edge, device, sleep
                                         #   exit_with_message
from laser_detector import LaserDetector # interaction detector
from threading import Thread, Lock       # semi-synchronous parallel processing
from queue import Queue                  # mouse position thread communication
from time import time                    # timing
from mss import mss                      # fast cross-platform screenshots
from MyImgLib import imadjust            # image value scaling 
from analyser import Analyser            # test mode analysis

# TODO:
#   - colour-map screen colours to camera colours
#   - set to constant exposure, lowered so that laser is visible on white
#   - argument parser
#   - pip library
#   - handle changing screen resolution
""" # cursor overlay not needed in laser/pen mode on google slides/powerpoint
#   - cursor overlay for windows/unix/macOS
#   - cursor overlay consistency - make it stay changed while program running
"""

class Controller(object):
    ''' A class for interfacing a screen with a laser pointer. '''

    COLOUR = {'r': 2, 'g': 1}

    def __init__(self, laser_colour='r', grayscale=False, light_ref=True,
                 compensation=True, user='', delay=0, testing=False):
        ''' Create a controller instance for the screen. '''
        self._cam = None
        self._colour = self.COLOUR[laser_colour]
        self._laser_colour = laser_colour
        self._detector = LaserDetector(laser_colour, kalman=compensation)
        self._GUI = None
        self._mouse_offset = np.array([10,10], dtype=np.float64)

        self._points = []

        self.grayscale = grayscale
        self.light_ref = light_ref

        if testing:
            filename = 'temp/stats_d' + str(delay)
            if user:
                filename += '_'+user
            if compensation:
                filename += '_c'
            self._filename = filename

        self._testing = testing

        self._delay = delay

        device.PAUSE = 0 # no need for safety delay

    def run(self, camera_id, downsample=4):
        ''' Link the specified camera to the screen and run test maze. '''
        self._queue = Queue()
        device_controller = Thread(name='device', target=self._move_mouse,
                                   daemon=True)
        self._mouse_gain = np.array([1,1], dtype=np.float64)
        self._res_changed = False

        try:
            connect_success = self._connect(camera_id)
            if not connect_success:
                exit_with_message("Connection Failure.")

            # start tkinter (it dies if opened after imshow in _calib_wait)
            self._GUI = GUI()

            # wait for user to confirm connection or quit
            if not self._calib_wait():
                exit_with_message('User quit application.')

            calibrate_success, self._screen = self._GUI.calibrate(self._cam,
                                                                  downsample)
            if not calibrate_success:
                exit_with_message('Calibration failure.')

            # set mouse gain so positions are over full screen
            self._mouse_gain *= self._screen.downsample

            # set boundaries to keep points inside
            self._mouse_offset /= 2
            maxs = self._GUI.resolution - self._mouse_offset
            mins = self._mouse_offset
            self._boundaries = [mins[0], maxs[0], mins[1], maxs[1]]
            self._mouse_offset[:] = [0,0]

            self._boundary_lines = Edge.get_rect_lines(self._boundaries)

            if self.light_ref:
                self._set_ref_lighting()

            if self._testing:
                self._GUI.draw_maze(self._laser_colour)
                self._points.append([*self._GUI._prev, time()])
            else:
                self._GUI._root.withdraw()
                self._points = [device.position()]

            device_controller.start()  # start mouse controller

            self._start_camera_handler()

            self._latest = 0
            self._last = time()
            if self._testing:
                self._GUI._root.after(self._delay, self._run_loop)
                self._GUI._root.mainloop()
            else:
                while True: self._run_loop()

        finally:
            if self._testing and len(self._points) > 2:
                #print(self._points)
                an = Analyser(self._points, self._GUI._maze_regions,
                         self._GUI._lines)
                print(min(an._dTs), max(an._dTs), np.mean(an._dTs))
                filename = an.save_stats(self._filename)
                print('SAVED STATS TO', filename)

            if not self._cam is None:
                self._cam.release() # release the camera
            try:
                self._GUI._root.destroy()
            except Exception: pass  # root already destroyed/never initialised
            cv2.destroyAllWindows() # stop displaying stuff
            device.mouseUp()        # release the mouse
            # device_controller and frame_thread daemons auto-destroyed

    def _start_camera_handler(self):
        ''' Starts the handler thread for the feedback camera. '''
        self._lock1 = Lock()
        self._lock2 = Lock()
        self._camera_handler = Thread(name='frame', target=self._handle_camera,
                                    daemon=True)
        self._lock1.acquire()       # start with main thread in control
        self._camera_handler.start() # start the camera handling thread

    def _handle_camera(self):
        ''' Captures and pre-processes latest image from the camera stream. '''
        while "running":
            self._wait_until_needed()
            # read the latest image from the camera
            read_success, frame = self._cam.read()
            if not read_success:
                raise Exception("Can't receive frame")

            # perform initial processing on the image
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tformed = self._screen.transform(frame)
            self._frame = cv2.GaussianBlur(tformed, (5,5), 0).astype(np.float32)

            self._inform_image_ready()

    def _wait_until_needed(self):
        ''' Wait for main to request the next image. '''
        self._lock2.acquire() # wait until previous image has been received
        self._lock1.acquire() # wait until next image is desired
        self._lock2.release() # inform camera is taking image

    def _inform_image_ready(self):
        ''' Inform main that next image is available. '''
        self._lock1.release() # inform next image is ready

    def _get_latest_image(self):
        ''' Ask camera handler for next image. '''
        self._lock1.release() # inform next image is desired
        self._lock2.acquire() # wait until camera is taking image

    def _wait_for_camera_image(self):
        ''' Wait until next image is available. '''
        self._lock1.acquire() # wait until next image is ready
        self._lock2.release() # inform image has been received

    def _run_loop(self, debug=False):
        ''' Internal loop for updating mouse position. '''
        # retrieve the latest image in a separate thread
        self._get_latest_image()
        # while generating a reference image in this one
        ref_img = self._get_ref_img()
        self._wait_for_camera_image()
        # combine latest image with its reference
        removed_bg = cv2.GaussianBlur(self._frame - ref_img, (5,5), 0)
        # imadjust (negative) min to 0
        process = imadjust(removed_bg, high_out=removed_bg.max())

        # only ever used manually
        if debug:
            cv2.imshow('new',new.astype(np.uint8))
            cv2.imshow('ref',imadjust(ref_img).astype(np.uint8))
            cv2.imshow('light',self._ref_light.astype(np.uint8))
            cv2.imshow('process',process.astype(np.uint8))
            print('n:',new.min(),new.max(),'r:',ref_img.min(),ref_img.max(),
                  'l:',self._ref_light.min(),self._ref_light.max(),
                  'p:',process.min(),process.max())

        # make and register a measurement
        self._detector.detect_interaction(process)
        self._predict() # estimate the current position

        if self._testing:
            # run this function again after self._delay ms
            self._GUI._root.after(self._delay, self._run_loop)
        # else loop run externally to avoid overloading call stack

    def _predict(self):
        ''' Makes a prediction - if valid moves mouse there. '''
        position = self._detector.predict()
        if position is not None:
            position = position.reshape(-1)[:2]
            # shift mouse position into valid range (scale to shift points from
            #   downsampled positions, and offset to avoid edges)
            mouse_pos = self._position_to_mouse(position)

            if mouse_pos is None:
                return

            # put the new position in the mouse-move queue
            self._queue.put(mouse_pos)
            if self._testing:
                # track the position for analysis
                self._points.append([*mouse_pos, time()])
                # draw the new position on the canvas
                self._GUI.draw_line(*mouse_pos)
            else:
                self._points = [mouse_pos]

    def _position_to_mouse(self, position):
        ''' Returns a valid mouse_position given the latest predicted position.

        Ensures points are within the desired boundary

        self._position_to_mouse(np.arr[float(x2)]) -> np.arr[float(x2)]

        '''
        mouse_pos = position * self._mouse_gain + self._mouse_offset
        if Edge.is_out_of_bounds(mouse_pos, self._boundaries):
            _, mouse_pos = Edge.get_path_interesection(
                [*mouse_pos, *self._points[-1][:2]], self._boundary_lines,
                position=True)
        return mouse_pos

    def _connect(self, camera_id):
        ''' Connect to the specified camera. '''
        first = True
        while first or not self._cam.isOpened():
            if first:
                # use expected camera id this time
                first = False
            else:
                print('Invalid Camera ID:', camera_id)
                camera_id = input('Camera ID: ')

            self._cam = cv2.VideoCapture(camera_id)

            if cv2.waitKey(1) == ord('q'):
                print('Application quit by user')
                return False

        return True

    def _calib_wait(self):
        ''' Wait for the user to begin the calibration. '''
        print('press c to begin calibration')
        while 'not started':
            key = cv2.waitKey(1)
            if key == ord('c'):
                break # waiting done
            elif key == ord('q'):
                return False # user quit

            read_success, frame = self._cam.read()
            if read_success:
                cv2.imshow('calibration', frame)
            else:
                print('failure')

        return True

    def _set_ref_lighting(self):
        ''' Set the reference image for lighting subtraction. '''
        self._GUI._fullscreen_gui()
        # make the screen black to see background lighting
        self._GUI._root.config(bg='black')
        self._GUI._update_projection()
        read_success, ref_img = self._cam.read()
        if read_success:
            self._ref_light = cv2.GaussianBlur(self._screen.transform(ref_img),
                                               (5,5), 0).astype(np.float32)
        else:
            raise Exception('Lighting reference image unable to be taken -'\
                            'check camera connection')

    def _get_ref_img(self):
        ''' Get the latest reference image using the screen and lighting. '''
        width, height = self._screen.resolution * self._screen.downsample
        monitor = {'top':0, 'left':0, 'width':width, 'height':height}
        with mss() as sct:
            #mouse_pos = device.position()
            img = np.array(sct.grab(monitor), dtype=np.float32)

        #self._add_cursor(img, *mouse_pos)

        # convert colour as appropriate
        if self.grayscale and not self.light_ref:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # resize with automatic antialiasing low pass filtering
        fx, fy = self._screen.resolution / img.shape[1::-1]
        """ # intended to deal with changing resolution - not working
        ifx, ify = 1/fx, 1/fy
        if not self._res_changed:
            downsample = self._screen.downsample
            if ifx != downsample or ify != downsample:
                # where is the 2 from???? TODO
                self._mouse_gain = np.array([ify, ifx]) #/ downsample
                self._res_changed = True
        """
        # only re-size if actually changing size
        if fx != 1 or fy != 1:
            img = cv2.resize(img, (0,0), fx=fx, fy=fx,
                             interpolation=cv2.INTER_AREA)
        if self.light_ref:
            img += self._ref_light
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _add_cursor(self, img, x, y):
        ''' Adds the cursor to the provided screenshot image. '''
        # crop cursor image to witihin the screen
        x_subi, y_subi, x_addi, y_addi = self._GUI.cursor_info
        x_sub = min(x_subi, x)
        y_sub = min(y_subi, y)
        ym, xm = img.shape[:2]
        xp, yp = xm - x, ym - y
        x_add = min(x_addi, xp)
        y_add = min(y_addi, yp)

        cursor_img = self._GUI.cursor_img[y_subi-y_sub:y_subi+y_add,
                                          x_subi-x_sub:x_subi+x_add]
        # add cursor to img using complementary alpha-channel information
        alpha = np.repeat((cursor_img[:,:,3] / 255.0).reshape(
            *cursor_img.shape[:2], 1), 3, axis=2)

        img[y-y_sub:y+y_add, x-x_sub:x+x_add, :3] = alpha*cursor_img[:,:,:3] + \
                (1.0 - alpha) * img[y-y_sub:y+y_add, x-x_sub:x+x_add, :3]

    def _move_mouse(self):
        ''' Receives continual instructions from the main thread for where to
            move the mouse.
        '''
        while 'still useful':
            device.moveTo(*self._queue.get())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='control the mouse using a laser-pointer')
    parser.add_argument('-NC', '--noCompensation', action='store_true',
                        help='disable Kalman compensation')
    parser.add_argument('-i', '--cameraID', default=0, help='camera ID',
                        type=int)
    parser.add_argument('-c', '--colour', help='laser colour', default='g',
                        choices=['g', 'r'])
    parser.add_argument('-d', '--downsample', help='downsample screen by',
                        type=int, default=4)
    parser.add_argument('-NL', '--noLightRef', action='store_true',
                        help='disable removing background lighting')
    parser.add_argument('-t', '--testing', help='use test mode',
                        action='store_true')
    parser.add_argument('-u', '--user', help='the user testing the system',
                        default='me')
    parser.add_argument('-D', '--delay', default=0, type=int, help='ms delay'+\
                        ' added between processing frames while testing')
    args = parser.parse_args()

    C = Controller(args.colour, compensation=not args.noCompensation,
                   light_ref=not args.noLightRef, user=args.user,
                   delay=args.delay, testing=args.testing)
    C.run(args.cameraID, args.downsample)
