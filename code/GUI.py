#!/usr/bin/env python3

from sys import platform # OS handling
from time import sleep   # efficient waiting
from screen import *     # Screen, np, cv2, device, Edge
import tkinter as tk     # GUI components

class GUI(object):
    ''' A class for managing the GUI used for calibration and testing. '''
    def __init__(self):
        ''' Create the GUI instance. '''
        self._root = tk.Tk()
        self._root.config(bg='black')

        """
        # get cursor
        self.cursor_info = np.zeros(4, dtype=int)
        if platform.startswith('darwin'): # macOS/OSX
            import plistlib
            self._cursor = '@cursor/cursor.pdf'
            with open('cursor/info.plist', 'rb') as info:
                cursor_info = plistlib.load(info)
            self.cursor_info[0] = cursor_info['hotx'] # x_sub
            self.cursor_info[1] = cursor_info['hoty'] # y_sub
        else:
            raise Exception("Platform not supported, requires unimplemented "+\
                            "cursor format")
        '''
        elif platform.startswith('win'):  # windows
            self._cursor = '@cursor/cursor.cur'
        else:                             # unix
            self._cursor = '@cursor/cursor.xbm red'
        '''
        self.cursor_img = cv2.imread('cursor/cursor.png', cv2.IMREAD_UNCHANGED)
        y_add, x_add = np.array(self.cursor_img.shape[:2]) - \
                [self.cursor_info[1], self.cursor_info[0]]
        self.cursor_info[2] = x_add
        self.cursor_info[3] = y_add

        self._root.config(cursor=self._cursor)
        """

    def _fullscreen_gui(self):
        ''' Make the GUI full-screen, and bring into focus. '''
        self._root.deiconify()
        self._root.attributes('-fullscreen', True)
        self._canvas.focus_force() # bring into focus

    def _create_canvas(self):
        ''' Create the canvas element for display control. '''
        width, height = self.resolution = Screen.get_resolution()
        self._canvas = tk.Canvas(master=self._root, width=width, height=height,
                                 bg='black', highlightthickness=0,
                                 cursor='none')#self._cursor)

        # set up binds to enable user to quit
        self._canvas.bind('<Key>', self.key)
        self._canvas.bind('<Escape>', lambda e: exit_with_message())

        self._canvas.pack(fill=tk.BOTH, expand=True) # fill space available
        self._canvas_lines = []

    def _calibration_setup(self):
        ''' Perform relevant setup for the calibration. '''
        # ensure full-screen works
        self._root.overrideredirect(True)
        self._root.overrideredirect(False)

        self._create_canvas()

        # create a resizable window for displaying calibration results
        cv2.namedWindow('calibration', cv2.WINDOW_NORMAL)
        cv2.moveWindow('calibration', 0, 20)

    def calibrate(self, cam, downsample):
        ''' Calibrate the specified camera using the GUI as a known screen. '''
        print('Beginning Calibration')
        self._calibration_setup()

        while 'not calibrated':
            print('attempting to calibrate')
            self._fullscreen_gui()

            # take a background image
            self._canvas.config(bg='black')
            self._update_projection()
            bg_success, bg = cam.read()
            bg = bg.astype(np.int64) # allow for negatives in subtraction

            # take an image with the screen white
            self._canvas.config(bg='white')
            self._update_projection()
            frame_success, frame = cam.read()

            if not (frame_success or bg_success):
                print("Can't receive frame (stream end?)")
                return False, None

            try:
                image = frame - bg
                image[image < 0] = 0
                points = Screen.points_from_img(np.uint8(image), colour='W')
            except Exception as e:
                print('calibration failed:', e)
                continue

            screen = Screen(points, resolution=self.resolution,
                            downsample=downsample)

            # adjust the calibration to improve accuracy
            try:
                key = self._display_calibration(frame, screen)
                if key == ord('c'):
                    cv2.destroyWindow('calibration') # cleanup
                    return True, screen
                elif key == ord('q'):
                    exit_with_message('User quit application')
            except Exception as e:
                print('Adjusting failed. Retrying...')
                continue

            # NOT REACHED
            # TODO: exposure reduction to avoid over-exposure of white
            #  --> Presently not possible for Macbook webcam due to
            #       no API for adjusting camera settings
            """
            # PRESENTLY INCOHERENT - DO NOT USE
            # not sure if this is the right value to set
            # cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, False)
            exposure = cam.get(cv2.CAP_PROP_EXPOSURE)
            direct = exposure > 0
            available = cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if available:
                # TODO: determine direction correctly (-230?)
                direction = np.sign(cv2.GaussianBlur(screen.transform(
                        cam.read()[1]), (5,5), 0).max())
                while 'overexposed': # TODO direction parameter
                    if direct:
                        # exposure is in time units
                        exposure *= (1 + direction * 0.1)
                    else:
                        # exposure is a power (e.g. 2**(exposure) seconds)
                        exposure -= 1
                    cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
                    _, frame = cam.read()
                    img = cv2.GaussianBlur(screen.transform(frame),
                                           (5,5), 0)
                        if img.max() < 230: # TODO: set as range
                        # success
                        break
            """

    def _display_calibration(self, display_frame, screen):
        ''' Displays the detected screen location on the display_frame. '''
        self._root.withdraw()
        cv2.polylines(display_frame, np.int32([screen._points]), True,
                      (255,0,0), 2)
        cv2.imshow('calibration', display_frame)
        return cv2.waitKey(0)

    def draw_maze(self, laser_colour):
        ''' Draw a maze on the screen for testing purposes. '''
        self._fullscreen_gui()
        self._root.config(bg='black')
        width, height = self.resolution
        self._canvas.config(bg='#333333')

        self._lines = np.array([[0,0,4,0],
                                [0,1,0,5],
                                [0,5,4,5],
                                [1,0,1,4],
                                [1,4,3,4],
                                [2,1,2,3],
                                [2,3,4,3],
                                [3,0,3,2],
                                [4,1,4,5]])
        scale = 0.9 * min(width, height)/10
        offset = (np.array([width, height]) - (np.array([4,5]) * scale)) / 2
        offset = np.array(list(offset) * 2)
        self._lines = list(self._lines * scale + offset)
        self._maze_scale = scale

        self._maze_regions = {'start_':[0,1,0,1],
                              't0_1':[0,1,1,4],
                              't90_1':[0,1,4,5],
                              't0_2':[1,3,4,5],
                              't180_1':[3,4,3,5],
                              't0_3':[2,3,3,4],
                              't90_2':[1,2,3,4],
                              't0_4':[1,2,1,3],
                              't180_2':[1,3,0,1],
                              't0_5':[2,3,1,2],
                              't180_3':[2,4,2,3],
                              't0_6':[3,4,1,2],
                              'end_':[3,4,0,1]}
        offset = offset.reshape((2,2)).T.reshape(-1)
        for region in self._maze_regions:
            self._maze_regions[region] = \
                    list(np.array(self._maze_regions[region]) * scale + offset)

        if laser_colour == 'r':
            fill = 'green'
        else:
            fill = 'red'

        for line in self._lines:
            self._canvas.create_line(*line, width=5, fill=fill)
        self._update_projection()

        device.moveTo(20,20)
        self._prev = [20,20]

    def draw_line(self, mouse_pos, y=None):
        ''' Draw a line following the mouse position. (callback function) '''
        if not y is None:
            new = [mouse_pos, y] # [x,y]
        else:
            new = [mouse_pos.x, mouse_pos.y]
        self._canvas_lines.append(self._canvas.create_line(*(self._prev + new),
                width=3, fill='blue'))

        if len(self._canvas_lines) > 20:
            self._canvas.delete(self._canvas_lines.pop(0))

        self._prev = new
        self._root.update()

    def key(self, event):
        ''' Key-press callback: quit on 'q'. '''
        #print('key event:', event)
        if event.char == 'q':
            self._root.destroy()
            sleep(0.3)
            exit_with_message()

    def _update_projection(self, wait=0.5):
        ''' Wait for the projector to update. '''
        self._root.update()
        for i in range(5):
            sleep(wait/5)
            self._root.update()


def exit_with_message(msg='User quit application.', retval=0):
    ''' Prints message and exits with code retval. '''
    print(msg + ' Exiting...')
    exit(retval)
