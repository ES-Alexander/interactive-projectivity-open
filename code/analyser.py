#!/usr/bin/env python3

import numpy as np
from edgy_lines.Edge2 import Edge
from os import path

class Attempt(object):
    ''' A class for storing a maze attempt. '''
    def __init__(self, points, dTs, lines, classes, maze_lines):
        ''' Construct a maze attempt using the collected data. '''
        self._points = points
        self._dTs = dTs
        self._lines = lines
        self._classes = classes
        self._maze_lines = maze_lines
        self._stats = None

    def analyse(self):
        ''' Analyse the internal data and update internal statistics. '''
        stats = {'t90':0, 't180':0, 't0':0, 'outside':0, 'total_time':0,
                 'failures':0}

        l = len(self._lines)
        intersections = np.abs(Edge.get_intersections(
                np.array(self._lines+self._maze_lines), j_min=l, i_max=l))

        for index, dT in enumerate(self._dTs):
            # can't account for intersections because don't know if return
            #   to correct region before continuing
            stats[self._classes[index]] += dT
            stats['total_time'] += dT

        stats['failures'] = intersections.sum() / 2
        stats['distance'] = Edge.get_path_length(self._points)

        self._stats = stats

    def get_stats(self, force_recompute=False):
        ''' Retrieve the dictionary of statistics for this attempt. '''
        if self._stats is None or force_recompute:
            self.analyse()
        return self._stats

class Analyser(object):
    ''' A class for detecting and analysing maze attempts in a test run. '''
    def __init__(self, points, maze_regions, maze_lines):
        ''' Construct the analyser and perform basic pre-processing. '''
        X, Y, t = np.array(points).T
        self._points = np.array([X,Y]).T
        self._maze_regions = maze_regions
        self._maze_lines = maze_lines
        self._classify_points()
        self._dTs = [*np.diff(t)]
        self._lines = Edge.get_path_lines(self._points).tolist()
        self._attempts = []
        self._split_attempts()
        self._stats = None

    def _classify_points(self):
        ''' Classifies points based on the maze region they exist within. '''
        self._point_classes = []
        for index, point in enumerate(self._points):
            classified = False
            for region in self._maze_regions:
                if not Edge.is_out_of_bounds(point, self._maze_regions[region]):
                    # point is in this region
                    self._point_classes.append(region.split('_')[0])
                    classified = True
                    break
            if not classified:
                self._point_classes.append('outside') # point not in maze

    def _split_attempts(self):
        ''' Split the stored data into classes.

        Can be performed multiple times, but only works for attempt data added
            since the last split, not if previously split data is modified.

        '''
        started = False
        ended = True
        for index, region in enumerate(self._point_classes):
            if not started and region == 'start':
                start = index
                started = True
                ended = False
            elif started and not ended and region == 'end':
                ended = True
            elif started and ended and region != 'end':
                self._attempts.append(Attempt(self._points[start:index],
                                              self._dTs[start:index],
                                              self._lines[start:index],
                                              self._point_classes[start:index],
                                              self._maze_lines))
                started = False
            if region in ['start', 'end']:
                self._point_classes[index] = 't90'
        print('recorded {} attempts'.format(len(self._attempts)))

    def analyse(self, force_recompute=False):
        ''' Analyse the stored attempts and store results internally. '''
        stats = []
        for attempt in self._attempts:
            stats.append(attempt.get_stats(force_recompute))
        self._stats = stats

    def get_stats(self, force_recompute=False):
        ''' Returns the list of attempt analysis statistics (dicts). '''
        if self._stats is None or force_recompute:
            self.analyse(force_recompute)
        return self._stats

    def save_stats(self, filename='stats'):
        ''' Save the internal statistics to the given filename +.csv. '''
        suffix=0
        while path.exists(filename+str(suffix)+'.csv'):
            suffix += 1
        filename = filename + str(suffix) + '.csv'

        with open(filename, 'w') as out:
            first = True
            for attempt_stats in self.get_stats():
                if first:
                    out.write(','.join(attempt_stats.keys()) + '\n')
                    first=False
                out.write(','.join(str(v) for v in attempt_stats.values()) \
                          + '\n')
        return filename






