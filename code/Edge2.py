#!/usr/bin/env python3
#############################################
#                                           #
# Edge module, for lines and edges.         #
# Author: ES Alexander                      #
# Date: 6 Apr 2019                          #
# Last Modified: 14 Oct 2019                #
#                                           #
#############################################
import numpy as np
from sys import float_info
epsilon = float_info.epsilon

class Edge(object):
    ''' A class for edge extraction and processing. '''
    @staticmethod
    def is_point(line):
        ''' Returns true if the specified line is in fact a point.

        'line' is a line segment of form [x0,y0,x1,y1].

        Edge.is_point(np.arr[float(x4)]) -> bool

        '''
        x0,y0,x1,y1 = line
        return x0 == x1 and y0 == y1

    @staticmethod
    def get_line_points(lines, *indices):
        ''' Returns a vector of points found in lines[indices] as [X,Y].

        Points are maintained in the order they appear in the relevant lines.

        'lines' is a vector of line segments in form [X0,Y0,X1,Y1].
        'indices' is a vector of indices with max(indices) < len(lines).

        Edge.get_line_points(np.arr[[float(x4)]], *int) -> arr[[float(x2)]]

        '''
        return lines[np.array(indices)].reshape(-1,2)

    @staticmethod
    def get_path_lines(path, closed=False):
        ''' Returns a vector of line segments traced by path, as [X0,Y0,X1,Y1].

        'path' is a vector of points in form [X0,Y0].
        'closed' is a boolean specifying if the path is closed - if True,
            creates a line segment from the last to the first point in 'path'.

        Edge.get_path_lines(np.arr[[float(x2)]], *bool) -> arr[[float(x4)]]

        '''
        lines = np.roll(path.repeat(2, axis=0), 1, axis=0).reshape((-1,4))
        if not closed:
            lines = lines[1:]
        return lines

    @classmethod
    def get_rect_lines(cls, rect):
        ''' Returns the lines that make up the rectangle 'rect'.

        'rect' is of the form [xmin, xmax, ymin, ymax].

        cls.get_rect_lines(np.arr[float(x4)])

        '''
        xmin, xmax, ymin, ymax = rect
        points = np.array([[xmin, ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        return cls.get_path_lines(points, closed=True)

    @classmethod
    def get_path_interesection(cls, line, path_lines, position=False,
                               angle=False):
        ''' Returns True if line intersects with path_lines.

        'line' is of form [x0,y0,x1,y1].
        'path_lines' is of form [X0, Y0, X1, Y1].
        'position' is a boolean specifier determining if the position of a
            detected intersection point is returned.
        'angle' is a boolean specifier determining if the angle of a detected
            intersection between line and a path_line is returned.

        cls.get_path_interesection(np.arr[float(x4)], np.arr[[float(x4)]],
                                   *bool, *bool)

        '''
        for path_line in path_lines:
            output = cls.get_intersections(np.array([path_line, line]),
                                           dist_thresh=1,
                                           angles=angle, positions=position)
            intersection = output[0,0,1]
            if not intersection:
                continue
            if angle:
                i_angle = output[1,0,1]
                if position:
                    i_pos = output[2:,0,1]
                    return True, i_angle, i_pos
                return True, i_angle
            elif position:
                return True, output[1:,0,1]
            return True
        if angle and position:
            return False, None, None
        if angle or position:
            return False, None
        return False

    @staticmethod
    def get_path_length(path_points):
        ''' Returns the length of the path specified by path_points.

        'path_points' is an array of points, in form [[x0,x1,...,xn],...].

        Edge.get_path_length(np.arr[np.arr[float]]) -> float

        '''
        return np.sqrt((np.diff(path_points, axis=0) ** 2).sum(axis=1)).sum()

    @staticmethod
    def distsq(p0, p1):
        ''' Returns the squared Euclidian distance between p0 and p1.

        'p0' and 'p1' are points of form [x1,x2,...,xn], of same order.

        Edge.distsq(list[float], list[float]) -> float

        '''
        return sum((p1 - p0)**2)

    @classmethod
    def get_line(cls, points):
        ''' Returns a joined line segment covering the given points.

        Assumes the points are reasonably collinear and joins those furthest
            apart.

        'points' is an array of points in form [X,Y].

        cls.get_line(np.arr[[float(x2)]]) -> arr[float(x4)]

        '''
        num_points = len(points)
        max_dist = 0
        im = 0; jm = 0;
        for i in range(num_points):
            for j in range(i+1, num_points):
                new_dist = cls.distsq(points[i], points[j])
                if new_dist > max_dist:
                    max_dist = new_dist
                    im = i; jm = j
        return np.array([points[im], points[jm]]).reshape(4)

    @staticmethod
    def equal_lines(line0, line1):
        ''' Returns True if line0 and line1 are equivalent, else False.

        'line0' and 'line1' are line segments of form [x0,y0,x1,y1].

        Edge.equal_lines(np.arr[float(x4)], np.arr[float(x4)]) -> bool

        '''
        # break line segments into constituent points
        p0, p1 = line0.reshape((2,2))
        p0 = list(p0); p1 = list(p1)
        p2, p3 = line1.reshape((2,2))
        p2 = list(p2); p3 = list(p3)
        # compare in both directions
        return (p0 == p2 and p1 == p3) or (p0 == p3 and p1 == p2)

    @classmethod
    def add_new_lines(cls, lines, *extra_lines):
        ''' Returns lines with any additional extra_lines not already present.

        'lines' is a vector of line segments in form [X0,Y0,X1,Y1].
        'extra_lines' is an unspecified number of additional lines to add.

        cls.add_new_lines(np.arr[arr[float(x4)]], *arr[float(x4)])
                -> arr[arr[float(x4)]]

        '''
        new_lines = np.array(lines)
        for query_line in extra_lines:
            duplicate = False
            for existing_line in new_lines:
                if cls.equal_lines(existing_line, query_line):
                    duplicate = True
                    break
            if not duplicate:
                # genuine new line, add to list
                if new_lines.size > 0:
                    new_lines = np.r_[new_lines, [query_line]]
                else:
                    new_lines = np.array([query_line])
        return new_lines

    @staticmethod
    def duplicate_point(points):
        ''' Returns true if points contains multiples of one or more points.

        'points' is a list of points in form [X,Y].

        Edge.duplicate_point(list[list[float]]) -> bool

        '''
        return len(set(tuple(point) for point in points)) != len(points)

    @staticmethod
    def line_angle(line, degrees=True):
        ''' Returns the angle of 'line' with respect to the positive x-axis.

        Edge.line_angle(np.arr[float(x4)]) -> float

        '''
        x0,y0,x1,y1 = line
        angle = np.arctan2(y1-y0, x1-x0)
        if degrees:
            return np.degrees(angle)
        return angle

    @classmethod
    def angle_between_lines(cls, line0, line1):
        ''' Returns the angle (degrees) between the given lines.

        cls.angle_between_lines(np.arr[float(x4)], np.arr[float(x4)]) -> float

        '''
        return cls.line_angle(line1) - cls.line_angle(line0)

    @classmethod
    def dist_point_to_segment(cls, p0,p1,p2,positions=False):
        r''' Returns the closest distance from point p0 to segment p1-p2.

        'p0','p1','p2' are points of form [x,y].
        'positions' is a boolean specifier which, if True, appends the
            approximate point of intersection to the return value.

        Algorithm from:
        "https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line"

        cls.dist_point_to_segment(np.arr[float(x2)], arr[float(x2)],
                                  arr[float(x2)], *bool)
                -> float, *arr[float(x2)]

        '''
        d01sq = cls.distsq(p0,p1); d01 = np.sqrt(d01sq)
        d02sq = cls.distsq(p0,p2); d02 = np.sqrt(d02sq)
        d12sq = cls.distsq(p1,p2); d12 = np.sqrt(d12sq)

        if d01 == 0 or d02 == 0:
            # point on segment endpoint
            if positions:
                return 0, p0
            return 0
        elif d12 == 0:
            # segment is a point
            if positions:
                return d01, (p0 + p1)/2
            return d01
        u = np.round((d01sq + d12sq - d02sq) / (2 * d01 * d12), 10)
        if u <= 0:
            if positions:
                return d01, (p0 + p1)/2
            return d01
        elif d01 * u >= d12:
            if positions:
                return d02, (p0 + p2)/2
            return d02
        else:
            dist = d01 * np.sqrt(1 - u**2)
            if positions:
                pdiff = p2 - p1 + epsilon
                grad = pdiff[1]/pdiff[0]
                p4 = p1 + np.array([1,grad]) * dist / np.sqrt(grad**2 + 1)
                return dist, (p0 + p4)/2
            return dist

    @classmethod
    def min_dist_segments(cls, lines, position=False):
        ''' Returns the minimum distance  between the segments in 'lines'.

        'positions' is a boolean specifier which, if True, appends the
            approximate point of intersection to the return value.

        cls.min_dist_segments(np.arr[arr[float(x2)]], *bool)
                -> float, *arr[float(x2)]

        '''
        if position:
            output = cls.get_intersections(lines, positions=True)
            intersections = output[0]
            positions = output[1:]
        else:
            intersections = cls.get_intersections(lines)
        if np.abs(intersections).sum() == 2:
            # segments intersect
            if position:
                return 0, np.array([positions[0,0,1], positions[1,0,1]])
            return 0
        dist = np.Inf
        pos = None
        points = cls.get_line_points(lines, *range(len(lines)))
        point_triples = np.array([[0,2,3],[1,2,3],[2,0,1],[3,0,1]])
        for point in range(4):
            ps = points[point_triples[point]]
            if position:
                query_dist, query_pos = cls.dist_point_to_segment(*ps,
                        positions=True)
            else:
                query_dist = cls.dist_point_to_segment(*ps)

            if query_dist < dist:
                dist = query_dist
                if position:
                    pos = query_pos
        if position:
            return dist, pos
        return dist

    @staticmethod
    def offset_line(line, offset):
        ''' Returns 'line' offset by 'offset'.

        'line' is a line segment of form [x0,y0,x1,y1].
        'offset' is an [x,y] point which is added to 'line'.

        Edge.offset_line(np.arr[float(x4)], arr[float(x2)]) -> arr[float(x4)]

        '''
        return (line.reshape((2,2)) + offset).reshape(4)

    @staticmethod
    def get_intersection_points(positions, intersections):
        ''' Extracts and returns a list of [X,Y] points from 'positions'.

        'positions' is an output array from Edge.get_intersections.

        Edge.get_intersection_points(np.arr[arr[arr[float(xN)](xN)](x2)],
                                     np.arr[arr[float(xN)](xN)])
                -> arr[arr[float(x2)](xN)]

        '''
        X = positions[0]
        Y = positions[1]
        n = len(X)
        points = []

        for i in range(n):
            for j in range(i+1, n):
                if intersections[i,j]:
                    points += [[X[i,j], Y[i,j]]]
        return points

    @staticmethod
    def cross_orientation(a, b, c):
        ''' Returns the orientation of the cross-product between points a,b,c.
        Orientation is -1, 0, or +1.

        'a','b','c' are [x,y] points, and have a 1 added for the cross product.

        Edge.cross_orientation(np.arr[float(x2)], np.arr[float(x2)],
                               np.arr[float(x2)]) -> int

        '''
        ax, ay = a; bx, by = b; cx, cy = c
        return np.sign(np.linalg.det([[bx,by],[cx,cy]]) - \
                       np.linalg.det([[ax,ay],[cx,cy]]) + \
                       np.linalg.det([[ax,ay],[bx,by]]))

    @classmethod
    def line_intersection(cls, line0, line1):
        ''' Returns True if line0 intersects line1, else False.
        Returns -1 if line0 and line1 are collinear and intersect.

        cls.test_intersection(np.arr[float(x4)], np.arr[float(x4)]) -> int

        '''
        p0, q0 = line0.reshape((2,2))
        p1, q1 = line1.reshape((2,2))
        ori_0p1 = cls.cross_orientation(p0, q0, p1)
        ori_0q1 = cls.cross_orientation(p0, q0, q1)
        ori_1p0 = cls.cross_orientation(p1, q1, p0)
        ori_1q0 = cls.cross_orientation(p1, q1, q0)

        if ori_0p1 == ori_0q1 == 0 or ori_1p0 == ori_1q0 == 0:
            return -1 # collinear

        return ori_0p1 != ori_0q1 and ori_1p0 != ori_1q0

    @staticmethod
    def bounding_box_overlap(line0, line1):
        ''' Returns True if the bounding boxes of line0 and line1 intersect.
        Returns False otherwise, indicating a collision is impossible.

        Edge.bounding_box_overlap(np.arr[float(x4)], np.arr[float(x4)]) -> bool

        '''
        x00, y00, x01, y01 = line0
        x10, y10, x11, y11 = line1
        x0_min = min(x00, x01)
        x0_max = max(x00, x01)
        x1_min = min(x10, x11)
        x1_max = max(x10, x11)
        if x0_max < x1_min or x1_max < x0_min:
            return False

        y0_min = min(y00, y01)
        y0_max = max(y00, y01)
        y1_min = min(y10, y11)
        y1_max = max(y10, y11)
        if y0_max < y1_min or y1_max < y0_min:
            return False

        return True

    @classmethod
    def get_intersections(cls, lines, dist_thresh=0, angles=False,
                          positions=False, i_min=0, j_min=0, i_max=None,
                          j_max=None):
        ''' Returns a len(lines)*len(lines)*([1-4]) array of intersections.

        Lines are considered not to be self-intersecting.

        If lines[i] intersects lines[j] | i!=j, intersections[i,j] = 1, unless
            the lines are collinear --> intersections[i,j] = -1.
            No intersection --> intersections[i,j] = 0.

        When positions are being returned, collinear intersections have their
            position as the center of the overlapping region.

        'lines' is a vector of line segments in form [X0,Y0,X1,Y1].
        'dist_thresh' is the allowable distance between segments within which
            an intersection is still registered.
        'angles' is a boolean specifier determining if intersection angles are
            included in the output matrix. (default False)
        'positions' is a boolean specifier determining if intersection positions
            are included in the output matrix. (default False)
        'i_min' specifies which line to start from. (default 0)
        'j_min' specifies which line to compare from. (default 0, but
            internally set to min(j_max, i+1) for each line i)
        'i_max' specifies which line to stop before. If left as None,
            checks intersections for all the lines after i_min.
        'j_max' specifies which line to stop comparing before. If left as None,
            checks intersections for all the lines after max(j_min, i+1).

        To determine intersections between lists lines_m and lines_n, use
            Edge.get_intersections(np.array(lines_m+lines_n), j_min=m, i_max=m)

        cls.get_intersections(np.array[np.arr[float]], *int, *bool, *bool,
                              *int, *int, *int, *int)
                -> np.arr[arr[float]],*arr[arr[float]],*arr[arr[float]]

        '''
        # initialise relevant state variables
        l = len(lines)
        intersections = np.zeros((l,l))
        kwargs = {}
        if angles:
            r_angles = np.zeros((l,l))
            kwargs['angles'] = r_angles
        if positions:
            r_positions = np.zeros((l,l,2))
            kwargs['positions'] = r_positions
        if i_max is None:
            i_max = l
        if j_max is None:
            j_max = l

        # iterate over the lines to check for intersection
        for i in range(i_min, i_max):
            for j in range(max(j_min, i+1), j_max):
                if not dist_thresh and \
                   not cls.bounding_box_overlap(*lines[[i,j]]):
                    continue # intersection not possible

                # extract a list of endpoints from the lines
                points = np.array(cls.get_line_points(lines, i, j))

                # check if the lines share an end-point
                if cls.duplicate_point(points):
                    intersections[i,j] = True
                    if angles:
                        r_angles[i,j] = cls.angle_between_lines(*lines[[i,j]])
                    if positions:
                        for p0 in range(2):
                            for p1 in range(2,4):
                                if (points[p0] - points[p1]).sum() == 0:
                                    r_positions[i,j] = points[p0]
                    continue # intersection handled

                # check if the lines intersect
                intersection = cls.line_intersection(*lines[[i,j]])
                if not dist_thresh:
                    if not intersection:
                        continue # lines don't intersect

                if positions:
                    X, Y = points.T

                if intersection < 0:
                    intersections[i,j] = -1 # collinear
                    if angles:
                        r_angles[i,j] = 0
                    if positions:
                        indices = np.ones(X.shape, dtype=bool)
                        if cls.line_angle(lines[i]) in [0., -180., 180.]:
                            var = X
                        else:
                            var = Y
                        indices[[np.argmin(var), np.argmax(var)]] = False
                        points = points[indices]
                        r_positions[i,j] = points.mean(axis=0)
                elif intersection > 0:
                    intersections[i,j] = 1 # standard intersection
                    if angles:
                        r_angles[i,j] = cls.angle_between_lines(*lines[[i,j]])
                    if positions:
                        # basis transform
                        # set min x point of line i as origin
                        x_min_index = np.argmin(X[:2])
                        offset = points[x_min_index]
                        offset_points = points - offset
                        # set line to 1st/4th quadrant vector for angle calc
                        line = list(offset) + list(points[(x_min_index+1)%2])
                        # rotate lines so line i is along +ve x-axis
                        theta = -cls.line_angle(line, degrees=False)
                        sin_t = np.sin(theta); cos_t = np.cos(theta)
                        rot_mat = np.array([[cos_t, -sin_t],
                                            [sin_t, cos_t]])
                        X, Y = np.dot(rot_mat, offset_points.T)
                        x_temp = X[2] - Y[2] * (X[3] - X[2]) / (Y[3] - Y[2])
                        rot_mat *= [[1,-1],[-1,1]] # reverse rotation angle
                        point = np.dot(rot_mat, [x_temp,0]).T
                        r_positions[i,j] = point + offset
                elif dist_thresh:
                    # check for pseudo-intersection (if enabled)
                    cls._pseudo_intersection(lines, i, j, intersections,
                                             dist_thresh, **kwargs)

        # convert from triangular to symmetric matrices for output
        #   diagonals all 0, so no need to halve
        intersections += intersections.T
        if angles:
            r_angles += r_angles.T
        if positions:
            r_positions += r_positions.transpose((1,0,2))
            # swap position axes to have x/y as first index, not row
            r_positions = r_positions.swapaxes(0,2).swapaxes(1,2)

        # generate output as selected by flags
        depth = 1 + angles + 2*positions
        output = np.zeros((depth,l,l))
        output[0] = intersections
        if angles:
            output[1] = r_angles
            if positions:
                output[2:] = r_positions
        elif positions:
            output[1:] = r_positions

        return output

    @classmethod
    def get_joined_lines(cls, lines, col_thresh, dist_thresh):
        ''' Returns the lines which can be joined with the given requirements.

        Also returns a list of truth values over the indices of the inputted
            lines, denoting whether or not they are involved in a line joining.

        'lines' is a vector of line segments in form [X1,Y1,X2,Y2].
        'col_thresh' is a collinearity threshold in degrees.
            Intersecting lines with an angle of less than col_thresh between
                them are joined together.
        'dist_thresh' is a distance threshold in coordinate units, allowing
            for pseudo-intersections to be detected if line-segments are within
            dist_thresh of one another.

        cls.get_joined_lines(list[list[int]], float, float)
                -> (list[list[int]], list[bool])

        '''
        joined_lines = []   # storage for joined lines
        l = len(lines)
        # track lines which have been joined to others
        joins = np.zeros(l, dtype=bool)

        intersections, angles = cls.get_intersections(lines, dist_thresh,
                                                      angles=True)

        # iterate over lines
        for i in range(l):
            for j in range(i+1,l):
                # if line i intersects with line j, and neither have already
                #   been joined to another line,
                if intersections[i,j] and not joins[i] and not joins[j]:
                    angle = angles[i,j]

                    # if lines are collinear within desired tolerance
                    if abs(angle) <= col_thresh or \
                       abs(180 - angle) <= col_thresh:
                        # lines are approximately collinear
                        #   -> (add both to joins list)
                        joins[i] = joins[j] = True
                        # get points of lines
                        points = cls.get_line_points(lines, i, j)
                        # determine endpoints of joined line
                        new_line = cls.get_line(points)
                        # add the new line to the list
                        joined_lines = cls.add_new_lines(joined_lines,
                                                         new_line)

        return joined_lines, joins

    @classmethod
    def reduce_lines(cls, lines, col_thresh, min_changes=0, dist_thresh=0,
                     stability_thresh=0):
        ''' Returns the reduced lines based on the inputted minimum collinearity
            and number of changes between iterations.

        'lines' is a vector of line segments in form [X1,Y1,X2,Y2].
        'col_thresh' is a collinearity threshold in degrees.
            Intersecting lines with an angle of less than col_thresh between
                them are reduced to single lines
        'min_changes' is an optimisation threshold.
            If less than min_changes occur after an optimisation, the
                optimisation is stopped. (default 0)
        'dist_thresh' is a distance threshold in coordinate units, allowing
            for pseudo-intersections to be detected if line-segments are within
            dist_thresh of one another. (default 0)
        'stability_thresh' is a stability threshold for consecutive runs of
            less than min_changes changes. (default 0)

        cls.reduce_lines(np.arr[arr[float(x4)]], float, int, float, int)
                -> arr[arr[float(x4)]]

        '''
        # count the number of times min_changes has been reached
        min_count = 0

        while min_count <= stability_thresh:
            l = len(lines)
            # get all lines that can be joined together
            new_lines, joins = cls.get_joined_lines(lines, col_thresh,
                                                     dist_thresh)
            # add unjoined lines to new_lines (they're still valid)
            new_lines = cls.add_new_lines(new_lines, *lines[np.invert(joins)])
            num_changes = abs(l - len(new_lines))
            lines = new_lines
            if num_changes <= min_changes:
                min_count += 1
            else:
                min_count = 0

        return new_lines

    @staticmethod
    def is_out_of_bounds(query_point, boundaries):
        ''' Returns True if the query point is outside the boundaries,
            else False.

        'queryPoint' is an [x,y] coordinate
        'boundaries' is a list of [xmin, xmax, ymin, ymax]
            -> NOTE: boundaries can be +/-Inf if only some are desired

        Edge.is_out_of_bounds(np.arr[float(x4)], arr[float(x2)]) -> bool

        '''
        x_min, x_max, y_min, y_max = boundaries
        x_q, y_q = query_point
        return not (x_min <= x_q <= x_max and y_min <= y_q <= y_max)

    @classmethod
    def scale_lines(cls, lines, scale_factor, boundaries=[-np.Inf,np.Inf]*2):
        ''' Returns 'lines' scaled about their centres by 'scale_factor'.

        'lines' is a vector of line segments in form [X1,Y1,X2,Y2].
        'scaleFactor' is a +ve float, 1 is unity scale.
        'boundaries' is a list of [xmin, xmax, ymin, ymax], which, if provided,
            limit the scaling range of each line. Lines scaled with boundaries
            in place are extended to the minimum of the new extension and the
            boundary. Boundaries can be None if only some are desired.

        Algorithm ignores the possibility of a line detected that lies on a
            boundary line.

        cls.scale_lines(np.arr[arr[float(x4)]], float, *list[float(x4)])
                -> arr[arr[float(x4)]]

        '''
        lines = np.array(lines)
        new_lines = lines # initialise to old lines
        # logical array of line validity - false excluded at the end
        include = np.ones(len(lines), dtype=bool)

        # generate boundary lines
        xmin, xmax, ymin, ymax = boundaries
        boundary_lines = np.array(
            [[xmin, ymin, xmin, ymax], #xmin
             [xmax, ymin, xmax, ymax], #xmax
             [xmin, ymin, xmax, ymin], #ymin
             [xmin, ymax, xmax, ymax], #ymax
             np.zeros(4)])

        # iterate over lines
        for line_ind in range(len(lines)):
            line = lines[line_ind]
            # check if line is a point (remove from list if so)
            if cls.is_point(line):
                include[line_ind] = False
                continue

            if scale_factor != 1:
                # get current points
                p0, p1 = line.reshape((2,2))
                midpoint = (p0 + p1) / 2;
                # scale points about midpoint
                p_new = np.array(
                    [(p0 - midpoint) * scale_factor + midpoint,
                     (p1 - midpoint) * scale_factor + midpoint])
            else:
                p_new = line.reshape((2,2))

            # check boundary conditions, fix if required
            p0out = cls.is_out_of_bounds(p_new[0], boundaries)
            p1out = cls.is_out_of_bounds(p_new[1], boundaries)

            # check if scaled points are outside the boundary
            if p0out or p1out:
                # add new line to boundary lines list
                boundary_lines[4] = p_new.reshape(4)

                # get boundary intersections
                ip = cls.get_intersections(boundary_lines, positions=True)
                intersects = np.array(np.abs(ip[0,4]), dtype=bool)
                num_intersections = intersects.sum()
                new_poss = np.unique(np.array([ip[1,4], ip[2,4]]).T[intersects],
                                     axis=0)
                num_crosses = len(new_poss)

                if num_intersections == 0:
                    # line is fully outside the boundaries (remove)
                    include[line_ind] = False
                    continue
                if num_crosses == 1:
                    # 1 out, 1 in (redefine one point)
                    new_point = new_poss
                    if p0out:
                        p_new[0] = new_point
                    else:
                        p_new[1] = new_point
                elif num_crosses == 2:
                    # 2 boundaries uniquely intersected with, line redefined by
                    #   its boundary intersections
                    p_new = new_poss
                else:
                    # NOT REACHED (or at least it shouldn't be...)
                    print('NO!', num_intersections, num_crosses, ip, new_poss)
            # replace old points with the new ones
            new_lines[line_ind] = p_new.reshape(4)
        # remove out of bounds lines
        return new_lines[include]

    @staticmethod
    def scale_to_boundaries(line, boundaries):
        ''' Returns 'line' scaled to the provided boundaries.

        'line' is of the form [x0,y0,x1,y1].
        'boundaries' is of the form [xmin, xmax, ymin, ymax].

        Edge.scale_to_boundaries(np.arr[float(x4)], arr[float(x4)])
                -> np.arr[float(x4)]

        '''
        x0,y0,x1,y1 = line
        x0p, y0p, x1p, y1p = x0, y0, x1, y1 # copy for use in functions
        y = lambda x : (y1p - y0p) * (x - x0p) / (x1p - x0p + 1e-9) + y0p
        x = lambda y : (x1p - x0p) * (y - y0p) / (y1p - y0p + 1e-9) + x0p
        xmin, xmax, ymin, ymax = boundaries
        # scale to x boundaries
        x0, y0 = xmin, y(xmin)
        x1, y1 = xmax, y(xmax)
        # if outside y boundaries, scale back to y-boundaries
        if y0 < ymin or y0 > ymax:
            y0 = ymin if y0 < ymin else ymax
            x0 = x(y0)
        if y1 < ymin or y1 > ymax:
            y1 = ymin if y1 < ymin else ymax
            x1 = x(y1)
        return np.array([x0,y0,x1,y1], dtype=int)

    @classmethod
    def _pseudo_intersection(cls, lines, i, j, intersections, dist_thresh,
                             **kwargs):
        ''' Evaluates if a pseudo-intersection has occurred.

        Updates relevant variables if an intersection is registered.

        cls._pseudo_intersection(list[list[float]], int, int, float,
                                  list[list[bool]]) -> None

        '''
        # parse kwargs
        angles = kwargs.pop('angles', None)
        calculate_angle = angles is not None and len(angles) > 0
        positions = kwargs.pop('positions', None)
        calculate_position = positions is not None and len(positions) > 0

        # determine if pseudo-intersection has occurred.
        dist = cls.min_dist_segments(lines[[i,j]], position=calculate_position)
        if calculate_position:
            dist, pos = dist # extract correct variables from result

        if dist < dist_thresh:
            intersections[i,j] = True
            if calculate_angle:
                angles[i,j] = cls.angle_between_lines(lines[i], lines[j])
            if calculate_position:
                positions[i,j] = pos

