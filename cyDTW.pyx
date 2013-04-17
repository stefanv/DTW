cdef extern from "float.h":
    double DBL_MAX

cdef extern from "math.h":
    double sqrt(double)

import numpy as np


cdef int type1 = 1
cdef int type2 = 2
cdef int type3 = 3


cdef inline double euclidean(double x, double y):
    return sqrt((x - y) * (x - y))


cdef inline double min3(double[3] v):
    cdef int i, m = 0

    for i in range(1, 3):
        if v[i] < v[m]:
            m = i

    return v[m]


def dtw(double[:] x, double[:] y, int case=1):
    """DTW(sequence1, sequence2, case=type1)

    Dynamic time warping (DTW) is an algorithm for measuring similarity
    between two sequences which may vary in time or speed

    For instance, similarities in walking patterns would be detected,
    even if in one video the person was walking slowly and if in another
    he or she were walking more quickly, or even if there were accelerations
    and decelerations during the course of one observation

    After instantiating the class, use the method calculate() to perform the
    matching and return the difference measure and get_path() to return the
    matching.

    Parameters
    ----------
    x : 1-D ndarray, dtype float64
        A 1-dimensional sequence of points.
    y : 1-D ndarray, dtype float64
        A 1-dimensional sequence of points.
    case : int, {1, 2, 3}
        Type-1 DTW uses 27-, 45- and 63-degree local path constraint.
        Type-2 DTW uses 0-, 45- and 90-degree local path constraint.
        Type-3 DTW uses a combination of Type-1 and Type-2

    References
    ----------
    .. [1] http://mirlab.org/jang/books/dcpr/dpDtw.asp?title=8-4%20Dynamic%20Time%20Warping
    .. [2] http://en.wikipedia.org/wiki/Dynamic_time_warping

    """
    cdef:
        int m = len(x)
        int n = len(y)
        double[:, ::1] distance
        int i, j, min_i, min_j
        double[3] costs
        double prev_cost

    distance = np.zeros((m + 1, n + 1)) + DBL_MAX
    distance[1, 1] = 0

    # Step forward
    for i in range(2, m + 1):
        for j in range(2, n + 1):
            # Could limit the fan here and save some time

            costs[0] = distance[i - 1, j - 1]
            costs[1] = distance[i - 2, j - 1]
            costs[2] = distance[i - 1, j - 2]

            distance[i, j] = euclidean(x[i - 1], y[j - 1]) + min3(costs)

    # Trace back
    cdef list path = [(m - 1, n - 1)]

    i = m
    j = n
    while not ((i == 1) and (j == 1)):
        path.append((i - 1, j - 1))

        min_i, min_j = i - 1, j - 1

        if distance[i - 2, j - 1] < distance[min_i, min_j]:
            min_i, min_j = i - 2, j - 1

        if distance[i - 1, j - 2] < distance[min_i, min_j]:
            min_i, min_j = i - 1, j - 2

        i, j = min_i, min_j

    path.append((0, 0))

    return path, np.asarray(distance)
