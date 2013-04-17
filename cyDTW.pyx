# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False

__author__ = 'marcdeklerk'

cdef extern from "float.h":
    double DBL_MAX

from cython.view cimport array as cvarray

cdef int type1 = 0
cdef int type2 = 1
cdef int type3 = 2

cdef class DTW:
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
    sequence1 : Numpy.ndarray[ndim=1, dtype=numpy.double]
        A 1 dimensional sequence of points
    sequence2 : Numpy.ndarray[ndim=1, dtype=numpy.double]
        A 1 dimensional sequence of points
    case : Enum type?
        Type-1 DTW uses 27-, 45- and 63-degree local path constraint.
        Type-2 DTW uses 0-, 45- and 90-degree local path constraint.
        Type-3 DTW uses a combination of Type-1 and Type-2

    References
    ----------
    http://mirlab.org/jang/books/dcpr/dpDtw.asp?title=8-4%20Dynamic%20Time%20Warping
    http://en.wikipedia.org/wiki/Dynamic_time_warping
    """

    cdef int case

    cdef double[:] _seq1
    cdef double[:] _seq2

    cdef readonly double[:, :] _dist_matrix

    def __cinit__(self, double[:] seq1, double[:] seq2, int case=type1):
        self._seq1 = seq1
        self._seq2 = seq2
        self.case = case

        self._dist_matrix = cvarray(shape=(2+len(seq1), 2+len(seq2)), itemsize=sizeof(double), format="d")

        self._dist_matrix[:] = -1
        self._dist_matrix[2:seq1.shape[0]+2, 0:2] = DBL_MAX
        self._dist_matrix[0:2, 2:seq2.shape[0]+2] = DBL_MAX
        self._dist_matrix[0:2, 0:2] = 0

    cdef inline double dp_backwards(self, int p, int q) except *:
        if p == 0 and q == 0:
            pass

        if self._dist_matrix[2+p, 2+q] != -1:
            return self._dist_matrix[2+p, 2+q]

        cdef int min_p, min_q
        cdef double cost

        min_p, min_q = p-1, q-1
        cost = self.dp_backwards(p-1, q-1)

        if self.case == type1 or self.case == type3:
            cost2 = self.dp_backwards(p-2, q-1)
            if cost2 < cost: min_p, min_q, cost = p-2, q-1, cost2

            cost2 = self.dp_backwards(p-1, q-2)
            if cost2 < cost: min_p, min_q, cost = p-1, q-2, cost2

        if self.case == type2 or self.case == type3:
            cost2 = self.dp_backwards(p-1, q)
            if cost2 < cost: min_p, min_q, cost = p-1, q, cost2

            cost2 = self.dp_backwards(p, q-1)
            if cost2 < cost: min_p, min_q, cost = p, q-1, cost2

        self._dist_matrix[2+p, 2+q] = self.distance(self._seq1[p], self._seq2[q]) + self.dp_backwards(min_p, min_q)

        return self._dist_matrix[2+p, 2+q]

    def calculate(self):
        return self.dp_backwards(self._seq1.shape[0]-1, self._seq2.shape[0]-1)

    def get_path(self):
        cdef int p = self._seq1.shape[0]-1
        cdef int q = self._seq2.shape[0]-1

        path = []

        cdef int min_p, min_q
        cdef double cost, cost2

        while p != -1 and q != -1:
            path.append((p, q))

            min_p, min_q = p-1, q-1
            cost = self.dp_backwards(p-1, q-1)

            if self.case == type1 or self.case == type3:
                cost2 = self.dp_backwards(p-2, q-1)
                if cost2 < cost: min_p, min_q, cost = p-2, q-1, cost2

                cost2 = self.dp_backwards(p-1, q-2)
                if cost2 < cost: min_p, min_q, cost = p-1, q-2, cost2

            if self.case == type2 or self.case == type3:
                cost2 = self.dp_backwards(p-1, q)
                if cost2 < cost: min_p, min_q, cost = p-1, q, cost2

                cost2 = self.dp_backwards(p, q-1)
                if cost2 < cost: min_p, min_q, cost = p, q-1, cost2

            p, q = min_p, min_q

        return path

    cdef inline double distance(self, double x, double y):
        return x-y if x > y else y -x
