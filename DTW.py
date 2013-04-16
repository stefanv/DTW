__author__ = 'marcdeklerk'

import numpy as np

class DTW():
	CASE1 = 0
	CASE2 = 1
	CASE3 = 2

	def __init__(self, seq1, seq2, dist_func, case=CASE1):

		self._seq1 = seq1
		self._seq2 = seq2
		self._dist_func = dist_func
		self.case = case

		shape = (2+len(seq1), 2+len(seq2))
		self._dist_matrix = np.zeros(shape)

		self._dist_matrix[:] = -1
		self._dist_matrix[2:len(seq1)+2, 0:2] = float('inf')
		self._dist_matrix[0:2, 2:len(seq2)+2] = float('inf')
		self._dist_matrix[0:2, 0:2] = 0

	def dp_backwards(self, p, q):
		if p == 0 and q == 0:
			pass

		if self._dist_matrix[2+p, 2+q] != -1:
			return self._dist_matrix[2+p, 2+q]

		if self.case == DTW.CASE1:
			min_p, min_q = min((p-2, q-1), (p-1, q-1), (p-1, q-2), key=lambda x: self.dp_backwards(*x))
		elif self.case == DTW.CASE2:
			min_p, min_q = min((p-1, q), (p-1, q-1), (p, q-1), key=lambda x: self.dp_backwards(*x))
		elif self.case == DTW.CASE3:
			min_p, min_q = min((p-2, q-1), (p-1, q), (p-1, q-1), (p, q-1), (p-1, q-2), key=lambda x: self.dp_backwards(*x))

		self._dist_matrix[2+p, 2+q] = self._dist_func(self._seq1[p], self._seq2[q]) + self.dp_backwards(min_p, min_q)

		return self._dist_matrix[2+p, 2+q]

		return

	def calculate(self):
		return self.dp_backwards(len(self._seq1)-1, len(self._seq2)-1)

	def get_path(self):
		p, q = (len(self._seq1)-1, len(self._seq2)-1)

		path = []

		while p != -1 and q != -1:
			path.append((p, q))

			if self.case == DTW.CASE1:
				min_p, min_q = min((p-2, q-1), (p-1, q-1), (p-1, q-2), key=lambda x: self.dp_backwards(*x))
			elif self.case == DTW.CASE2:
				min_p, min_q = min((p-1, q), (p-1, q-1), (p, q-1), key=lambda x: self.dp_backwards(*x))
			elif self.case == DTW.CASE3:
				min_p, min_q = min((p-2, q-1), (p-1, q), (p-1, q-1), (p, q-1), (p-1, q-2), key=lambda x: self.dp_backwards(*x))

			p, q = min_p, min_q

		return path

def distance(x, y):
	return abs(x-y)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	np.set_printoptions(threshold=np.nan)
	np.set_printoptions(linewidth=np.nan)

	vec1=[71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68, 76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79, 80, 80, 78];
	vec2=[69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69, 69, 71, 73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80, 78];

#	vec1=[69, 73, 80, 75, 79, 80, 79, 78, 76, 73, 71, 70, 70, 69, 69, 71, 73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80, 78]
#	vec2=[71, 73, 75, 80, 78, 75, 71, 75, 73, 71, 72, 75, 76, 76, 76, 76, 75, 73, 78, 79, 80, 80, 78]

	t = np.array(vec1)
	r = np.array(vec2)

	f1 = plt.figure()
	f2 = plt.figure()
	f3 = plt.figure()

	for f, case in [(f1, DTW.CASE1), (f2, DTW.CASE2), (f3,DTW.CASE3)]:
		dtw = DTW(t, r, distance, case)
		dtw.calculate()

		m = np.zeros((len(t), len(r)))

		path = dtw.get_path()

		for point in path:
			m[point[0], point[1]] = 1

		np.set_printoptions(formatter={'all':lambda x: str('X') if x != 0 else str(' ')})
		print m

		np.set_printoptions()

		stretch = True

		if stretch:
			stretch = float(len(t))/len(r)

		ax = f.add_subplot(111, projection='3d')

		ax.plot(np.arange(len(t)), [0]*len(t), t)
		ax.plot(stretch*np.arange(len(r)), [1]*len(r), r)

		i = 0
		for edge in reversed(path):
			ax.plot([edge[0], stretch*edge[1]], [0, 1], [t[edge[0]], r[edge[1]]],  color="gray")

			i += 1

	plt.show()

	True