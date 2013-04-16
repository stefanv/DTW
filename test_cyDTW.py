__author__ = 'marcdeklerk'

from cyDTW import DTW

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)

vec1=[71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68, 76, 76, 75, 73, 71, 70, 70, 69, 68, 68, 72, 74, 78, 79, 80, 80, 78];
vec2=[69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69, 69, 69, 71, 73, 75, 76, 76, 76, 76, 76, 75, 73, 71, 70, 70, 71, 73, 75, 80, 80, 80, 78];

t = np.array(vec1, np.double)
r = np.array(vec2, np.double)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()

for f, case in [(f1, 0), (f2, 1), (f3, 2)]:
	dtw = DTW(t, r, case)
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

	ax.plot(np.arange(len(t)), [0]*len(t), t, 'b')
	ax.plot(np.arange(len(t)), [0]*len(t), t, 'b.')
	ax.plot(stretch*np.arange(len(r)), [1]*len(r), r, 'r')
	ax.plot(stretch*np.arange(len(r)), [1]*len(r), r, 'r.')

	i = 0
	for edge in reversed(path):
		print [edge[0], edge[1]]
		ax.plot([edge[0], stretch*edge[1]], [0, 1], [t[edge[0]], r[edge[1]]],  color="gray")

		i += 1

	print dtw._dist_matrix

plt.show()

True