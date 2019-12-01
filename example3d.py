import curver
import matplotlib.pyplot as plt
import numpy as np

########################

def helix_3d(how_many):
    return np.array([ (np.cos(t), np.sin(t), t) for t in np.linspace(0, 2 * np.pi, how_many) ])

########################




from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


pts = helix_3d(200)


ax.scatter(*(pts.T))

#plt.show()

curver.thin_single_pt_3d(pts[80], pts, ax=ax)

#res_pts = curver.thin_pt_cloud_3d(pts)
#ax.scatter(*(res_pts.T), color='green', alpha=0.5)

plt.show()
