import curver
import matplotlib.pyplot as plt
import numpy as np

########################

HELIX_SCALE = 5

def helix_3d(how_many):
    pts = np.array([ (HELIX_SCALE * 3 * np.cos(t), HELIX_SCALE * 3 * np.sin(t), t) for t in np.linspace(0, HELIX_SCALE * np.pi, how_many) ])
    #noise = np.random.normal(0, 0.55, size=pts.shape)
    noise = np.array([ 1.8 * np.cos(t * 6) * np.random.normal(0, 0.55, size=3) for t in np.linspace(0, HELIX_SCALE * np.pi, how_many) ])

    return pts, noise

def make_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return ax
    

########################

NUM_ITERATIONS = 9


from mpl_toolkits.mplot3d import Axes3D



pts, noise = helix_3d(300)
#noise = np.random.normal(0, 0.55, size=pts.shape)
pts_with_noise = pts + noise

print("L2 error of original pts:", curver.l2_error(pts, pts))
print("L2 error of noisy pts:", curver.l2_error(pts, pts_with_noise))

ax = make_ax()
ax.scatter(*(pts.T), alpha=0.5)
ax.scatter(*(pts_with_noise.T), color='red', alpha=0.5)
plt.show()


#ax = make_ax()
#ax.scatter(*(pts_with_noise.T), color='red', alpha=0.5)
#curver.thin_single_pt_3d(pts[240], pts, ax=ax)
#plt.show()


#return


res_pts = pts_with_noise
for i in range(NUM_ITERATIONS):

    res_pts = curver.thin_pt_cloud_3d(res_pts)
    print("Thinning iteration: ", i)
    print("Number of remaining pts:", len(res_pts))
    print("L2 error of thinned pts:", curver.l2_error(pts, res_pts))

    if i % 3 != 0:
        continue

    ax = make_ax()
    ax.set_title('Iteration: %s' % i)
    ax.scatter(*(pts.T), alpha=0.5)
    ax.scatter(*(res_pts.T), color='green', alpha=0.5)
    plt.show()

    ax = make_ax()
    ax.scatter(*(pts_with_noise.T), color='red', alpha=0.5)
    curver.thin_single_pt_3d(pts[240], pts, ax=ax)
    plt.show()



ax = make_ax()
ax.scatter(*(pts.T), alpha=0.5)
ax.scatter(*(pts_with_noise.T), color='red', alpha=0.5)
ax.scatter(*(res_pts.T), color='green', alpha=0.5)
plt.show()
