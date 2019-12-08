import curver
import matplotlib.pyplot as plt
import numpy as np

########################

HELIX_SCALE = 5

# Generate points on a helix and appropriate noise 
def helix_3d(how_many_pts):
    pts = np.array([ (HELIX_SCALE * 3 * np.cos(t), HELIX_SCALE * 3 * np.sin(t), t) for t in np.linspace(0, HELIX_SCALE * np.pi, how_many_pts) ])
    #noise = np.random.normal(0, 0.55, size=pts.shape)
    noise = np.array([ 1.8 * np.cos(t * 6) * np.random.normal(0, 0.3, size=3) for t in np.linspace(0, HELIX_SCALE * np.pi, how_many_pts) ])

    return pts, noise

def make_ax_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return ax
    

########################

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap


pts, noise = helix_3d(300)
pts_with_noise = pts + noise

# We shuffle the points to show how we don't need the original order
np.random.shuffle(pts_with_noise)

# We embed the   curve with noise   onto 1-dimensional space
embedding = Isomap(n_components=1)
X_transformed = embedding.fit_transform(pts_with_noise)

# For every point, we now have its image in the real line
# We size the original 3D points by this location (after translating so 
# that all the sizes are positive).
sizes = X_transformed - min(X_transformed)


# Plot points with ordering given by size
ax = make_ax_3d()
ax.scatter(*(pts_with_noise.T), s=sizes, alpha=0.5)
plt.show()
