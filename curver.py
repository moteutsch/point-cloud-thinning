import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize, curve_fit
import scipy.stats
from scipy.spatial.transform import Rotation as R

############
# TODO: 
# + Use EMST
# + TODO: Use weighting of line, quadratic, plane!
# + 3d
############


#pts = [ np.polynomial.polynomial.polyval(t, p) for t in np.linspace(-10, 10, 20) ]
def circle(how_many_pts):
    return np.array([ (np.sin(t), np.cos(t)) for t in np.linspace(0, 2 * np.pi, how_many_pts) ])

def polynomial_curve(c, how_many_pts):
    poly = np.poly1d(c)
    return np.array([ (t, poly(t)) for t in np.linspace(-2, 2, how_many_pts) ])

def pts_with_noise(c, n, sigma=0.05):
    #return c(n) + np.random.normal(0, sigma, size=(n, 2))
    return np.array([ a + np.random.normal(0, sigma * (abs((1 - abs(a[0]))) ** (1/2)) * 2, size=(2)) for a in c(n) ])

def ball(center, r, all_pts):
    return np.array([ pt for pt in all_pts if np.linalg.norm(center - pt) < r ])

def ball_c(center, r, all_pts):
    return np.array([ pt for pt in all_pts if np.linalg.norm(center - pt) >= r ])



def linear_regression_line(pts, p):
    x = pts[:, 0]
    y = pts[:, 1]
    p0 = 0, 1 # initial guess

    #popt, pcov = curve_fit(f, x, y, p0, sigma=sigma, absolute_sigma=True)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    x = scipy.stats.linregress(x,y)
    #print(x)
    return x

    popt, pcov = curve_fit(f, x, y, p0)
    #print(pot, pcov)
    return popt, pcov

def rot(theta):
    return np.array([
        [ np.cos(theta), -np.sin(theta) ],
        [ np.sin(theta), np.cos(theta) ]
    ])

def pt_correlations(pts, p):
    s, i, _, _, _ = linear_regression_line(pts, p)

    # Transform the pt set so that line is parallel to x-axis and p is origin
    #assert (p == np.array([0, i])).all()
    #print((0, i))

    theta = np.arctan(s)
    R = rot((np.pi / 4) - theta)

    #### TODO: Remove
    #r = np.array([ R.dot((t, s * t + i) - p) + p for t in np.linspace(-0.5, 0.5, 100) ])
    #plt.scatter(*(r.T), c='green', s=0.1)
    ####

    pts2 = np.array([ R.dot(pt - p) for pt in pts ])
    p2 = p - p

    rho, _ = scipy.stats.pearsonr(*(pts2.T))
    return rho, pts2


# TODO: Weighted regressions?
def quadractic_regression_curve(pts, p, plot=False):
    s, i, _, _, _ = linear_regression_line(pts, p)
    theta = np.arctan(s)
    R = rot(theta)
    R_inv = np.linalg.inv(R)

    pts2 = np.array([ R.dot(pt - p) for pt in pts ])
    z = np.polyfit(*(pts2.T), 2) # 2 = quadratic (degree)

    if plot:
        # Plot:
        poly = np.poly1d(z)
        poly_pts = np.array([ R_inv.dot((t, poly(t))) + p for t in np.linspace(-0.3, 0.3, 100) ])
        plt.scatter(*(poly_pts.T), c='cyan', s=0.1)
        #

    # p_proj = p projected onto quadratic curve
    p_proj = R_inv.dot((0, z[-1])) + p

    if plot:
        plt.scatter([ p_proj[0] ], [ p_proj[1] ], c='black') # To
        plt.scatter([ p[0] ], [ p[1] ], c='gray') # From

    return z, p_proj



    # TODO: Inverse operation

# Algorithm2 = Collect2 (non-iterative version--contents of loop)
def collect2(pt, r, corr_tol, r_step, all_pts, max_iters=30):
    #print("## Collect2")
    H = r
    rho = -1

    i = 0
    while rho < corr_tol:
        i += 1
        #assert i < 10 # Max steps until crash
        if i >= max_iters:
            return None

        A = ball(pt, H, all_pts)
        H += r_step
        if len(A) <= 2: continue # Not enough points for correlation
        rho, rot_pts = pt_correlations(A, pt)

        #print("Iteration: %d; Rho: %s; H = %s" % (i, rho, H))

    return A


def thin_single_pt_2d(pt, all_pts):
    r = 0.35
    r_step = 0.03
    corr_tol = 0.5

    A = collect2(pt, r, corr_tol, r_step, all_pts)
    if A is None:
        print("### Warning: Could not find nbhd with sufficient correlation")
        # Simply don't modify p
        #new_pts.append(p)
        return None
    else:
        curve, pt_proj = quadractic_regression_curve(A, pt)
        return pt_proj


def thin_pt_cloud(pts):
    new_pts = []
    for pt in pts:
        res = thin_single_pt_2d(pt, pts)
        if res is not None:
            new_pts.append(res)

    return np.array(new_pts)

########## 3D

# TODO: Collect2 for 3D

def thin_single_pt_3d(pt, all_pts, ax=None):
    
    # TODO: Iterative
    H = 0.35 


    #for pt in pts:
    #pt = all_pts[80]
    A = ball(pt, H, all_pts) # TODO: EMST
    M = np.vstack(
        (
            np.ones(len(A)),
            (A.T)[0:2]
        )
    ).T
    print("M5", M[:5, :])
    print("A5", A[:5, :])
    z = A[:, 2]
    print("z5", z[:5])

    p, res, rnk, s = scipy.linalg.lstsq(M, z)
    print(p, res, rnk, s)


    def plane(x, y):
        return p[0] + p[1] * x + p[2] * y


    # TODO: Rename
    how_many_pts = 50
    grid = np.linspace(-1.5, 0.5, how_many_pts)
    plane_pts = np.array([ (x, y, plane(x, y)) for x in grid for y in grid ])

    # Tranform our plane to the plane {z = 0}
    #plane_rot = R.from_euler('xy', [np.arctan(p[1]), np.arctan(p[2])]).as_dcm()
    plane_rot = R.from_euler('xy', [-np.arctan(p[2]), np.arctan(p[1])]).as_dcm()
    plane_rot_inv = np.linalg.inv(plane_rot)

    new_origin = np.array((0, 0, p[0]))

    #pts_at = plane_rot.dot((plane_pts - new_origin).T).T
    print(plane_pts[:5])
    
    def map_pts(pt_set):
        return np.array([ plane_rot.dot((a_pt - new_origin).T) for a_pt in pt_set ])

    def inv_map_pts(pt_set):
        return np.array([ plane_rot_inv.dot(a_pt) + new_origin for a_pt in pt_set ])

    mapped_plane_pts = map_pts(plane_pts)

    # For some reason this doesn't send exactly to z = 0; so keep track of error in z
    avg_z_err = mapped_plane_pts[:, 2].mean()
    print('Average z error on plane', avg_z_err)

    print('After transform', mapped_plane_pts[:5])

    if ax is not None:
        ax.scatter(*(A.T), s=40, color='green')
        ax.scatter( *(np.array([ pt ]).T), color='grey', s=100)
        ax.scatter(*(plane_pts.T), s=0.3, color='orange')
        ax.scatter(*(mapped_plane_pts.T), s=0.3, color='yellow')
        ax.scatter(*((map_pts(A)).T), s=30, color='cyan')

    # 2D projection
    pt_on_plane = map_pts([pt])[0]
    res_2d = thin_single_pt_2d(pt_on_plane[:2], mapped_plane_pts[:, :2])
    if res_2d is None:
        # TODO: Better
        print("3D WARNING!")
        return

    # We make it back to a 3D on mapped plane (we put the original coordinate and not "0" 
    # as there might be a small error in z-coordinate when projecting (maybe pt_on_plane[2] /= 0)


    res_3d_mapped = np.hstack((res_2d, pt_on_plane[2]))
    #res_3d_mapped = np.hstack((res_2d, 0))
    res_3d = inv_map_pts([ res_3d_mapped ])[0]

    if ax is not None:
        ax.scatter( *(np.array([ res_3d ]).T), color='black', s=100)

    
def thin_pt_cloud_3d(pts, ax=None):
    new_pts = []
    for pt in pts:
        res = thin_single_pt_3d(pt, pts, ax=ax)
        if res is not None:
            new_pts.append(res)

    return np.array(new_pts)

