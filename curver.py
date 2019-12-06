import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize, curve_fit
import scipy.stats
from scipy.spatial.transform import Rotation as R
import curver_config as config

############
# TODO: 
# + Write captions and unify colors in examples
# + Use EMST for "ball"
# + Use iterative ball radius until sufficient correlation in 3D
# + Use weighting (by distance; w_i in paper) of line, quadratic, plane regressions!
############

# Given two point sets containing: GT and predictions (note: points don't correspond 1-1); compute mean l^2 error of points from GT; 
def l2_error(gt_pts, pred_pts):
    return np.array([
        min([ np.linalg.norm(pred_pt - gt_pt) for gt_pt in gt_pts ]) # Distance between "pred_pt" and "gt_pts" (pt from set)
        for pred_pt in pred_pts
    ]).mean()




# Generate points on a circle
def circle(how_many_pts):
    return np.array([ (np.sin(t), np.cos(t)) for t in np.linspace(0, 2 * np.pi, how_many_pts) ])

# Generate points on a polynomial with coefficients "c"
def polynomial_curve(c, how_many_pts, dom_size=2):
    poly = np.poly1d(c)
    return np.array([ (t, poly(t)) for t in np.linspace(-dom_size, dom_size, how_many_pts) ])

# Generate "n" points on curve "curve(t)" (a function taking a single parameter "t") with normal noise of STD "sigma"
def pts_with_noise(curve, n, sigma=0.05):
    #return c(n) + np.random.normal(0, sigma, size=(n, 2))
    return np.array([ a + np.random.normal(0, sigma * (abs((1 - abs(a[0]))) ** (1/2)) * 2, size=(len(a))) for a in curve(n) ])

# A ball of radius "r" around point "center" from points "all_pts"
def ball(center, r, all_pts):
    return np.array([ pt for pt in all_pts if np.linalg.norm(center - pt) < r ])

# Complement of the ball (see "ball()")
def ball_c(center, r, all_pts):
    return np.array([ pt for pt in all_pts if np.linalg.norm(center - pt) >= r ])


# Compute linear regression line for "pts" ("p" is not used now, but will be used if we implement
# weighted linear regression--less cost for points that are far away from "p")
def linear_regression_line(pts, p):
    x = pts[:, 0]
    y = pts[:, 1]
    res = scipy.stats.linregress(x, y)
    return res


# 2d rotation matrix for angle "theta"
def rot(theta):
    return np.array([
        [ np.cos(theta), -np.sin(theta) ],
        [ np.sin(theta), np.cos(theta) ]
    ])


# Compute point pearson correlation of points after rotating by angle of best linear regession line
def pt_correlations(pts, p):
    s, i, _, _, _ = linear_regression_line(pts, p)

    # Transform the pt set so that line is parallel to x-axis and p is origin

    if np.isnan(s):
        # Infinite slope: i.e., vertical line 
        theta = np.pi / 2
    else:
        theta = np.arctan(s)

    R = rot((np.pi / 4) - theta)

    pts2 = np.array([ R.dot(pt - p) for pt in pts ])
    p2 = p - p
        

    #print(s, theta, R, pts, pts2)
    #print("INFO", s, theta, R, pts, pts2)
    rho, _ = scipy.stats.pearsonr(*(pts2.T))

    if config.is_debug:
        print("Pearson: %s" % rho)
        
        if config.debug_show_plots:
            line_pts = np.array([ (t, t) for t in np.linspace(-5, 5, 100) ])

            plt.scatter(*(pts.T), color='red', alpha=0.5)
            plt.scatter(*(line_pts.T), color='purple', alpha=0.3, s=0.5)
            plt.show()

    return rho, pts2


# Compute best quadratic regression curve that fits "pts" focused around point "p"
# We do this by:
# 1. finding linear regression line L
# 2. applying an affine transformation so that "p" is at origin and L = x-axis
# 3. Finding best quadratic appoximation function
# 4. Rotating back
#
# Note that rotations are necessary as, if "pts" form a vertical semi-circle, the best quadratic curve
# isn't a function y(x) (it isn't single-valued). But after rotation, it will be.

# If not "fit_quadratic" then uses linear fit
def quadratic_regression_curve(pts, p, fit_quadratic):
    s, i, _, _, _ = linear_regression_line(pts, p)
    l = np.array([ s, i ])

    if np.isnan(s):
        # Infinite slope: i.e., vertical line 
        theta = np.pi / 2
    else:
        theta = np.arctan(s)

    R = rot(theta)
    R_inv = np.linalg.inv(R)

    pts2 = np.array([ R.dot(pt - p) for pt in pts ])
    z = np.polyfit(*(pts2.T), 2) # 2 = quadratic (degree)

    if config.debug_show_plots:
        # Plot:
        poly = np.poly1d(z)
        poly_pts = np.array([ R_inv.dot((t, poly(t))) + p for t in np.linspace(-0.3, 0.3, 100) ])
        plt.scatter(*(poly_pts.T), c='cyan', s=0.1)

        line_poly = np.poly1d(l)
        line_pts = np.array([ (t, line_poly(t)) for t in np.linspace(-0.3, 0.3, 100) ])
        plt.scatter(*(line_pts.T), c='orange', s=0.1)


        #

    # "p" is at origin, so the projection onto polynomial curve (in the local coordinate system)
    # is just the pt: (0, "the free coefficient of the polynomial")
    if fit_quadratic:
        # p_proj = p projected onto quadratic curve
        local_coordinates_proj_p = (0, z[-1])
        p_proj = R_inv.dot(local_coordinates_proj_p) + p
    else:
        # We calculate projection onto line, by translating plane moving line to pass through origin (isometry)
        # projecting point onto line, and translating back
        new_orig = np.array((0, l[-1]))
        a = p - new_orig
        line_vec = np.array( (np.cos(theta), np.sin(theta)) )
        proj_len = np.dot(a, line_vec)
        p_proj = proj_len * line_vec + new_orig
        # p_proj = p projected ont linear curve
        #local_coordinates_proj_p = (0, l[-1])


    if config.debug_show_plots:
        plt.scatter(*(pts.T), color='red', alpha=0.5, s=15)
        plt.scatter([ p_proj[0] ], [ p_proj[1] ], c='black') # To
        plt.scatter([ p[0] ], [ p[1] ], c='gray') # From
        plt.show()

    return p_proj, theta, z


# Collect2 algorithm from paper (non-iterative version--contents of loop)
def collect2(pt, r, corr_tol, r_step, all_pts):
    H = r
    rho = 0 # initial value--no chance of passing threshold

    i = 0 # Count number of iterations

    # We increase the size of ball of points around "pt" that are used for regressions
    # until their Pearson-coefficient passes "corr_tol" tolerance

    A_history = []
    rho_history = []

    H -= r_step # To balance addition at beginning
    while abs(rho) < corr_tol:
        H += r_step
        i += 1
        if i >= config.two_dim['max_collect2_iterations']:
            # Max steps until failure
            #return None

            # If we haven't found correlation passing the threshold, we return maximum we've seen so far
            i_max = np.argmax(rho_history)

            if config.is_debug:
                print("### Warning: Passed max collect2 iterations; returning maximal correlation nbhd")
                print(pt, rho_history)
                print()

            return A_history[i_max], rho_history[i_max]

        A = ball(pt, H, all_pts)
        if len(A) <= config.min_pts_for_correlation: 
            print("##### Warning: Not enough points for correlation")
            print("Debug info; pt:", pt, "H:", H, "A:", len(A))
            print()
            continue # Not enough points for correlation


        if config.debug_show_plots:
            plt.scatter(*(A.T), color='red', alpha=0.5, s=15)
            plt.scatter(*(all_pts.T), color='green', alpha=0.5, s=5)
            plt.scatter(*(np.array([pt]).T), color='purple', alpha=1, s=15)
            plt.show()


        rho, rot_pts = pt_correlations(A, pt)

        A_history.append(A)
        rho_history.append(rho)

        if config.is_debug:
            print("Iteration: %d; Rho: %s; H = %s" % (i, rho, H))

    return A, rho


# Perform 2D curve "thinning" for point "pt"
# Returns where "pt" is projected to
def thin_single_pt_2d(pt, all_pts):
    r = config.two_dim['r']
    r_step = config.two_dim['r_step']
    corr_tol = config.two_dim['min_correlation']

    A, rho = collect2(pt, r, corr_tol, r_step, all_pts)
    if A is None:
        print("### Warning: Could not find nbhd with sufficient correlation")
        return None

    
    fit_quadratic = (abs(rho) < config.two_dim['min_correlation'])

    pt_proj, line_theta, curve = quadratic_regression_curve(A, pt, fit_quadratic=fit_quadratic)
    return pt_proj, line_theta, A


# Perform 2D curve "thinning" for all points
# Returns projections of all points
# Some points may be lost if they fail to reach correlation threshold
def thin_pt_cloud_2d(pts):
    new_pts = []
    for pt in pts:
        res = thin_single_pt_2d(pt, pts)
        if res is None:
            if not config.two_dim['remove_low_correlation_pts']:
                # If not remove, simply put back the original pt.
                new_pts.append(pt)
            else:
                pass # We ignore/remove this point
        else:
            pt_proj, _, _ = res
            new_pts.append(pt_proj)

    return np.array(new_pts)


def propogate_normals(pts):
    
    G = nx.Graph()
    mapping = {}
    for pt, i in zip(pts, range(len(pts))):
        res = thin_single_pt_2d(pt, pts)
        if res is None:
            print('Skipped point: ', pt)
            continue

        p_proj, l_theta, A = res

        #print(l_theta)
        #n = np.array((1, l[0] + l[1])) - np.array((0, l[1]))

        n = np.array( ( np.cos(l_theta), np.sin(l_theta) ) )

        #n = n / np.linalg.norm(n)
        #print("Normal: ", n)

        A_i = [ np.isclose(pts, A_pt).all(axis=1).nonzero()[0][0] for A_pt in A ]
        #print(A_i)

        mapping[i] = {
            'normal': n,
            'l_theta': l_theta,
            'neighbors': A_i
        }
        G.add_node(i, posxy=pt)

    pt_indices = mapping.keys()
    if len(pt_indices) == 0:
        print("Empty! Stopping.")
        return

    # Remove neighbors that were removed in previous step
    for pt_i in pt_indices:
        mapping[pt_i]['neighbors'] = list(set(pt_indices).intersection(set(mapping[pt_i]['neighbors'])))


    for pt_i in pt_indices:
        for n_i in mapping[pt_i]['neighbors']:
            n1 = mapping[pt_i]['normal']
            n2 = mapping[n_i]['normal']
            if not G.has_edge(pt_i, n_i):
                w = 1 - abs(np.dot(n1, n2))
                G.add_edge(pt_i, n_i, weight=w)


    T = nx.minimum_spanning_tree(G)

    from networkx.drawing.nx_pylab import draw
    positions = nx.get_node_attributes(T, 'posxy')
    draw(T, positions, node_size=5.0)
    plt.show()

    #print("Hello")
    #print(T.edges(data=True))
    #print("Bye")

    pt = 0

    edge_order = list(nx.bfs_edges(T, source=pt))
    for edge in edge_order:
        n1 = mapping[edge[0]]['normal']
        n2 = mapping[edge[1]]['normal']

        # Re-orient if necessary
        if np.dot(n1, n2) < 0:
            mapping[edge[1]]['normal'] = -n2

    # Now that normal orientations are propogated, we calculate the STD of normals by neighborhood
    pts_i = list(mapping.keys())
    #print(pts_i)
    stds = []
    normals = []
    for pt_id in pts_i:
        neighbors = mapping[pt_id]['neighbors']
        std = np.std([ mapping[neighbor_i]['normal'] for neighbor_i in neighbors ], axis=0).mean()
        mapping[pt_id]['std'] = float(std)
        stds.append(std)
        normals.append(mapping[pt_id]['normal'])
    normals = np.array(normals)
    stds = np.array(stds)


    #print(pts)
    #print(pts_i)
    #print(np.histogram(stds))
    # PLOT
    #plt.clear()


    pt_id = 0
    #l = mapping[pt_id]['l']

    #line_pts = np.array([ (t, t * l[0] + l[1]) for t in np.linspace(-1, 1, 100) ])
    #plt.scatter(*(line_pts.T), s=0.3, alpha=0.3, color='orange')


    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    norm = Normalize()
    norm.autoscale(stds)

    colormap = cm.inferno

    #print('STDs', stds)


    print(np.histogram(stds))
    #plt.scatter(*(pts[pts_i].T), s=(stds * 50))
    sc = plt.quiver(*(pts[pts_i].T), *(normals).T, color=colormap(norm(stds)), cmap=colormap)
    plt.colorbar(sc)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    return mapping

    #print(G)

        


########################
########## 3D
#######################

# Perform 3D curve "thinning" for point "pt"
# Returns where "pt" is projected to
# Currently non-iterative: works with fixed radius of ball
# Finds best plane regression around point "pt", projects points (in nbhd) onto this plane
# and does 2D algorithm (for a single point) on this plane. Then back-projects resulting point 
# back to 3D
# 
# If "ax" is passed (PLT axis) it will draw the transformations onto it
def thin_single_pt_3d(pt, all_pts, ax=None):
    
    H = config.three_dim['r']

    for i in range(config.three_dim['max_collect2_iterations']):

        if config.is_debug:
            print('Doing 3D point thinning with H = %s' % H)

        A = ball(pt, H, all_pts)
        M = np.vstack(
            (
                np.ones(len(A)),
                (A.T)[0:2]
            )
        ).T
        z = A[:, 2]

        p, res, rnk, s = scipy.linalg.lstsq(M, z)
        #print(p, res, rnk, s)

        def plane(x, y):
            return p[0] + p[1] * x + p[2] * y

        # Tranform our plane to the plane {z = 0}
        #plane_rot = R.from_euler('xy', [np.arctan(p[1]), np.arctan(p[2])]).as_dcm()
        plane_rot = R.from_euler('xy', [-np.arctan(p[2]), np.arctan(p[1])]).as_dcm()
        plane_rot_inv = np.linalg.inv(plane_rot)

        new_origin = np.array((0, 0, p[0]))

        #pts_at = plane_rot.dot((plane_pts - new_origin).T).T
        #print(plane_pts[:5])
        
        def map_pts(pt_set):
            return np.array([ plane_rot.dot((a_pt - new_origin).T) for a_pt in pt_set ])

        def inv_map_pts(pt_set):
            return np.array([ plane_rot_inv.dot(a_pt) + new_origin for a_pt in pt_set ])


        # For some reason this doesn't send exactly to z = 0; so keep track of error in z
        #avg_z_err = mapped_plane_pts[:, 2].mean()
        #print('Average z error on plane', avg_z_err)

        #print('After transform', mapped_plane_pts[:5])

        if ax is not None:
            # If PLT axis is passed, draw points

            grid = np.linspace(-3.0, 3.0, 50)
            plane_pts = np.array([ (x, y, plane(x, y)) for x in grid for y in grid ])
            mapped_plane_pts = map_pts(plane_pts)

            ax.scatter(*(A.T), s=40, color='green')
            ax.scatter( *(np.array([ pt ]).T), color='grey', s=100)
            ax.scatter(*(plane_pts.T), s=0.3, color='orange')
            ax.scatter(*(mapped_plane_pts.T), s=0.3, color='yellow')
            ax.scatter(*((map_pts(A)).T), s=30, color='cyan')

        # 2D projection
        pt_on_plane = map_pts([pt])[0]
        mapped_A = map_pts(A)
        res_2d = thin_single_pt_2d(pt_on_plane[:2], mapped_A[:, :2])
        if res_2d is None:
            print('Low correlation with %s:' % H)
            H += config.three_dim['r_step']
            continue

        # We make it back to a 3D on mapped plane (we put the original coordinate and not "0" 
        # as there might be a small error in z-coordinate when projecting (maybe pt_on_plane[2] /= 0)

        res_3d_mapped = np.hstack((res_2d[0], pt_on_plane[2]))
        #res_3d_mapped = np.hstack((res_2d, 0))
        res_3d = inv_map_pts([ res_3d_mapped ])[0]

        if ax is not None:
            ax.scatter( *(np.array([ res_3d ]).T), color='black', s=100)

    if res_2d is None:
        return None
    return res_3d


# Perform 3D curve "thinning" for all points
# Returns projections of all points
# Some points may be lost if they fail to reach correlation threshold
def thin_pt_cloud_3d(pts, ax=None):
    new_pts = []
    for pt in pts:
        res = thin_single_pt_3d(pt, pts, ax=ax)

        if res is None:
            if not config.three_dim['remove_low_correlation_pts']:
                # If not remove, simply put back the original pt.
                print("# Leaving in original")
                new_pts.append(pt)
            else:
                print("# Removing")
                # We ignore/remove this point
        else:
            new_pts.append(res)

    return np.array(new_pts)
