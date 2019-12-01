import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize, curve_fit
import scipy.stats

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
        rho, rot_pts = pt_correlations(A, pt)

        H += r_step
        #print("Iteration: %d; Rho: %s; H = %s" % (i, rho, H))

    return A


def thin_pt_cloud(pts):
    r = 0.35
    r_step = 0.03
    corr_tol = 0.7

    new_pts = []
    for p in pts:
        A = collect2(p, r, corr_tol, r_step, pts)
        if A is None:
            print("### Warning: Could not find nbhd with sufficient correlation")
            # Simply don't modify p
            #new_pts.append(p)
        else:
            curve, p_proj = quadractic_regression_curve(A, p)
            new_pts.append(p_proj)

    return np.array(new_pts)

        
########################

NUM_PTS = 100
CURVE = circle
#CURVE = lambda x: polynomial_curve([1, 0, 0], x)

plt.gca().set_aspect('equal', adjustable='box')
        
plt.scatter(*(CURVE(NUM_PTS).T))

all_pts = pts_with_noise(CURVE, NUM_PTS, 0.05)

plt.scatter(*(all_pts).T, alpha=0.5, c='green')
plt.show()
#all_pts = ball_c(all_pts[50], 0.3, all_pts)

#all_pts = np.setdiff1d(all_pts, b)
#print(all_pts)



# Simple collect1

plt.scatter(*(all_pts.T), c='blue')
b = ball(all_pts[0], 0.3, all_pts)
plt.scatter(*(b.T), c='red')

# Collect2

r_step = 0.03
corr_tol = 0.7
b = collect2(all_pts[0], 0.25, corr_tol, r_step, all_pts)
plt.scatter(*(b.T), c='yellow')

# Linera regression test


rl = linear_regression_line(b, all_pts[0])
(slope, intercept, _, _, _) = rl
#print(slope)
r = np.array([ (t, slope * t + intercept) for t in np.linspace(-0.5, 0.5, 100) ])

plt.scatter(*(r.T), c='orange', s=0.1)

# Pt. correlations test

rho, rot_pts = pt_correlations(b, all_pts[0])
#print("Correlation", rho)

# Quadratic regression curve

quadractic_regression_curve(b, all_pts[0], plot=True)

#G = nx.dodecahedral_graph()
#nx.draw(G)
#plt.show()

#plt.scatter(*(rot_pts.T), c='green')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

thinned_p = thin_pt_cloud(all_pts)
plt.scatter(*(thinned_p.T), alpha=0.5, c='brown')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.scatter(*(thinned_p.T), alpha=1, c='brown')

thinned_p2 = thin_pt_cloud(thinned_p)

plt.scatter(*(thinned_p2.T), alpha=0.4, c='green')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

n = thinned_p2 
for i in range(3):
    n = thin_pt_cloud(n)

    plt.scatter(*(CURVE(NUM_PTS).T), alpha=0.3)
    plt.scatter(*(all_pts.T), alpha=0.3, c='green')
    plt.scatter(*(n.T), alpha=1, c='orange')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
