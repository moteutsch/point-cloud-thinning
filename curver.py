import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize, curve_fit
import scipy.stats

p = [0, 0, 1]

#pts = [ np.polynomial.polynomial.polyval(t, p) for t in np.linspace(-10, 10, 20) ]
def pts(how_many_pts):
    return np.array([ (np.sin(t), np.cos(t)) for t in np.linspace(0, 2 * np.pi, how_many_pts) ])
print(pts(50))

def pts_with_noise(n):
    return pts(n) + np.random.normal(0, 0.05, size=(n, 2))

def ball(center, r, all_pts):
    return np.array([ pt for pt in all_pts if np.linalg.norm(center - pt) < r ])


def linear_regression_line(pts, p):
    x = pts[:, 0]
    y = pts[:, 1]
    p0 = 0, 1 # initial guess

    #popt, pcov = curve_fit(f, x, y, p0, sigma=sigma, absolute_sigma=True)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    x = scipy.stats.linregress(x,y)
    print(x)
    return x

    popt, pcov = curve_fit(f, x, y, p0)
    print(pot, pcov)
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
    print((0, i))

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


#def quadractic_regression_curve(pts, p):



    # TODO: Inverse operation

# Algorithm2 = Collect2 (non-iterative version--contents of loop)
def local_regression_line(pt, r, corr_tol, all_pts):
    A = ball(pt, r, all_pts)


p = pts_with_noise(100)



plt.scatter(*(p.T), c='blue')
b = ball(p[0], 0.2, p)
plt.scatter(*(b.T), c='red')

rl = linear_regression_line(b, p[0])
(slope, intercept, _, _, _) = rl
print(slope)
r = np.array([ (t, slope * t + intercept) for t in np.linspace(-0.5, 0.5, 100) ])

plt.scatter(*(r.T), c='orange', s=0.1)

rho, rot_pts = pt_correlations(b, p[0])
print("Correlation", rho)

#G = nx.dodecahedral_graph()
#nx.draw(G)
#plt.show()

#plt.scatter(*(rot_pts.T), c='green')
plt.show()
