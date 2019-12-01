import curver
import matplotlib.pyplot as plt
import numpy as np

########################

NUM_PTS = 100
CURVE = curver.circle
#CURVE = lambda x: polynomial_curve([1, 0, 0], x)

plt.gca().set_aspect('equal', adjustable='box')
        
plt.scatter(*(CURVE(NUM_PTS).T))

all_pts = curver.pts_with_noise(CURVE, NUM_PTS, 0.05)

plt.scatter(*(all_pts).T, alpha=0.5, c='green')
plt.show()
#all_pts = ball_c(all_pts[50], 0.3, all_pts)

#all_pts = np.setdiff1d(all_pts, b)
#print(all_pts)



# Simple collect1

plt.scatter(*(all_pts.T), c='blue')
b = curver.ball(all_pts[0], 0.3, all_pts)
plt.scatter(*(b.T), c='red')

# Collect2

r_step = 0.03
corr_tol = 0.7
b = curver.collect2(all_pts[0], 0.25, corr_tol, r_step, all_pts)
plt.scatter(*(b.T), c='yellow')

# Linera regression test


rl = curver.linear_regression_line(b, all_pts[0])
(slope, intercept, _, _, _) = rl
#print(slope)
r = np.array([ (t, slope * t + intercept) for t in np.linspace(-0.5, 0.5, 100) ])

plt.scatter(*(r.T), c='orange', s=0.1)

# Pt. correlations test

rho, rot_pts = curver.pt_correlations(b, all_pts[0])
#print("Correlation", rho)

# Quadratic regression curve

curver.quadractic_regression_curve(b, all_pts[0], plot=True)

#G = nx.dodecahedral_graph()
#nx.draw(G)
#plt.show()

#plt.scatter(*(rot_pts.T), c='green')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

thinned_p = curver.thin_pt_cloud(all_pts)
plt.scatter(*(thinned_p.T), alpha=0.5, c='brown')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.scatter(*(thinned_p.T), alpha=1, c='brown')

thinned_p2 = curver.thin_pt_cloud(thinned_p)

plt.scatter(*(thinned_p2.T), alpha=0.4, c='green')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

n = thinned_p2 
for i in range(3):
    n = curver.thin_pt_cloud(n)

    plt.scatter(*(CURVE(NUM_PTS).T), alpha=0.3)
    plt.scatter(*(all_pts.T), alpha=0.3, c='green')
    plt.scatter(*(n.T), alpha=1, c='orange')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

