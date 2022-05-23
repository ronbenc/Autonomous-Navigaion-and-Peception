from BMDP import BMDPscenaraio
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """

    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    print(vals)
    width, height = 2 * nstd * np.sqrt(vals)
    print(width,height)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,color='red', **kwargs)

    ax.add_artist(ellip)
    return ellip




if __name__ == '__main__':
    beacons = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0], [4.0, 0.0], [4.0, 4.0], [4.0, 8.0], [8.0, 0.0], [8.0, 4.0], [8.0, 8.0]], dtype=float)
    initial_belief_mean = np.array([0.0, 0.0], dtype=float)
    initial_belief_cov = np.eye(2, dtype=float)
    process_cov = 0.01 * np.eye(2, dtype=float)
    initial_gt = np.array([-0.5, -0.2], dtype=float)

    bmdp = BMDPscenaraio(initial_belief_mean, initial_belief_cov, np.eye(2), process_cov, beacons, d=1, rmin=0.1)

    # test
    T = 10
    action = np.array([0.5, 0.5])

    belief_means_x = []
    belief_means_y = []
    belief_covs = []
    beacons_x = []
    beacons_y = []
    for i in range(T):
        bmdp.transit_belief_MDP(action)
        belief_means_x.append(bmdp.belief_mean[0])
        belief_means_y.append(bmdp.belief_mean[1])
        belief_covs.append(bmdp.belief_cov)
    for i in range(len(beacons)):
        beacons_x.append(beacons[i][0])
        beacons_y.append(beacons[i][1])

    print(belief_covs[0])
    print([belief_means_x[0],belief_means_y[0]])
    plt.scatter(belief_means_x, belief_means_y)
    plt.scatter(belief_means_x, belief_means_y)
    for i in range(T):
        plot_cov_ellipse(cov=belief_covs[i],pos=[belief_means_x[i],belief_means_y[i]],alpha=0.1)

    plt.scatter(beacons_x,beacons_y)
    plt.show()

