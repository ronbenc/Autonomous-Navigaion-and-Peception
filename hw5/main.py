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
    actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], 
                    [1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)], [0, 0]], dtype=float)
    bmdp = BMDPscenaraio(initial_belief_mean, initial_belief_cov, np.eye(2), process_cov, beacons, actions, d=1, rmin=0.1, lambda_reg=0.3)
    x_0_gt = np.array([-0.5, -0.2], dtype=float)

    # test
    # T = 25
    # action = np.array([0.5, 0.5])

    # belief_means_x = []
    # belief_means_y = []
    # belief_covs = []
    # bmdp.transit_belief_MDP( action = np.array([0.0, 0.0]))
    # for i in range(T):
    #     bmdp.transit_belief_MDP(action)
    #     belief_means_x.append(bmdp.belief_mean[0])
    #     belief_means_y.append(bmdp.belief_mean[1])
    #     belief_covs.append(bmdp.belief_cov)

    # plt.scatter(belief_means_x, belief_means_y)
    # plt.scatter(beacons[:, 0], beacons[:, 1])
    # for i in range(T):
    #     plot_cov_ellipse(cov=belief_covs[i],pos=[belief_means_x[i],belief_means_y[i]],alpha=0.1)

    # plt.show()

    # part c + d + e
    # x_g = np.random.uniform(0, 8, 2)
    x_g = np.array([8, 1])
    horizion = 3
    number_of_samples = 2
    T = 10

    ground_truths = [x_0_gt]
    observations = []
    belief_means, belief_covs = [], []

    for i in range(T):
        action, val = bmdp.sparse_sampling(x_g, number_of_samples, horizion)
        ground_truths.append(bmdp.sample_motion_model(action, ground_truths[-1]))
        obs = bmdp.transit_belief_MDP(action)
        if obs is not None:
            observations.append(obs)
        belief_means.append(bmdp.belief_mean)
        belief_covs.append(bmdp.belief_cov)


    plt.scatter(beacons[:, 0], beacons[:, 1], label="beacons")
    plt.scatter(*zip(*ground_truths), label="ground truth")
    plt.scatter(*zip(*observations), label="observations")
    plt.scatter(*zip(*belief_means), label="belief means")
    plt.legend(loc='upper left')

    for i in range(T):
        plot_cov_ellipse(cov=belief_covs[i], pos=belief_means[i],alpha=0.1)
    plt.show()





