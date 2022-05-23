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

##### 2.b
def adjust_params():
    beacons = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0], [4.0, 0.0], [4.0, 4.0], [4.0, 8.0], [8.0, 0.0], [8.0, 4.0], [8.0, 8.0]], dtype=float)
    initial_belief_mean = np.array([0.0, 0.0], dtype=float)
    initial_belief_cov = np.eye(2, dtype=float)
    process_cov = 0.01 * np.eye(2, dtype=float)
    initial_gt = np.array([-0.5, -0.2], dtype=float)
    actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                        [1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [1 / np.sqrt(2), -1 / np.sqrt(2)], [-1 / np.sqrt(2), -1 / np.sqrt(2)], [0, 0]], dtype=float)
    x_0_gt = np.array([-0.5, -0.2], dtype=float)
    x_g = np.array([7, 7])
    horizions = [1 ,2 , 3]
    lambda_regs = [0.3,0.5, 0.8]
    number_of_samples = 4
    T = 10

    for horizion in horizions:
        for lambda_reg in lambda_regs:
            x_0_gt = np.array([-0.5, -0.2], dtype=float)
            ground_truths = [np.copy(x_0_gt)]
            observations = []
            bmdp = BMDPscenaraio(x_0_gt,np.copy(initial_belief_mean), np.copy(initial_belief_cov), np.eye(2),
                                 np.copy(process_cov), beacons, actions, d=1,
                                 rmin=0.1, lambda_reg=lambda_reg)
            belief_means, belief_covs = [np.copy(bmdp.belief_mean)], [np.copy(bmdp.belief_cov)]
            for i in range(T):
                action, val = bmdp.sparse_sampling(np.copy(x_g), number_of_samples, horizion,discount_factor=0.9)
                bmdp.sample_motion_model(action)
                ground_truths.append(np.copy(bmdp.x_gt))
                obs = bmdp.transit_belief_MDP(action)
                if obs is not None:
                    observations.append(obs)
                belief_means.append(np.copy(bmdp.belief_mean))
                belief_covs.append(np.copy(bmdp.belief_cov))
            plt.close()
            plt.title(f'horizion is {horizion} and lambda is {lambda_reg}')
            plt.scatter(beacons[:, 0], beacons[:, 1], label="beacons", marker='^')
            plt.scatter(x_g[0], x_g[1], label="goal", marker='*')
            plt.scatter(*zip(*ground_truths), label="ground truth")
            if observations:
                plt.scatter(*zip(*observations), label="observations")
            plt.scatter(*zip(*belief_means), label="belief means")
            plt.legend(loc='upper left')
            for i in range(T+1):
                plot_cov_ellipse(cov=belief_covs[i], pos=belief_means[i], alpha=0.1)
            plt.savefig(f'2_b_horizion_{horizion}_lambda_{lambda_reg}.png')
            plt.show()


##### 2.c
def run_simulation():
    beacons = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0], [4.0, 0.0], [4.0, 4.0], [4.0, 8.0], [8.0, 0.0], [8.0, 4.0], [8.0, 8.0]], dtype=float)
    initial_belief_mean = np.array([0.0, 0.0], dtype=float)
    initial_belief_cov = np.eye(2, dtype=float)
    process_cov = 0.01 * np.eye(2, dtype=float)
    initial_gt = np.array([-0.5, -0.2], dtype=float)
    actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                        [1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [1 / np.sqrt(2), -1 / np.sqrt(2)], [-1 / np.sqrt(2), -1 / np.sqrt(2)], [0, 0]], dtype=float)
    x_0_gt = np.array([-0.5, -0.2], dtype=float)
    x_g = np.array([7, 7])
    horizion = 15
    lambda_reg = 0.5
    number_of_samples = 4
    T = 15
    x_0_gt = np.array([-0.5, -0.2], dtype=float)
    ground_truths = [np.copy(x_0_gt)]
    observations = []
    bmdp = BMDPscenaraio(x_0_gt,np.copy(initial_belief_mean), np.copy(initial_belief_cov), np.eye(2),
                                 np.copy(process_cov), beacons, actions, d=1,
                                 rmin=0.1, lambda_reg=lambda_reg)
    belief_means, belief_covs = [np.copy(bmdp.belief_mean)], [np.copy(bmdp.belief_cov)]
    for i in range(T):
        action, val = bmdp.sparse_sampling(np.copy(x_g), number_of_samples, horizion,discount_factor=0.9)
        bmdp.sample_motion_model(action)
        ground_truths.append(np.copy(bmdp.x_gt))
        obs = bmdp.transit_belief_MDP(action)
        if obs is not None:
            observations.append(obs)
        belief_means.append(np.copy(bmdp.belief_mean))
        belief_covs.append(np.copy(bmdp.belief_cov))
        plt.close()
    plt.title(f'horizion is {horizion} and lambda is {lambda_reg}')
    plt.scatter(beacons[:, 0], beacons[:, 1], label="beacons", marker='^')
    plt.scatter(x_g[0], x_g[1], label="goal", marker='*')
    plt.scatter(*zip(*ground_truths), label="ground truth")
    if observations:
        plt.scatter(*zip(*observations), label="observations")
    plt.scatter(*zip(*belief_means), label="belief means")
    plt.legend(loc='upper left')
    for i in range(T+1):
            plot_cov_ellipse(cov=belief_covs[i], pos=belief_means[i], alpha=0.1)
    plt.savefig(f'2_c_horizion_{horizion}_lambda_{lambda_reg}.png')
    plt.show()

##### 2.d.e.f
def run_plots():
    beacons = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0], [4.0, 0.0], [4.0, 4.0], [4.0, 8.0], [8.0, 0.0], [8.0, 4.0], [8.0, 8.0]], dtype=float)
    initial_belief_mean = np.array([0.0, 0.0], dtype=float)
    initial_belief_cov = np.eye(2, dtype=float)
    process_cov = 0.01 * np.eye(2, dtype=float)
    initial_gt = np.array([-0.5, -0.2], dtype=float)
    actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                        [1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [1 / np.sqrt(2), -1 / np.sqrt(2)], [-1 / np.sqrt(2), -1 / np.sqrt(2)], [0, 0]], dtype=float)
    x_0_gt = np.array([-0.5, -0.2], dtype=float)
    x_g = np.array([7, 7])

    horizion = 3
    lambda_reg = 0.5
    number_of_samples = 4
    T = 10
    x_0_gt = np.array([-0.5, -0.2], dtype=float)
    ground_truths = [np.copy(x_0_gt)]
    observations = []
    bmdp = BMDPscenaraio(x_0_gt,np.copy(initial_belief_mean), np.copy(initial_belief_cov), np.eye(2),
                                 np.copy(process_cov), beacons, actions, d=1,
                                 rmin=0.1, lambda_reg=lambda_reg)
    belief_means, belief_covs = [np.copy(bmdp.belief_mean)], [np.copy(bmdp.belief_cov)]
    for i in range(T):
        action, val = bmdp.sparse_sampling(np.copy(x_g), number_of_samples, horizion,discount_factor=0.9)
        bmdp.sample_motion_model(action)
        ground_truths.append(np.copy(bmdp.x_gt))
        obs = bmdp.transit_belief_MDP(action)
        if obs is not None:
            observations.append(obs)
        belief_means.append(np.copy(bmdp.belief_mean))
        belief_covs.append(np.copy(bmdp.belief_cov))
        plt.close()

    # 2.d --- plots ground truth, beacons
    plt.title(f'ground truth beacons')
    plt.scatter(beacons[:, 0], beacons[:, 1], label="beacons", marker='^')
    plt.scatter(x_g[0], x_g[1], label="goal", marker='*')
    plt.scatter(*zip(*ground_truths), label="ground truth")
    plt.legend(loc='upper left')
    plt.savefig(f'ground_truth_beacons.png')
    plt.show()

    # 2.e --- plots ground truth, beacons, observations
    plt.close()
    plt.title(f'ground truth beacons observation')
    plt.scatter(beacons[:, 0], beacons[:, 1], label="beacons", marker='^')
    plt.scatter(x_g[0], x_g[1], label="goal", marker='*')
    plt.scatter(*zip(*ground_truths), label="ground truth")

    if observations:
        plt.scatter(*zip(*observations), label="observations")
    plt.legend(loc='upper left')
    plt.savefig(f'ground_truth_beacons_observations.png')
    plt.show()

    # 2.f --- plots ground truth, beacons, belife
    plt.close()
    plt.title(f'ground truth beacons belief')
    plt.scatter(beacons[:, 0], beacons[:, 1], label="beacons", marker='^')
    plt.scatter(x_g[0], x_g[1], label="goal", marker='*')
    plt.scatter(*zip(*ground_truths), label="ground truth")
    plt.scatter(*zip(*belief_means), label="belief means",color='purple')
    plt.legend(loc='upper left')
    plt.savefig(f'ground_truth_beacons_belief.png')
    plt.show()





if __name__ == '__main__':

    ### 2.b
    # adjust_params()

    ### 2.c
    # run_simulation()

    ### 2.d.e.f
    run_plots()