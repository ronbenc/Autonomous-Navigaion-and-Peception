from BMDP import BMDPscenaraio
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    beacons = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0], [4.0, 0.0], [4.0, 4.0], [4.0, 8.0], [8.0, 0.0], [8.0, 4.0], [8.0, 8.0]], dtype=float)
    initial_belief_mean = np.array([0.0, 0.0], dtype=float)
    initial_belief_cov = np.eye(2, dtype=float)
    process_cov = 0.01 * np.eye(2, dtype=float)
    initial_gt = np.array([-0.5, -0.2], dtype=float)
    actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], 
                    [1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]], dtype=float)
    bmdp = BMDPscenaraio(initial_belief_mean, initial_belief_cov, np.eye(2), process_cov, beacons, actions, d=1, rmin=0.1)
    x_0_gt = np.array([-0.5, -0.2])

    # test
    T = 25
    action = np.array([0.5, 0.5])

    belief_means_x = [initial_belief_mean[0]]
    belief_means_y = [initial_belief_mean[1]]
    belief_covs = [initial_belief_cov]
    for i in range(T):
        bmdp.transit_belief_MDP(action)
        belief_means_x.append(bmdp.belief_mean[0])
        belief_means_y.append(bmdp.belief_mean[1])
        belief_covs.append(bmdp.belief_cov)

    plt.scatter(beacons[:, 0], beacons[:, 1])
    plt.scatter(belief_means_x, belief_means_y)
    plt.show()
