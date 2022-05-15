from BMDP import BMDPscenaraio
import numpy as np
import matplotlib.pyplot as plt


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

    belief_means = []
    belief_covs = []
    for i in range(T):
        print(i)
        bmdp.transit_belief_MDP(action)
        belief_means.append(bmdp.belief_mean)
        belief_covs.append(bmdp.belief_cov)

    plt.scatter(belief_means[0], belief_means[1])
    plt.show()
