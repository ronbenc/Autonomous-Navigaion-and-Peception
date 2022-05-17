from msilib.schema import Class
from tkinter.messagebox import NO
import numpy as np

class BMDPscenaraio:
    def __init__(self, belief_mean, belief_cov, F, process_cov, beacons, actions, d, rmin, lambda_reg):
        self.belief_mean = belief_mean
        self.belief_cov = belief_cov
        self.F = F
        self.process_cov = process_cov
        self.beacons = beacons # 2XN
        self.actions = actions
        self.d = d
        self.rmin=rmin
        self.lambda_reg = lambda_reg
        self.last_obs_cov = None


    def _generate_observation_from_beacons(self, state):
        distances_from_beacons = np.linalg.norm(self.beacons-state, axis=1)
        min_dist = np.min(distances_from_beacons)
        if min_dist <= self.d:
            obs_cov = 0.01*max(min_dist, self.rmin)*np.eye(2)
            self.last_obs_cov = obs_cov
            obs = state + np.random.multivariate_normal(np.zeros_like(state), obs_cov)
            return obs
        else:
            return None
    

    def _propogate_belief(self, action):
        self.belief_mean = (self.F @ self.belief_mean) + action
        self.belief_cov = (self.F @ self.belief_cov @ self.F.T) + self.process_cov

 
    def _update_belief(self, obs):
        assert self.last_obs_cov is not None
        k = self.belief_cov @ np.linalg.inv(self.belief_cov + self.last_obs_cov)
        self.belief_mean = self.belief_mean + (k @ (obs-self.belief_mean))
        self.belief_cov = (np.eye(2) - k) @ self.belief_cov


    def transit_belief_MDP(self, action):
        # propogate belief using action and transition model (KF)
        self._propogate_belief(action)

        # sample a state from propogated belief, create observation and if not null update belief
        sampled_state = np.random.multivariate_normal(self.belief_mean, self.belief_cov)
        obs = self._generate_observation_from_beacons(sampled_state)
        if obs is not None:
            self._update_belief(obs)
    

    def _cost(self, goal_state):
        cost = np.linalg.norm(self.belief_mean - goal_state) + lambda_reg*np.linalg.det(self.belief_cov)
        return cost


class BMPD_planner:
    def __init__(self, goal_state, number_of_samples) -> None:
        self.goal_state = goal_state
        self.number_of_samples = number_of_samples

    
    def sparse_sampling(self, goal_state, number_of_samples, horizon):
        if horizon <= 0:
            return None, None
        best_action, best_value = None, -np.inf
        curr_belief_mean, curr_belief_cov = self.belief_mean, self.belief_cov
        for action in self.actions:
            q = 0
            for i in range(number_of_samples):
                self.transit_belief_MDP(action)
                cost = self._cost(goal_state)
                selected_action, value = self.sparse_sampling(goal_state, number_of_samples, horizon-1)



